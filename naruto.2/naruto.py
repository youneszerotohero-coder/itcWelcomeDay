import cv2
import mediapipe as mp
import time
import pickle
import collections
import threading
import numpy as np

from clone_effect import CloneEffect
from sign_features import (hands_from_result, features, geometry, geometry_ok, is_crossed,
                           MAX_FINGER_GAP, MIN_CROSS_ANGLE, MIN_INDEX_EXT,
                           MAX_OPEN_EXT)

MODEL_PATH  = "hand_landmarker.task"
SIGN_MODEL  = "sign_model.pkl"
CONFIDENCE  = 0.6  # min model probability for a frame to count as the sign

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# ── Load trained model ──
with open(SIGN_MODEL, "rb") as f:
    classifier = pickle.load(f)
# the first prediction starts the model's worker threads (~1.5 s) — do it now,
# not the first time you make the sign
classifier.predict_proba(np.zeros((1, classifier.n_features_in_)))
print("Model loaded!")

# ── MediaPipe setup ──
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    # crossed hands overlap; lower thresholds keep both hands tracked
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

# ── Smoothing ──
SMOOTH_FRAMES   = 5
landmark_buffer = [
    collections.deque(maxlen=SMOOTH_FRAMES),
    collections.deque(maxlen=SMOOTH_FRAMES),
]

def smooth_landmarks(result, hand_idx, w, h):
    if not result.hand_landmarks or hand_idx >= len(result.hand_landmarks):
        landmark_buffer[hand_idx].clear()
        return None
    points = [(lm.x, lm.y) for lm in result.hand_landmarks[hand_idx]]
    landmark_buffer[hand_idx].append(points)
    avg_points = []
    for i in range(21):
        avg_x = sum(f[i][0] for f in landmark_buffer[hand_idx]) / len(landmark_buffer[hand_idx])
        avg_y = sum(f[i][1] for f in landmark_buffer[hand_idx]) / len(landmark_buffer[hand_idx])
        avg_points.append((int(avg_x * w), int(avg_y * h)))
    return avg_points


def is_shadow_clone(result, aspect):
    """Model must be confident AND the fingers must actually touch and cross.
    Returns (ok, debug text explaining the decision)."""
    h0, h1 = hands_from_result(result, aspect)[:2]
    g = geometry(h0, h1)
    shape_ok = geometry_ok(g)
    p = 0.0
    if shape_ok:  # only pay for the model when the geometry already looks right
        proba = classifier.predict_proba([features(h0, h1)])[0]
        p = proba[list(classifier.classes_).index("shadow_clone")]
    checks = [
        (f"gap {g['finger_gap']:.2f}",    g["finger_gap"] <= MAX_FINGER_GAP),
        (f"angle {g['angle']:.0f}",       g["angle"] >= MIN_CROSS_ANGLE),
        (f"index {g['min_index_ext']:.2f}", g["min_index_ext"] >= MIN_INDEX_EXT),
        (f"curl {g['open_ext']:.2f}",     g["open_ext"] <= MAX_OPEN_EXT),
        (f"cross {g['cross_0']:.2f}/{g['cross_1']:.2f}", is_crossed(g)),
        (f"model {p:.2f}" if shape_ok else "model -", p >= CONFIDENCE),
    ]
    ok = all(passed for _, passed in checks)
    return ok, checks


class LatestFrame:
    """Reads the camera on its own thread and keeps only the newest frame.
    Without this, frames queue up whenever processing is slower than the
    camera, and the video you see falls further and further behind."""

    def __init__(self, cap):
        self.cap     = cap
        self.frame   = None
        self.ok      = True
        self.new     = threading.Event()
        self.running = True
        self.thread  = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            self.ok, frame = self.cap.read()
            if not self.ok:
                self.new.set()
                break
            self.frame = frame
            self.new.set()

    def read(self):
        self.new.wait()
        self.new.clear()
        return self.ok, self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)  # let it finish before the camera is released


# ── Prediction smoothing (avoid flickering) ──
VOTES_NEEDED   = 4      # sign must be seen in 4 of the last 8 frames
PRED_BUFFER    = collections.deque(maxlen=8)
debug_checks   = []
show_debug     = True   # press D to toggle the debug line
flash_timer    = 0
FLASH_DURATION = 2.0
start_time     = time.time()
clones         = CloneEffect()  # press C to make the clones vanish early

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera   = LatestFrame(cap)
    fps      = 0.0
    last_t   = time.time()

    while cap.isOpened():
        success, frame = camera.read()
        if not success:
            break
        now    = time.time()
        fps    = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)  # smoothed
        last_t = now

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        small     = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
        timestamp = int((time.time() - start_time) * 1000)
        result    = landmarker.detect_for_video(mp_image, timestamp)

        # ── Predict sign using trained model ──
        # (a frame where one crossed hand is briefly lost counts as a miss
        #  instead of wiping the whole vote)
        if result.hand_landmarks and len(result.hand_landmarks) == 2:
            ok, debug_checks = is_shadow_clone(result, w / h)
            PRED_BUFFER.append("shadow_clone" if ok else "other")
        else:
            PRED_BUFFER.append("other")
            debug_checks = []

        if list(PRED_BUFFER).count("shadow_clone") >= VOTES_NEEDED:
            flash_timer = time.time()
            clones.trigger()
            PRED_BUFFER.clear()

        # ── Clones (uses the same RGB frame the hand tracker saw) ──
        frame = clones.apply(frame, small)

        # ── Draw smoothed landmarks ──
        if result.hand_landmarks and not clones.active:
            for i in range(len(result.hand_landmarks)):
                points = smooth_landmarks(result, i, w, h)
                if points is None:
                    continue
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, points[start_idx], points[end_idx],
                             (0, 255, 0), 2)
                for (x, y) in points:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                if result.handedness and i < len(result.handedness):
                    label = result.handedness[i][0].display_name
                    cv2.putText(frame, label,
                                (points[0][0] - 30, points[0][1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # ── Show Shadow Clone Jutsu message ──
        if time.time() - flash_timer < FLASH_DURATION:
            text       = "SHADOW CLONE JUTSU!"
            font       = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.2
            thickness  = 3
            text_size  = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x     = (w - text_size[0]) // 2
            text_y     = 85  # top of the screen, so it doesn't cover the clones

            cv2.putText(frame, text, (text_x + 3, text_y + 3),
                        font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, text, (text_x, text_y),
                        font, font_scale, (0, 140, 255), thickness)

            sub      = "Kage Bunshin no Jutsu!"
            sub_size = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            sub_x    = (w - sub_size[0]) // 2
            cv2.putText(frame, sub, (sub_x, text_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ── Hand count ──
        hand_count = len(result.hand_landmarks) if result.hand_landmarks else 0
        cv2.putText(frame, f"Hands: {hand_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ── Debug: each check in green (pass) or red (fail) ──
        if show_debug:
            x = 10
            for text, passed in debug_checks:
                color = (0, 220, 0) if passed else (0, 0, 255)
                cv2.putText(frame, text, (x, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] + 14

        cv2.imshow("Shadow Clone Jutsu Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            show_debug = not show_debug
        if key == ord('c'):
            clones.dismiss()

    camera.stop()
    cap.release()
    cv2.destroyAllWindows()
    clones.close()