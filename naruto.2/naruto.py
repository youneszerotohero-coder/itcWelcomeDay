import cv2
import mediapipe as mp
import time
import pickle
import collections
import numpy as np

MODEL_PATH  = "hand_landmarker.task"
SIGN_MODEL  = "sign_model.pkl"

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
print("Model loaded!")

# ── MediaPipe setup ──
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
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


def extract_features(result):
    """Extract same 84 features used during training."""
    data  = []
    hands = result.hand_landmarks if result.hand_landmarks else []
    for hand_idx in range(2):
        if hand_idx < len(hands):
            for lm in hands[hand_idx]:
                data += [round(lm.x, 4), round(lm.y, 4)]
        else:
            data += [0.0] * 42
    return data


# ── Prediction smoothing (avoid flickering) ──
PRED_BUFFER    = collections.deque(maxlen=8)
flash_timer    = 0
FLASH_DURATION = 2.0
start_time     = time.time()

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        small     = cv2.resize(frame, (320, 240))
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
        timestamp = int((time.time() - start_time) * 1000)
        result    = landmarker.detect_for_video(mp_image, timestamp)

        # ── Draw smoothed landmarks ──
        if result.hand_landmarks:
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

        # ── Predict sign using trained model ──
        if result.hand_landmarks and len(result.hand_landmarks) == 2:
            features = extract_features(result)
            prediction = classifier.predict([features])[0]
            PRED_BUFFER.append(prediction)

            # Only trigger if last 6 out of 8 frames say shadow_clone
            shadow_votes = list(PRED_BUFFER).count("shadow_clone")
            if shadow_votes >= 6:
                flash_timer = time.time()
        else:
            PRED_BUFFER.clear()

        # ── Show Shadow Clone Jutsu message ──
        if time.time() - flash_timer < FLASH_DURATION:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.6, 0, frame)

            text       = "SHADOW CLONE JUTSU!"
            font       = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.2
            thickness  = 3
            text_size  = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x     = (w - text_size[0]) // 2
            text_y     = h // 2

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

        cv2.imshow("Shadow Clone Jutsu Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()