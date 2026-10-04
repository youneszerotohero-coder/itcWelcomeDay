"""
collect_data.py — Shadow Clone Sign Data Collector
===================================================
Controls:
  S — save current landmarks as "shadow_clone"
  O — save current landmarks as "other"
  Q — quit
"""

import cv2
import mediapipe as mp
import csv
import time
import os

MODEL_PATH = "hand_landmarker.task"
CSV_FILE   = "sign_data.csv"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        for hand in range(2):
            for i in range(21):
                header += [f"h{hand}_x{i}", f"h{hand}_y{i}"]
        writer.writerow(header)
    print(f"Created {CSV_FILE}")


def extract_landmarks(result):
    data  = []
    hands = result.hand_landmarks if result.hand_landmarks else []
    for hand_idx in range(2):
        if hand_idx < len(hands):
            for lm in hands[hand_idx]:
                data += [round(lm.x, 4), round(lm.y, 4)]
        else:
            data += [0.0] * 42
    return data


def save_sample(label, data):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label] + data)


def count_samples():
    counts = {"shadow_clone": 0, "other": 0}
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
    return counts


start_time    = time.time()
last_saved    = 0
SAVE_COOLDOWN = 0.3

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\n===== DATA COLLECTOR =====")
    print("S = save as shadow_clone")
    print("O = save as other")
    print("Q = quit")
    print("==========================\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame     = cv2.flip(frame, 1)
        h, w      = frame.shape[:2]
        small     = cv2.resize(frame, (320, 240))
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
        timestamp = int((time.time() - start_time) * 1000)
        result    = landmarker.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                for s, e in HAND_CONNECTIONS:
                    cv2.line(frame, points[s], points[e], (0, 255, 0), 2)
                for (x, y) in points:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        counts     = count_samples()
        hand_count = len(result.hand_landmarks) if result.hand_landmarks else 0

        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"Hands: {hand_count}   shadow_clone: {counts['shadow_clone']}   other: {counts['other']}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(frame, (0, h - 45), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "S = shadow_clone  |  O = other  |  Q = quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Data Collector — Shadow Clone", frame)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        if now - last_saved > SAVE_COOLDOWN:
            if key == ord('s'):
                if hand_count == 2:
                    data = extract_landmarks(result)
                    save_sample("shadow_clone", data)
                    last_saved = now
                    print(f"  ✓ Saved shadow_clone #{counts['shadow_clone'] + 1}")
                else:
                    print("  ✗ Need BOTH hands visible!")

            elif key == ord('o'):
                data = extract_landmarks(result)
                save_sample("other", data)
                last_saved = now
                print(f"  ✓ Saved other #{counts['other'] + 1}")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print(f"\nDone! Final counts: {count_samples()}")
print(f"Data saved to: {CSV_FILE}")