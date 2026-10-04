from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("./last.pt")

# Run detection on an image
while True :
    results = model.predict(source="./image.jpg", show=True, save=True)

# Optional: print detections with class names
for r in results:
    for box in r.boxes:
        cls_idx = int(box.cls[0])
        conf = box.conf[0]
        name = r.names[cls_idx]
        print(f"Detected: {name} ({conf:.2f})")