import cv2
from ultralytics import YOLO
import time

# Load YOLOv8n model (first run will auto-download)
model = YOLO("yolov8n.pt")

# Open default camera (0 = first camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not access camera.")
    exit()

# Window name
cv2.namedWindow("PiVision Detect", cv2.WINDOW_NORMAL)

# Confidence threshold
CONF_THRESHOLD = 0.5

print("✅ Starting live detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame.")
        break

    # Run detection
    results = model(frame, verbose=False)

    detected_objects = []

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        detected_objects.append(label)

    # Show detected labels on screen
    if detected_objects:
        unique_objs = ", ".join(sorted(set(detected_objects)))
        cv2.putText(frame, f"Detected: {unique_objs}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Show frame
    cv2.imshow("PiVision Detect", frame)

    # Exit if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Detection stopped.")
