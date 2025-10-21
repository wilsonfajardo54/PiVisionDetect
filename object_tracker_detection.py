"""
PiVision Object Tracker + Object Name Detection
Author: Wilson Fajardo
Platform: Raspberry Pi or Laptop
Description:
    - Uses YOLOv8 for real-time object detection
    - Lets user select an object (ROI) with 's' key
    - Starts tracking that object and prints its name
    - Uses OpenVC for YOLO model. 
"""

import cv2
import time
from ultralytics import YOLO

class PiVisionTracker:
    def __init__(self, model_path="yolov8n.pt", camera_index=0, conf_threshold=0.5):
        """Initialize camera, model, and window."""
        self.cap = cv2.VideoCapture(camera_index)
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tracker = None
        self.tracking = False
        self.tracked_label = None
        self.last_detections = []

        if not self.cap.isOpened():
            raise IOError("Cannot open camera. Check connection or index.")

        cv2.namedWindow("PiVision Object Tracker", cv2.WINDOW_NORMAL)
        print("PiVision Object Tracker Initialized.")
        print("Press 's' to select an object, 'q' to quit.\n")

    def detect_objects(self, frame):
        """Run YOLO detection on the frame."""
        results = self.model(frame, verbose=False)
        detections = []

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            cls = int(box.cls[0])
            label = self.model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((label, (x1, y1, x2, y2)))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.last_detections = detections
        return frame

    def select_object(self, frame):
        """Allow user to select an object and start tracking it."""
        print("Select a region on the video...")
        bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False)
        cv2.destroyWindow("Select Object to Track")

        if bbox == (0, 0, 0, 0):
            print("⚠️ No region selected.")
            return

        # Match selection to YOLO detections
        (x, y, w, h) = bbox
        x2, y2 = x + w, y + h
        selected_label = "Unknown"
        for (label, (dx1, dy1, dx2, dy2)) in self.last_detections:
            if (x >= dx1 and y >= dy1) and (x2 <= dx2 and y2 <= dy2):
                selected_label = label
                break

        # Create CSRT tracker
        if hasattr(cv2, "legacy"):
            self.tracker = cv2.legacy.TrackerCSRT_create()
        else:
            self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.tracking = True
        self.tracked_label = selected_label
        print(f"Tracking started — Object: {self.tracked_label}")

    def run(self):
        """Main loop for detection and tracking."""
        prev_time = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Frame capture failed.")
                break

            # Compute FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time else 0
            prev_time = current_time

            key = cv2.waitKey(1) & 0xFF

            # Press 's' to select object
            if key == ord('s'):
                self.select_object(frame)

            # Run YOLO detection if not tracking
            if not self.tracking:
                frame = self.detect_objects(frame)

            # Update tracker
            if self.tracking and self.tracker is not None:
                success, bbox = self.tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (255, 255, 0), 2)
                    cv2.putText(frame, f"Tracking: {self.tracked_label}",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 0), 2)
                else:
                    print(f"Lost track of {self.tracked_label}")
                    self.tracking = False
                    self.tracked_label = None

            # FPS overlay
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Show frame
            cv2.imshow("PiVision Object Tracker", frame)

            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("Tracker stopped.")


if __name__ == "__main__":
    tracker = PiVisionTracker(model_path="yolov8n.pt", camera_index=0)
    tracker.run()


if __name__ == "__main__":
    detector = PiVisionDetector(model_path="yolov8n.pt", conf_threshold=0.5)
    detector.run()

