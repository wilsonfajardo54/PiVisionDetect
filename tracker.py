"""
Vision Object Tracker
Author: Wilson Fajardo 
Class: Communicating Robots
Platform: Laptop
Description:
    Simple real-time object tracking using OpenCV (CSRT Tracker).
    Press 's' to select an object to track, 'q' to quit.
"""

import cv2
import time


class VisionTracker:
    def __init__(self, camera_index=0):
        """
        Initialize camera and tracker.
        camera_index = 0 for default webcam or Pi camera.
        """
        self.cap = cv2.VideoCapture(camera_index)
        self.tracker = None
        self.tracking = False
        self.window_name = "PiVision Object Tracker"

        if not self.cap.isOpened():
            raise IOError("Cannot open camera. Check connection or index.")

        # Create display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        print("Vision Object Tracker Initialized.")
        print("Press 's' to select an object to track, 'q' to quit.\n")

    def select_object(self, frame):
        """
        Allow user to select ROI (object) to track.
        """
        bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False)
        cv2.destroyWindow("Select Object to Track")
        if bbox != (0, 0, 0, 0):
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame, bbox)
            self.tracking = True
            print("Tracking started.")
        else:
            print("No region selected.")

    def run(self):
        """
        Main loop: Capture frames, update tracker, show FPS.
        """
        prev_time = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # Compute FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time else 0
            prev_time = current_time

            key = cv2.waitKey(1) & 0xFF

            # Press 's' to select an object
            if key == ord('s'):
                self.select_object(frame)

            # Update tracker if active
            if self.tracking and self.tracker is not None:
                success, bbox = self.tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Tracking...", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Lost Object", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    self.tracking = False

            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Show video feed
            cv2.imshow(self.window_name, frame)

            # Press 'q' to quit
            if key == ord('q'):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Tracker stopped.")


if __name__ == "__main__":
    tracker = VisionTracker(camera_index=0)
    tracker.run()
