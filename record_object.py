from datetime import datetime
import csv
import os

class ObjectRecorder:
    def __init__(self, filename="Detections.csv"):
        os.makedirs("Detections", exist_ok=True)
        self.filepath = os.path.join("Detections", filename)

        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Object Name"])
        print(f"Logging detections to : {self.filepath}")
    
    def record(self, object_name):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, object_name])
