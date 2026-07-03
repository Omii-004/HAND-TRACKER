#not using this right now, but keeping it for future reference. This is a simple data logger that saves the hand pose sequences to a CSV file for training a gesture recognition model.
import csv
import os
import time

class DataLogger:
    def __init__(self, filepath="gesture_dataset.csv", max_frames=30):
        self.filepath = filepath
        self.max_frames = max_frames
        self._initialize_csv()

    def _initialize_csv(self):
        # Create the file and write headers if it doesn't exist
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                headers = ["timestamp", "label"]
                for i in range(self.max_frames):
                    headers.extend([f"x{i+1}", f"y{i+1}"])
                writer.writerow(headers)
            print(f"[LOG] Created new dataset file: {self.filepath}")

    def log_sequence(self, label, history_deque):
        history = list(history_deque)
        
        # ML models need fixed-size inputs. Pad with the last known position if short.
        while len(history) < self.max_frames:
            if len(history) > 0:
                history.append(history[-1])
            else:
                history.append((0, 0))

        # Flatten the (x, y) tuples into a single 1D list
        flat_coords = []
        for x, y in history:
            flat_coords.extend([x, y])

        # Append the new row to the CSV
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Use milliseconds for timestamp
            row = [int(time.time() * 1000), label] + flat_coords
            writer.writerow(row)
        
        print(f"[DATA PIPELINE] Saved sequence for: {label}")