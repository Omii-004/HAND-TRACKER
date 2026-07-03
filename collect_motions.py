import cv2
import csv
import os
import time
from collections import deque
from hand_tracker import HandTracker
from utils import draw_hand

tracker = HandTracker()
cap = cv2.VideoCapture(0)
csv_file = "motion_dataset.csv"

# Initialize CSV with 120 trajectory features + label
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = []
        for i in range(30):
            headers.extend([f"tx{i}", f"ty{i}", f"ix{i}", f"iy{i}"])
        headers.append("label")
        writer.writerow(headers)

print("--- MOTION DATA COLLECTION ---")
print("1. Press 'r' to START recording a motion (Takes ~1 second).")
print("2. Perform your swipe or pinch.")
print("3. Type the label and save.")

thumb_history = deque(maxlen=30)
index_history = deque(maxlen=30)
recording = False
record_counter = 0

while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    result = tracker.detect(frame)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        frame = draw_hand(frame, hand, w, h)
        
        # Track Thumb (4) and Index (8)
        thumb_history.append((hand[4].x, hand[4].y))
        index_history.append((hand[8].x, hand[8].y))

    if recording:
        record_counter += 1
        cv2.putText(frame, f"RECORDING: {record_counter}/30", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        if record_counter >= 30:
            recording = False
            label = input("Enter motion (e.g., Swipe Right, Zoom In, None): ")
            
            # --- THE OPTIMIZATION: TRANSLATION INVARIANCE ---
            t_start_x, t_start_y = thumb_history[0]
            i_start_x, i_start_y = index_history[0]
            
            row = []
            for i in range(30):
                tx, ty = thumb_history[i]
                ix, iy = index_history[i]
                # Subtract the start position to normalize the trajectory
                row.extend([tx - t_start_x, ty - t_start_y, ix - i_start_x, iy - i_start_y])
            
            row.append(label)
            with open(csv_file, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            print(f"[SAVED] Added 1 sample for: {label}")

    cv2.imshow("Motion Collector", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r') and len(thumb_history) == 30:
        recording = True
        record_counter = 0
    elif key == 27:
        break

cap.release()
tracker.close()
cv2.destroyAllWindows()