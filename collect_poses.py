# This collects the hand pose data and saves it to a CSV file for training a gesture recognition model.
import cv2
import csv
import os
from hand_tracker import HandTracker
from utils import draw_hand

tracker = HandTracker()
cap = cv2.VideoCapture(0)
csv_file = "pose_dataset.csv"

# Initialize CSV with headers (x0, y0, x1, y1 ... x20, y20, label)
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = []
        for i in range(21):
            headers.extend([f"x{i}", f"y{i}"])
        headers.append("label")
        writer.writerow(headers)

print("--- POSE DATA COLLECTION ---")
print("1. Strike a pose in front of the camera.")
print("2. Press 's' to save the pose.")
print("3. Type the gesture name in the terminal and hit Enter.")
print("4. Press 'ESC' to quit.")

while True:
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    result = tracker.detect(frame)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        frame = draw_hand(frame, hand, w, h)
        
        # When 's' is pressed, save the 21 landmarks
        if cv2.waitKey(1) & 0xFF == ord('s'):
            label = input("Enter gesture label (e.g., Open Palm, Peace, Spiderman): ")
            
            row = []
            for lm in hand:
                row.extend([lm.x, lm.y]) # Save raw normalized coordinates
            row.append(label)
            
            with open(csv_file, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            print(f"[SAVED] Added 1 sample for: {label}")

    cv2.imshow("Data Collector", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
tracker.close()
cv2.destroyAllWindows()