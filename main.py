import cv2
from collections import deque
from hand_tracker import HandTracker
from gesture_detector import GestureDetector
from utils import draw_hand, draw_movement_trail

tracker = HandTracker()
gesture = GestureDetector()

fingertip_ids = [4, 8, 12, 16, 20]

# ✅ Split memory into Left and Right compartments to avoid crisscrossing lines
movement_history = {
    "Left": {fid: deque(maxlen=30) for fid in fingertip_ids},
    "Right": {fid: deque(maxlen=30) for fid in fingertip_ids}
}

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    result = tracker.detect(frame)
    
    # Keep track of which hands are on screen this exact frame
    detected_labels = []

    if result.hand_landmarks and result.handedness:
        # Loop through ALL hands detected in the frame
        for hand, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            detected_labels.append(label)
            
            # 1. Draw the hand skeleton
            frame = draw_hand(frame, hand, w, h)

            # 2. Get gesture name AND the list of raised fingers
            gesture_name, fingers_up = gesture.detect(hand, w, h, label)

            # 3. Dynamic Tracking Logic (Using the specific Left/Right label)
            for i, fid in enumerate(fingertip_ids):
                if fingers_up[i] == 1:
                    # Finger is OPEN: Track coordinates in the correct hand compartment
                    lm = hand[fid]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    movement_history[label][fid].append((cx, cy))
                else:
                    # Finger is CLOSED: Clear its history
                    movement_history[label][fid].clear()

            # Display the gesture result on screen (Space out Left and Right text)
            y_pos = 80 if label == "Right" else 120
            cv2.putText(frame, f'{label}: {gesture_name}', (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 4. Memory Cleanup: If a hand leaves the screen, clear its entire memory
    for label in ["Left", "Right"]:
        if label not in detected_labels:
            for fid in fingertip_ids:
                movement_history[label][fid].clear()

    # 5. Draw the movement trajectory trails
    frame = draw_movement_trail(frame, movement_history)

    cv2.putText(frame, 'Dual Hand Dynamic Tracking Active', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
tracker.close()
cv2.destroyAllWindows()