import cv2
from collections import deque
from hand_tracker import HandTracker
from gesture_detector import GestureDetector
from motion_detector import MotionDetector
from data_logger import DataLogger  # ✅ Import the new data pipeline
from utils import draw_hand, draw_movement_trail

tracker = HandTracker()
gesture = GestureDetector()
motion = MotionDetector(swipe_threshold=200)
logger = DataLogger()  # ✅ Initialize the logger

fingertip_ids = [4, 8, 12, 16, 20]
movement_history = {
    "Left": {fid: deque(maxlen=30) for fid in fingertip_ids},
    "Right": {fid: deque(maxlen=30) for fid in fingertip_ids}
}

active_swipe = ""
swipe_display_timer = 0

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    result = tracker.detect(frame)
    detected_labels = []

    if result.hand_landmarks and result.handedness:
        for hand, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            detected_labels.append(label)
            
            frame = draw_hand(frame, hand, w, h)
            gesture_name, fingers_up = gesture.detect(hand, w, h, label)

            for i, fid in enumerate(fingertip_ids):
                if fingers_up[i] == 1:
                    lm = hand[fid]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    movement_history[label][fid].append((cx, cy))
                else:
                    movement_history[label][fid].clear()

            # Analyze the Index Finger (ID 8) for motions
            if gesture_name in ["Open Palm", "Point"]:
                detected_motion = motion.detect_swipe(movement_history[label][8])
                
                if detected_motion:
                    active_swipe = detected_motion
                    swipe_display_timer = 20
                    
                    # ✅ Save the raw coordinate data to our CSV for future ML training
                    logger.log_sequence(active_swipe, movement_history[label][8])

            y_pos = 80 if label == "Right" else 120
            cv2.putText(frame, f'{label}: {gesture_name}', (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    for label in ["Left", "Right"]:
        if label not in detected_labels:
            for fid in fingertip_ids:
                movement_history[label][fid].clear()

    frame = draw_movement_trail(frame, movement_history)

    if swipe_display_timer > 0:
        cv2.putText(frame, active_swipe, (int(w/2) - 150, int(h/2)),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
        swipe_display_timer -= 1

    cv2.putText(frame, 'Data Pipeline Active -> Saving to CSV', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
tracker.close()
cv2.destroyAllWindows()