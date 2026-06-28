import cv2
from collections import deque
from hand_tracker import HandTracker
from gesture_detector import GestureDetector
from motion_detector import MotionDetector
from data_logger import DataLogger            # ✅ Data Pipeline
from system_controller import SystemController  # ✅ OS Control
from utils import draw_hand, draw_movement_trail

tracker = HandTracker()
gesture = GestureDetector()
motion = MotionDetector(swipe_threshold=200, zoom_threshold=40, cooldown_seconds=2.5) # ✅ Ensure zoom threshold is passed
logger = DataLogger()                         
sys_ctrl = SystemController()                 

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

            # -------------------------
            # ✅ FIXED: DYNAMIC MOTION ROUTING
            # -------------------------
            detected_motion = None
            
            # 1. If palm is open or pointing -> Check for Swipes
            if gesture_name in ["Open Palm", "Point"]:
                detected_motion = motion.detect_swipe(movement_history[label][8])
                
            # 2. If making an L-Sign -> Check for Pinching (Thumb & Index)
            elif gesture_name == "L-Sign / Pinch Ready":
                detected_motion = motion.detect_zoom(
                    movement_history[label][4],  # Pass Thumb memory
                    movement_history[label][8]   # Pass Index memory
                )
                
            # Trigger OS Actions and Data Logging
            if detected_motion:
                active_swipe = detected_motion
                swipe_display_timer = 20
                
                # SAVE THE DATA FOR ML TRAINING
                logger.log_sequence(active_swipe, movement_history[label][8])

                # TRIGGER THE ACTUAL SYSTEM KEYBOARD ACTION
                sys_ctrl.trigger_action(active_swipe)

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

    # ✅ 2. DRAW THE NEW COOLDOWN TIMER BAR
    cooldown_left = motion.get_remaining_cooldown()
    if cooldown_left > 0:
        # Calculate how wide the bar should be (Max width: 200px)
        bar_width = int(200 * (cooldown_left / motion.cooldown_seconds))
        # Draw a shrinking red rectangle
        cv2.rectangle(frame, (10, 60), (10 + bar_width, 80), (0, 0, 255), -1)
        # Put warning text next to it
        cv2.putText(frame, f"LOCK: Reset Hand", (10 + bar_width + 10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 3. Standard Status Text
    cv2.putText(frame, 'System Control + Zoom Active', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
tracker.close()
cv2.destroyAllWindows()