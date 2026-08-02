import cv2
import multiprocessing as mp
from collections import deque

# ---------------------------------------------------------
# 1. THE WORKER PROCESS (Handles all the heavy AI math)
# ---------------------------------------------------------
def hand_tracking_worker(input_queue, output_queue, mode):
    # Initialize all modules strictly INSIDE the worker process
    from src.hand_tracker import HandTracker
    from src.gesture_detector import GestureDetector
    from src.motion_detector import MotionDetector
    from src.system_controller import SystemController
    from src.utils import draw_hand, draw_movement_trail

    print("[SYSTEM] Starting AI Worker Process...")
    tracker = HandTracker()
    gesture = GestureDetector()
    motion = MotionDetector(cooldown_seconds=2.5) 
    sys_ctrl = SystemController()

    fingertip_ids = [4, 8, 12, 16, 20]
    movement_history = {
        "Left": {fid: deque(maxlen=30) for fid in fingertip_ids},
        "Right": {fid: deque(maxlen=30) for fid in fingertip_ids}
    }

    active_swipe = ""
    swipe_display_timer = 0

    while True:
        # Wait for a frame from the main camera process
        frame = input_queue.get()
        if frame is None:  # "Poison pill" to shut down the worker gracefully
            break

        h, w, _ = frame.shape
        result = tracker.detect(frame)
        detected_labels = []

        if result.hand_landmarks and result.handedness:
            for hand, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name
                detected_labels.append(label)
                
                frame = draw_hand(frame, hand, w, h)
                
                # We always run gesture.detect silently in the background because 
                # we need the 'fingers_up' variable for the motion trails to work properly
                gesture_name, fingers_up = gesture.detect(hand, w, h, label)

                for i, fid in enumerate(fingertip_ids):
                    lm = hand[fid]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    if fid in [4, 8]:
                        movement_history[label][fid].append((cx, cy))
                    else:
                        if fingers_up[i] == 1:
                            movement_history[label][fid].append((cx, cy))
                        else:
                            movement_history[label][fid].clear()

                # -------------------------
                # MOTION DETECTION & TRIGGERS (Only in Mode 2 or 3)
                # -------------------------
                if mode in ['2', '3']:
                    detected_motion = motion.detect_motion(
                        movement_history[label][4], 
                        movement_history[label][8]
                    )
                    if detected_motion:
                        active_swipe = detected_motion
                        swipe_display_timer = 20
                        # Trigger the actual system keyboard/mouse action
                        sys_ctrl.trigger_action(active_swipe)

                # -------------------------
                # STATIC POSE TEXT (Only in Mode 1 or 3)
                # -------------------------
                if mode in ['1', '3']:
                    y_pos = 80 if label == "Right" else 120
                    cv2.putText(frame, f'{label}: {gesture_name}', (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        for label in ["Left", "Right"]:
            if label not in detected_labels:
                for fid in fingertip_ids:
                    movement_history[label][fid].clear()

        # Render Trails (Only in Mode 2 or 3 since trails are for motion)
        if mode in ['2', '3']:
            frame = draw_movement_trail(frame, movement_history)

        # -------------------------
        # 🎨 HUD Updates (High-Contrast Black & Cyan Aesthetic)
        # -------------------------
        
        # 1. System Status Panel
        cv2.rectangle(frame, (10, 10), (340, 50), (0, 0, 0), -1) 
        
        # Dynamic Top-Left Label based on what mode the user selected
        sys_text = 'SYSTEM: AI CORE ACTIVE'
        if mode == '1':
            sys_text = 'SYSTEM: STATIC POSES ONLY'
        elif mode == '2':
            sys_text = 'SYSTEM: MOTION TRACKING ONLY'
            
        cv2.putText(frame, sys_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 2. Motion Action Popup & Cooldowns (Only in Mode 2 or 3)
        if mode in ['2', '3']:
            cooldown_left = motion.get_remaining_cooldown()
            if cooldown_left > 0:
                panel_x = w - 350
                cv2.rectangle(frame, (panel_x, 10), (panel_x + 340, 50), (0, 0, 0), -1) 
                
                max_bar_width = 160
                bar_width = int(max_bar_width * (cooldown_left / motion.cooldown_seconds))
                
                cv2.rectangle(frame, (panel_x + 10, 25), (panel_x + 10 + max_bar_width, 35), (50, 50, 50), -1)
                cv2.rectangle(frame, (panel_x + 10, 25), (panel_x + 10 + bar_width, 35), (255, 255, 0), -1)
                
                cv2.putText(frame, "LOCK: RESET", (panel_x + 185, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if swipe_display_timer > 0:
                text_size = cv2.getTextSize(active_swipe, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = 15
                text_y = h - 25 
                
                cv2.rectangle(frame, (text_x - 8, text_y - text_size[1] - 8), 
                                     (text_x + text_size[0] + 8, text_y + 8), (0, 0, 0), -1)
                cv2.putText(frame, active_swipe, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                swipe_display_timer -= 1

        # Send the fully processed frame back to the main thread for display
        if output_queue.full():
            try:
                output_queue.get_nowait()
            except:
                pass
        output_queue.put(frame)

    tracker.close()
    print("[SYSTEM] AI Worker Process Terminated.")


# ---------------------------------------------------------
# 2. THE MAIN PROCESS (Handles only Camera and Display)
# ---------------------------------------------------------
if __name__ == '__main__':
    # Required for Windows multiprocessing compatibility
    mp.freeze_support() 

    # --- NEW: User Mode Selection ---
    print("\n=== SYSTEM INITIALIZATION ===")
    print("1. Static Poses Only")
    print("2. Motion Tracking Only")
    print("3. Both Modes Active")
    mode = input("\nEnter choice (1 / 2 / 3): ").strip()
    
    if mode not in ['1', '2', '3']:
        print("[WARNING] Invalid choice. Defaulting to BOTH (3).")
        mode = '3'

    # Create Queues with maxsize=1 to ensure we only process the absolute newest frame
    input_queue = mp.Queue(maxsize=1)
    output_queue = mp.Queue(maxsize=1)

    # Pass the 'mode' variable into the worker process
    worker = mp.Process(target=hand_tracking_worker, args=(input_queue, output_queue, mode))
    worker.daemon = True 
    worker.start()

    cap = cv2.VideoCapture(0)
    
    display_frame = None
    print("\n[SYSTEM] Main Camera Loop Started.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)

        # 1. Send the newest frame to the AI Worker
        if not input_queue.full():
            input_queue.put(frame)

        # 2. Get the newest processed frame from the AI Worker
        if not output_queue.empty():
            display_frame = output_queue.get()

        # 3. Display the frame
        if display_frame is not None:
            cv2.imshow("Hand Tracker", display_frame)
        else:
            cv2.imshow("Hand Tracker", frame)

        # 4. Exit sequence
        if cv2.waitKey(1) & 0xFF == 27:
            print("[SYSTEM] Shutting down...")
            break

    # Clean up processes safely
    input_queue.put(None) 
    worker.join(timeout=2) 
    cap.release()
    cv2.destroyAllWindows()