import cv2
import multiprocessing as mp
from collections import deque

# ---------------------------------------------------------
# 1. THE WORKER PROCESS (Handles all the heavy AI math)
# ---------------------------------------------------------
def hand_tracking_worker(input_queue, output_queue):
    # Initialize all modules strictly INSIDE the worker process
    from hand_tracker import HandTracker
    from gesture_detector import GestureDetector
    from motion_detector import MotionDetector
    from data_logger import DataLogger
    from system_controller import SystemController
    from utils import draw_hand, draw_movement_trail

    print("[SYSTEM] Starting AI Worker Process...")
    tracker = HandTracker()
    gesture = GestureDetector()
    motion = MotionDetector(swipe_threshold=200, zoom_threshold=40, cooldown_seconds=2.5)
    logger = DataLogger()
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
                gesture_name, fingers_up = gesture.detect(hand, w, h, label)

                for i, fid in enumerate(fingertip_ids):
                    if fingers_up[i] == 1:
                        lm = hand[fid]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        movement_history[label][fid].append((cx, cy))
                    else:
                        movement_history[label][fid].clear()

                # Dynamic Motion Routing
                detected_motion = None
                if gesture_name in ["Open Palm", "Point"]:
                    detected_motion = motion.detect_swipe(movement_history[label][8])
                elif gesture_name == "L-Sign / Pinch Ready":
                    detected_motion = motion.detect_zoom(
                        movement_history[label][4], 
                        movement_history[label][8]   
                    )
                    
                if detected_motion:
                    active_swipe = detected_motion
                    swipe_display_timer = 20
                    logger.log_sequence(active_swipe, movement_history[label][8])
                    sys_ctrl.trigger_action(active_swipe)

                y_pos = 80 if label == "Right" else 120
                cv2.putText(frame, f'{label}: {gesture_name}', (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        for label in ["Left", "Right"]:
            if label not in detected_labels:
                for fid in fingertip_ids:
                    movement_history[label][fid].clear()

        frame = draw_movement_trail(frame, movement_history)

        # HUD Updates
        if swipe_display_timer > 0:
            cv2.putText(frame, active_swipe, (int(w/2) - 150, int(h/2)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
            swipe_display_timer -= 1

        cooldown_left = motion.get_remaining_cooldown()
        if cooldown_left > 0:
            bar_width = int(200 * (cooldown_left / motion.cooldown_seconds))
            cv2.rectangle(frame, (10, 60), (10 + bar_width, 80), (0, 0, 255), -1)
            cv2.putText(frame, f"LOCK: Reset Hand", (10 + bar_width + 10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, 'Multiprocessing AI Core Active', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Send the fully processed frame back to the main thread for display
        # We clear the queue first if it's full to prevent lagging
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

    # Create Queues with maxsize=1 to ensure we only process the absolute newest frame
    input_queue = mp.Queue(maxsize=1)
    output_queue = mp.Queue(maxsize=1)

    # Start the worker process
    worker = mp.Process(target=hand_tracking_worker, args=(input_queue, output_queue))
    worker.daemon = True # Ensures the worker dies if the main program crashes
    worker.start()

    cap = cv2.VideoCapture(0)
    
    # Optional: Force a lower resolution for maximum speed
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Variable to hold the last valid frame in case the AI is still calculating
    display_frame = None

    print("[SYSTEM] Main Camera Loop Started.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)

        # 1. Send the newest frame to the AI Worker (if it is ready for one)
        if not input_queue.full():
            input_queue.put(frame)

        # 2. Get the newest processed frame from the AI Worker (if it is done)
        if not output_queue.empty():
            display_frame = output_queue.get()

        # 3. Display the frame (either the newest processed one, or the old one if AI is busy)
        if display_frame is not None:
            cv2.imshow("Hand Tracker", display_frame)
        else:
            # Fallback before the first frame processes
            cv2.imshow("Hand Tracker", frame)

        # 4. Exit sequence
        if cv2.waitKey(1) & 0xFF == 27:
            print("[SYSTEM] Shutting down...")
            break

    # Clean up processes safely
    input_queue.put(None) # Send the poison pill
    worker.join(timeout=2) # Wait for the worker to finish
    cap.release()
    cv2.destroyAllWindows()