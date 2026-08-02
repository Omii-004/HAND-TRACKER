import cv2

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def draw_hand(frame, hand_landmarks, w, h):
    if not hand_landmarks:
        return frame

    points = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    for a, b in HAND_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (255, 0, 0), 2)

    return frame

def draw_movement_trail(frame, trail_history_dict):
    """Draws continuous lines for multiple hands and fingertips."""
    colors = {
        4: (0, 0, 255),    # Thumb: Red
        8: (0, 255, 255),  # Index: Yellow
        12: (0, 255, 0),   # Middle: Green
        16: (255, 0, 0),   # Ring: Blue
        20: (255, 0, 255)  # Pinky: Purple
    }

    # Loop through "Left" and "Right" memory compartments
    for hand_label, hand_history in trail_history_dict.items():
        for finger_id, history in hand_history.items():
            if len(history) < 2:
                continue
                
            color = colors.get(finger_id, (255, 255, 255))
            
            # Draw segments connecting the historic points
            for i in range(1, len(history)):
                thickness = int(2 + (i / len(history)) * 4)
                cv2.line(frame, history[i - 1], history[i], color, thickness)
                
    return frame