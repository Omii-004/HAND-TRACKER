class GestureDetector:
    def detect(self, hand_landmarks, w, h, hand_label="Right"):
        if not hand_landmarks:
            return "No Hand", [0, 0, 0, 0, 0]

        # Convert normalized coordinates to pixel dimensions
        pts = [(lm.x * w, lm.y * h) for lm in hand_landmarks]

        fingers = []

        # 👍 Thumb
        if hand_label == "Right":
            fingers.append(1 if pts[4][0] > pts[3][0] else 0)
        else:
            fingers.append(1 if pts[4][0] < pts[3][0] else 0)

        # ✋ Other fingers
        tip_ids = [8, 12, 16, 20]
        lower_ids = [6, 10, 14, 18]

        for tip, low in zip(tip_ids, lower_ids):
            fingers.append(1 if pts[tip][1] < pts[low][1] else 0)

        # -------------------------
        # 🎯 GESTURE CLASSIFICATION
        # -------------------------
        total = sum(fingers)
        gesture_name = "Unknown"

        if total == 0:
            gesture_name = "Fist"
        elif fingers == [0, 1, 0, 0, 0]:
            gesture_name = "Point"
        elif fingers == [0, 1, 1, 0, 0]:
            gesture_name = "Peace"
        elif total == 5:
            gesture_name = "Open Palm"
        elif fingers[0] == 1 and total == 1:
            gesture_name = "Thumbs Up"

        # ✅ NOW RETURNING BOTH: The name AND the list of which fingers are up
        return gesture_name, fingers