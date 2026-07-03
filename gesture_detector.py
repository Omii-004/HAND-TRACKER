import joblib
import warnings

# Suppress sklearn version warnings for a clean terminal
warnings.filterwarnings("ignore", category=UserWarning)

class GestureDetector:
    def __init__(self, model_path="custom_gesture_model.pkl"):
        print("[SYSTEM] Initializing ML Gesture Classifier...")
        try:
            self.model = joblib.load(model_path)
            self.ml_active = True
        except FileNotFoundError:
            print(f"[ERROR] Could not find {model_path}. Falling back to Unknown.")
            self.model = None
            self.ml_active = False

    def detect(self, hand_landmarks, w, h, hand_label="Right"):
        if not hand_landmarks:
            return "No Hand", [0, 0, 0, 0, 0]

        # -------------------------
        # 1. CALCULATE WHICH FINGERS ARE UP (Required for Main.py Trails)
        # -------------------------
        pts = [(lm.x * w, lm.y * h) for lm in hand_landmarks]
        fingers = []

        if hand_label == "Right":
            fingers.append(1 if pts[4][0] > pts[3][0] else 0)
        else:
            fingers.append(1 if pts[4][0] < pts[3][0] else 0)

        tip_ids = [8, 12, 16, 20]
        lower_ids = [6, 10, 14, 18]

        for tip, low in zip(tip_ids, lower_ids):
            fingers.append(1 if pts[tip][1] < pts[low][1] else 0)

        # -------------------------
        # 2. ML GESTURE CLASSIFICATION
        # -------------------------
        if not self.ml_active:
            return "Unknown", fingers

        # Flatten the 21 landmarks into exactly 42 features (x, y, x, y...)
        features = []
        for lm in hand_landmarks:
            features.extend([lm.x, lm.y])

        # Pass the features to the trained ML Model
        # model.predict expects a 2D array, so we wrap features in brackets: [features]
        gesture_name = self.model.predict([features])[0]

        return gesture_name, fingers