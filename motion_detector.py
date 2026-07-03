import time
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class MotionDetector:
    def __init__(self, model_path="custom_motion_model.pkl", cooldown_seconds=2.0):
        print("[SYSTEM] Initializing ML Motion Classifier...")
        try:
            self.model = joblib.load(model_path)
            self.ml_active = True
        except FileNotFoundError:
            print(f"[ERROR] Could not find {model_path}. Motion tracking disabled.")
            self.model = None
            self.ml_active = False

        self.last_action_time = 0  
        self.cooldown_seconds = cooldown_seconds

    def get_remaining_cooldown(self):
        elapsed = time.time() - self.last_action_time
        if elapsed < self.cooldown_seconds:
            return self.cooldown_seconds - elapsed
        return 0

    def detect_motion(self, thumb_history, index_history):
        if not self.ml_active:
            return None

        # 1. System Lock Check
        if self.get_remaining_cooldown() > 0:
            return None

        # 2. Wait for full history buffer
        if len(thumb_history) < 30 or len(index_history) < 30:
            return None

        # 3. Translation Invariance (Normalization)
        t_start_x, t_start_y = thumb_history[0]
        i_start_x, i_start_y = index_history[0]

        # ✅ THE DEADZONE FILTER: Check total displacement
        # Where did the index finger end up at frame 30 compared to frame 1?
        i_end_x, i_end_y = index_history[-1]
        x_movement = abs(i_end_x - i_start_x)
        y_movement = abs(i_end_y - i_start_y)

        # If the finger moved less than 30 pixels in any direction, it's just sitting still.
        # Bypass the AI and return None.
        if x_movement < 30 and y_movement < 30:
            return None

        features = []
        for i in range(30):
            tx, ty = thumb_history[i]
            ix, iy = index_history[i]
            # Normalize and format into a 1D list
            features.extend([tx - t_start_x, ty - t_start_y, ix - i_start_x, iy - i_start_y])

        # 4. Predict using the ML Model
        prediction = self.model.predict([features])[0]

        # 5. Trigger action if it's a valid motion
        if prediction != "None":
            self.last_action_time = time.time()
            return prediction
        
        return None