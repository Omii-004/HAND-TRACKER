import math
import time

class MotionDetector:
    def __init__(self, swipe_threshold=200, zoom_threshold=40, cooldown_seconds=1.5):
        self.swipe_threshold = swipe_threshold
        self.zoom_threshold = zoom_threshold
        
        # ✅ Track the exact time the last action occurred
        self.last_action_time = 0  
        self.cooldown_seconds = cooldown_seconds

    def get_remaining_cooldown(self):
        """Returns how many seconds are left in the cooldown lock."""
        elapsed = time.time() - self.last_action_time
        if elapsed < self.cooldown_seconds:
            return self.cooldown_seconds - elapsed
        return 0

    def detect_swipe(self, history):
        # 1. Check if we are currently locked in a cooldown
        if self.get_remaining_cooldown() > 0:
            return None

        if len(history) < 20: 
            return None

        start_x, start_y = history[0]
        end_x, end_y = history[-1]

        dx = end_x - start_x
        dy = end_y - start_y

        if abs(dx) > self.swipe_threshold and abs(dx) > abs(dy) * 1.5:
            self.last_action_time = time.time()  # ✅ Lock the system
            return "SWIPE RIGHT >>>" if dx > 0 else "<<< SWIPE LEFT"
        
        elif abs(dy) > self.swipe_threshold and abs(dy) > abs(dx) * 1.5:
            self.last_action_time = time.time()  # ✅ Lock the system
            return "SWIPE DOWN vvv" if dy > 0 else "SWIPE UP ^^^"

        return None

    def detect_zoom(self, thumb_history, index_history):
        # 1. Check if we are currently locked in a cooldown
        if self.get_remaining_cooldown() > 0:
            return None

        if len(thumb_history) < 15 or len(index_history) < 15:
            return None

        start_tx, start_ty = thumb_history[0]
        start_ix, start_iy = index_history[0]
        start_dist = math.hypot(start_tx - start_ix, start_ty - start_iy)

        end_tx, end_ty = thumb_history[-1]
        end_ix, end_iy = index_history[-1]
        end_dist = math.hypot(end_tx - end_ix, end_ty - end_iy)

        delta = end_dist - start_dist

        if delta > self.zoom_threshold:
            self.last_action_time = time.time()  # ✅ Lock the system
            return "ZOOM IN [+] (Pinch Out)"
        
        elif delta < -self.zoom_threshold:
            self.last_action_time = time.time()  # ✅ Lock the system
            return "ZOOM OUT [-] (Pinch In)"

        return None