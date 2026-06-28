class MotionDetector:
    def __init__(self, swipe_threshold=150):
        # The minimum pixel distance the finger must travel to count as a swipe
        self.swipe_threshold = swipe_threshold
        self.cooldown_counter = 0
        self.cooldown_frames = 20  # Prevents triggering 10 swipes in a single second

    def detect_swipe(self, history):
        # 1. Manage cooldown to prevent spamming
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None

        # 2. We need enough data points in the history to analyze a motion
        if len(history) < 20: 
            return None

        # 3. Grab the oldest coordinate (start) and newest coordinate (end)
        start_x, start_y = history[0]
        end_x, end_y = history[-1]

        # 4. Calculate the delta (change) in position
        dx = end_x - start_x
        dy = end_y - start_y

        # 5. Check Horizontal Swipes (X movement is greater than Y movement)
        if abs(dx) > self.swipe_threshold and abs(dx) > abs(dy) * 1.5:
            self.cooldown_counter = self.cooldown_frames
            if dx > 0:
                return "SWIPE RIGHT >>>"
            else:
                return "<<< SWIPE LEFT"
        
        # 6. Check Vertical Swipes (Y movement is greater than X movement)
        # Note: In OpenCV, the Y-axis is inverted (0 is top, height is bottom)
        elif abs(dy) > self.swipe_threshold and abs(dy) > abs(dx) * 1.5:
            self.cooldown_counter = self.cooldown_frames
            if dy > 0:
                return "SWIPE DOWN vvv"
            else:
                return "SWIPE UP ^^^"

        return None