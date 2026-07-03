import pyautogui

# Safeguard: Move mouse to any corner of the screen to stop execution if it loops out of control
pyautogui.FAILSAFE = True

class SystemController:
    def __init__(self):
        print("[SYSTEM] System Controller Initialized.")

    def trigger_action(self, motion_string):
        if not motion_string:
            return

        # Optimize performance: Normalize string to lowercase and strip whitespaces/arrows
        clean_motion = motion_string.lower().replace(">>>", "").replace("<<<", "").replace("^^^", "").replace("vvv", "").strip()

        # --- Swipe Actions (Web Navigation & Scrolling) ---
        if "swipe right" in clean_motion:
            pyautogui.press('right')
            print("[ACTION] Triggered Right Key")
            
        elif "swipe left" in clean_motion:
            pyautogui.press('left')
            print("[ACTION] Triggered Left Key")
            
        elif "swipe up" in clean_motion:
            # Native PyAutoGUI scroll (Negative moves the page down / scrolls up your view)
            pyautogui.scroll(-500)
            print("[ACTION] Scrolled Down")
            
        elif "swipe down" in clean_motion:
            pyautogui.scroll(500)
            print("[ACTION] Scrolled Up")

        # --- Zoom Actions (Time-Series ML Pipeline) ---
        elif "zoom in" in clean_motion:
            pyautogui.keyDown('ctrl')
            pyautogui.scroll(100)
            pyautogui.keyUp('ctrl')
            print("[ACTION] Zoomed In")
            
        elif "zoom out" in clean_motion:
            pyautogui.keyDown('ctrl')
            pyautogui.scroll(-100)
            pyautogui.keyUp('ctrl')
            print("[ACTION] Zoomed Out")