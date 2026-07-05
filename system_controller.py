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
            
        # ✅ THE FIX: Use 'pgup' and 'pgdn' keys instead of buggy mouse scrolling
        elif "swipe up" in clean_motion:
            # Swiping hand up pushes the page UP, revealing content below (Page Down)
            pyautogui.press('pgdn')
            print("[ACTION] Scrolled Page Down")
            
        elif "swipe down" in clean_motion:
            # Swiping hand down pulls the page DOWN, revealing content above (Page Up)
            pyautogui.press('pgup')
            print("[ACTION] Scrolled Page Up")

        # --- Zoom Actions (Time-Series ML Pipeline) ---
        elif "zoom in" in clean_motion:
            # ✅ THE FIX: Use explicit hotkeys instead of simulated mouse scrolling
            pyautogui.hotkey('ctrl', '+')
            # Note: On some keyboards, you may need to use '=' instead of '+'
            print("[ACTION] Zoomed In")
            
        elif "zoom out" in clean_motion:
            pyautogui.hotkey('ctrl', '-')
            print("[ACTION] Zoomed Out")