import pyautogui

pyautogui.FAILSAFE = True

class SystemController:
    def __init__(self):
        print("[SYSTEM] System Controller Initialized.")

    def trigger_action(self, motion_string):
        if not motion_string:
            return

        # --- Swipes ---
        if motion_string == "SWIPE RIGHT >>>":
            pyautogui.press('right')
        elif motion_string == "<<< SWIPE LEFT":
            pyautogui.press('left')
        elif motion_string == "SWIPE UP ^^^":
            pyautogui.scroll(-500)
        elif motion_string == "SWIPE DOWN vvv":
            pyautogui.scroll(500)

        # --- Zoom Controls ---
        elif motion_string == "ZOOM IN [+] (Pinch Out)":
            # Method 2: Holding Ctrl and scrolling up
            pyautogui.keyDown('ctrl')
            pyautogui.scroll(100)
            pyautogui.keyUp('ctrl')
            
            print("[ACTION] Zoomed In")
            
        elif motion_string == "ZOOM OUT [-] (Pinch In)":
            # Method 2: Holding Ctrl and scrolling down
            pyautogui.keyDown('ctrl')
            pyautogui.scroll(-100)
            pyautogui.keyUp('ctrl')
            
            print("[ACTION] Zoomed Out")