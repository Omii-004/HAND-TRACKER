import cv2
import mediapipe as mp
import time
import numpy as np

class HandTracker:
    def __init__(self, model_path="models/hand_landmarker.task"):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        try:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_hands=2
            )
            self.landmarker = HandLandmarker.create_from_options(options)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {model_path}\n{e}")

    def detect(self, frame):
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Generate a strictly increasing timestamp in milliseconds
        timestamp = int(time.perf_counter() * 1000)

        # Process the frame
        result = self.landmarker.detect_for_video(mp_image, timestamp)
        
        return result
        
    def close(self):
        self.landmarker.close()