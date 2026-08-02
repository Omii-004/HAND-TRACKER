import cv2
import os
import joblib
from src.hand_tracker import HandTracker

def run_test_dataset():
    # Paths based on your project structure
    test_folder = "datasets/archive/asl_alphabet_test/asl_alphabet_test" 
    model_path = "models/pose_model.pkl"

    if not os.path.exists(test_folder):
        print(f"[ERROR] Test folder not found at {test_folder}")
        return
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Train it first!")
        return

    print("[SYSTEM] Loading AI Model...")
    model = joblib.load(model_path)
    tracker = HandTracker(model_path="models/hand_landmarker.task")

    print("\n--- RUNNING BLIND TEST ON TEST DATASET ---")
    
    correct = 0
    total = 0

    # Loop through all the test images
    for image_name in os.listdir(test_folder):
        if not image_name.endswith(('.jpg', '.png', '.jpeg')):
            continue

        image_path = os.path.join(test_folder, image_name)
        frame = cv2.imread(image_path)
        
        if frame is None:
            continue

        # In the Kaggle test set, the actual label is usually the first letter of the filename (e.g., "A_test.jpg")
        expected_label = image_name.split('_')[0]

        result = tracker.detect(frame)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            
            # --- APPLY YOUR EXACT NORMALIZATION ---
            wrist_x = hand[0].x
            wrist_y = hand[0].y
            
            scale = ((hand[12].x - wrist_x)**2 + (hand[12].y - wrist_y)**2) ** 0.5
            scale = max(scale, 1e-6)
            
            features = []
            for lm in hand:
                norm_x = (lm.x - wrist_x) / scale
                norm_y = (lm.y - wrist_y) / scale
                features.extend([norm_x, norm_y])
            
            # Ask the AI to predict based on the features
            prediction = model.predict([features])[0]
            
            total += 1
            if prediction == expected_label:
                correct += 1
                print(f"✅ PASS | Image: {image_name} | AI Guessed: {prediction}")
            else:
                print(f"❌ FAIL | Image: {image_name} | Expected: {expected_label}, but AI Guessed: {prediction}")
        else:
            print(f"⚠️ NO HAND DETECTED | Image: {image_name}")

    if total > 0:
        print(f"\n[FINAL RESULT] Accuracy on Test Set: {(correct/total)*100:.2f}% ({correct}/{total})")
    
    tracker.close()

if __name__ == "__main__":
    run_test_dataset()