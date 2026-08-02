import cv2
import os
import csv
import time
from hand_tracker import HandTracker

def process_image_dataset(dataset_folder="datasets/archive/asl_alphabet_train/asl_alphabet_train", output_csv="datasets/poses_dataset.csv"):
    tracker = HandTracker(model_path="models/hand_landmarker.task")
    
    # Create CSV Headers
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ["sample_id"]
        for i in range(21):
            headers.extend([f"x{i}", f"y{i}"])
        headers.append("label")
        writer.writerow(headers)

    print(f"[SYSTEM] Starting bulk extraction from: {dataset_folder}")
    
    # Loop through every folder (each folder name is the gesture label)
    for label in os.listdir(dataset_folder):
        label_dir = os.path.join(dataset_folder, label)
        if not os.path.isdir(label_dir):
            continue
            
        print(f"Processing gesture: {label}...")
        
        # Loop through every image in the folder
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            frame = cv2.imread(image_path)
            
            if frame is None:
                continue
                
            # Detect hand in the image
            result = tracker.detect(frame)
            
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                
                # Apply your EXACT Feature Engineering (Normalization)
                wrist_x = hand[0].x
                wrist_y = hand[0].y
                
                scale = ((hand[12].x - wrist_x)**2 + (hand[12].y - wrist_y)**2) ** 0.5
                scale = max(scale, 1e-6)
                
                sample_id = int(time.time() * 1000)
                row = [sample_id]
                
                for lm in hand:
                    norm_x = (lm.x - wrist_x) / scale
                    norm_y = (lm.y - wrist_y) / scale
                    row.extend([norm_x, norm_y])
                    
                row.append(label)
                
                # Save to CSV
                with open(output_csv, 'a', newline='') as f:
                    csv.writer(f).writerow(row)

    tracker.close()
    print(f"[SUCCESS] Dataset converted and saved to {output_csv}!")

if __name__ == "__main__":
    process_image_dataset()