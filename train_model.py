import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_custom_ai(csv_file, model_output_name, tree_depth):
    """A reusable function to train any Random Forest model."""
    if not os.path.exists(csv_file):
        print(f"[ERROR] Could not find '{csv_file}'. Please run the data collector first.")
        return

    print(f"\n[SYSTEM] Loading dataset from {csv_file}...")
    df = pd.read_csv(csv_file)

    # --- 🧹 BACK DOOR DATA CLEANING PIPELINE ---
    print(f"--- Data Cleaning ---")
    initial_rows = len(df)

    # 1. Drop completely blank or corrupted rows (NaNs)
    df = df.dropna()

    # 2. Ensure all labels are strings and drop empty/whitespace labels
    df['label'] = df['label'].astype(str)
    df = df[df['label'].str.strip() != '']

    # 3. Drop exact duplicate rows (prevents the AI from over-memorizing static frames)
    df = df.drop_duplicates()

    final_rows = len(df)
    print(f"Removed {initial_rows - final_rows} bad or duplicate records.")
    print(f"Final clean dataset size: {final_rows} rows.")
    print(f"---------------------")

    # Split data into Features (X) and Labels (y)
    X = df.drop("label", axis=1)
    y = df["label"]

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"[SYSTEM] Training Random Forest Classifier (Depth: {tree_depth})...")
    model = RandomForestClassifier(n_estimators=100, max_depth=tree_depth, random_state=42)
    model.fit(X_train, y_train)

    # Test the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"[SUCCESS] Model trained with Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    joblib.dump(model, model_output_name)
    print(f"[SAVED] Exported trained AI to {model_output_name}\n")


# -------------------------
# INTERACTIVE MENU
# -------------------------
print("=== AI TRAINING HUB ===")
print("1. Train Static Pose AI (from pose_dataset.csv)")
print("2. Train Dynamic Motion AI (from motion_dataset.csv)")
print("3. Train BOTH Models")

choice = input("\nEnter your choice (1, 2, or 3): ").strip()

if choice == '1' or choice == '3':
    train_custom_ai("pose_dataset.csv", "custom_gesture_model.pkl", tree_depth=10)

if choice == '2' or choice == '3':
    train_custom_ai("motion_dataset.csv", "custom_motion_model.pkl", tree_depth=15)

if choice not in ['1', '2', '3']:
    print("[ERROR] Invalid choice. Shutting down.")