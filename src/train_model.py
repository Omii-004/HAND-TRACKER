import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    cross_val_score,
    train_test_split
)


# -------------------------------
# CORE TRAIN FUNCTION
# -------------------------------
def train_custom_ai(csv_file, model_output_name, tree_depth):
    if not os.path.exists(csv_file):
        print(f"[ERROR] File '{csv_file}' not found.")
        return

    print(f"\n[SYSTEM] Loading dataset from {csv_file}")
    df = pd.read_csv(csv_file)

    # Fix column spacing issues
    df.columns = df.columns.str.strip()

    print("\n[DEBUG] Columns:", df.columns.tolist())

    # -------------------------------
    # DATA CLEANING
    # -------------------------------
    print("\n--- DATA CLEANING ---")

    initial_rows = len(df)

    df = df.dropna()
    df['label'] = df['label'].astype(str)
    df = df[df['label'].str.strip() != '']
    df = df.drop_duplicates()

    final_rows = len(df)

    print(f"Removed {initial_rows - final_rows} bad rows")
    print(f"Final dataset size: {final_rows}")

    if final_rows < 20:
        print("[WARNING] Dataset too small. Collect more data for better results.")

    # -------------------------------
    # LABEL DISTRIBUTION
    # -------------------------------
    print("\n--- LABEL DISTRIBUTION ---")
    print(df['label'].value_counts())

    # -------------------------------
    # AUTO-FIX sample_id (IMPORTANT)
    # -------------------------------
    if "sample_id" not in df.columns:
        print("\n[WARNING] 'sample_id' missing → auto-generating...")

        GROUP_SIZE = 10  # 🔥 adjust if needed
        df["sample_id"] = df.index // GROUP_SIZE

    # -------------------------------
    # FEATURE / LABEL SPLIT
    # -------------------------------
    X = df.drop(["label", "sample_id"], axis=1)
    y = df["label"]
    groups = df["sample_id"]

    # -------------------------------
    # TRAIN / TEST SPLIT
    # -------------------------------
    if "sample_id" in df.columns:
        print("\n[SYSTEM] Using Group-based split")

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_idx, test_idx in gss.split(X, y, groups):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
    else:
        print("\n[SYSTEM] Using standard split")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # -------------------------------
    # MODEL TRAINING
    # -------------------------------
    print("\n[SYSTEM] Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=tree_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # -------------------------------
    # TEST EVALUATION
    # -------------------------------
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n[RESULT] Test Accuracy: {accuracy * 100:.2f}%")

    # -------------------------------
    # CROSS VALIDATION & STABILITY
    # -------------------------------
    print("\n[SYSTEM] Running Cross Validation...")

    cv_scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    
    mean_acc = cv_scores.mean()
    std_dev = cv_scores.std()
    variance = std_dev ** 2  # Variance is the square of standard deviation

    print(f"[CV] Mean Accuracy: {mean_acc * 100:.2f}%")
    print(f"[CV] Std Dev: {std_dev * 100:.2f}%")
    print(f"[CV] Variance: {variance:.5f}")

    # --- AI STABILITY DIAGNOSTIC ---
    print("\n--- MODEL STABILITY ---")
    if variance < 0.0005: 
        print("[STATUS] 🟢 LOW VARIANCE: Model is highly STABLE. It generalizes perfectly to new data.")
    elif variance < 0.0020:
        print("[STATUS] 🟡 MODERATE VARIANCE: Model is fairly stable, but shows slight fluctuations.")
    else:
        print("[STATUS] 🔴 HIGH VARIANCE: Model is UNSTABLE. It is likely overfitting to the training data.")

    # -------------------------------
    # SAMPLE PREDICTIONS
    # -------------------------------
    print("\n--- SAMPLE PREDICTIONS ---")
    for i in range(min(10, len(predictions))):
        print(f"Pred: {predictions[i]} | Actual: {y_test.iloc[i]}")

    # -------------------------------
    # SAVE MODEL
    # -------------------------------
    joblib.dump(model, model_output_name)
    print(f"\n[SAVED] Model saved as '{model_output_name}'")


# -------------------------------
# MENU (OPTIMIZED)
# -------------------------------
def main():
    print("=== AI TRAINING HUB ===")
    print("1. Train Static Pose Model")
    print("2. Train Motion Model")
    print("3. Train BOTH")

    choice = input("\nEnter your choice (1 / 2 / 3): ").strip()

    # ✅ THE FIX: Pointed properly to your datasets/ and models/ folders, and updated to poses_dataset.csv
    tasks = {
        '1': ("datasets/poses_dataset.csv", "models/pose_model.pkl", 5),
        '2': ("datasets/motion_dataset.csv", "models/motion_model.pkl", 7),
    }

    if choice == '3':
        selected = ['1', '2']
    elif choice in tasks:
        selected = [choice]
    else:
        print("[ERROR] Invalid choice.")
        return

    for key in selected:
        csv_file, model_name, depth = tasks[key]
        train_custom_ai(csv_file, model_name, depth)


if __name__ == "__main__":
    main()