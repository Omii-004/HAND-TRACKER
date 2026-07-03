# 🚀 ML-Driven Hand Gesture OS Controller

[![Status](https://img.shields.io/badge/Status-Ongoing-yellow)](#)
[![License](https://img.shields.io/badge/license-Open%20Source-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](#)

A real-time hand tracking and gesture recognition system built using OpenCV and MediaPipe Tasks API that utilizes custom machine learning models to control your operating system completely hands-free.

## 📌 Context & Problem Statement
* **Why was this project created?**
  This project was built to evolve a standard rule-based hand tracker into a robust, trainable AI pipeline capable of recognizing both static geometries and time-series trajectories.
* **What problem does it solve?**
  Traditional gesture controllers rely on hardcoded thresholds that break easily depending on screen distance or camera angle. By implementing translation invariance and a pure Python Scikit-Learn pipeline, this system normalizes movement so you can swipe, zoom, and navigate seamlessly from any position. 
* **Target Audience:**
  AI students interested in Computer Vision, Human-Computer Interaction (HCI), and deploying lightweight Random Forest models for real-time edge processing.

## ✨ Features
* **Custom Machine Learning Pipeline:** Dedicated scripts to easily collect coordinate data and train unique Random Forest classifiers from scratch.
* **Real-time webcam hand tracking:** Multi-hand detection (supports 2 hands).
* **Gesture recognition:** Classifies gestures like Fist, Peace, Point, Open Palm, and Thumbs Up.
* **Connected landmark visualization:** Displays a skeleton-like structure with smooth and visible keypoints directly on the video feed.
* **System Integration:** Maps ML trajectory predictions directly to PyAutoGUI for OS-level scrolling, swiping, and zooming.
* **Fast and lightweight:** Optimized pipeline that runs entirely on the CPU.

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3
* **Computer Vision:** OpenCV, MediaPipe Tasks API
* **Machine Learning:** Scikit-Learn (Random Forest), NumPy, Pandas, Joblib
* **System Automation:** PyAutoGUI

## 📂 Architecture & Project Structure

```text
Hand Tracker/
├── Data Collection & Training (Phase 1)
│   ├── collect_motions.py         # Records 30-frame time-series trajectory data
│   ├── collect_poses.py           # Records static (X,Y) spatial data for hand signs
│   └── train_model.py             # Unified ML hub to train and export .pkl models
│
├── Real-Time Inference (Phase 2)
│   ├── main.py                    # Core multiprocessing orchestrator and camera loop
│   ├── hand_tracker.py            # MediaPipe wrapper for landmark extraction
│   ├── gesture_detector.py        # Spatial inference using custom_gesture_model.pkl
│   ├── motion_detector.py         # Temporal inference using custom_motion_model.pkl
│   ├── system_controller.py       # Maps ML predictions to PyAutoGUI OS actions
│   ├── data_logger.py             # Logs gesture sequences and system triggers
│   └── utils.py                   # Drawing utilities for landmarks and trails
│
└── Configuration & Setup
    ├── requirements.txt           # Python dependencies
    └── .gitignore                 # Specifies intentionally untracked files

```

## 📦 Installation & Setup

Follow these steps to get the real-time inference environment running on your local machine.

# 1. Clone the repository
```bash
git clone <your-repo-url>
cd HandTracker
```

# 2. Create a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

# 3. Install dependencies
```bash
pip install -r requirements.txt
```

⚙️ Model Setup

📥 Download the Model
Get the required MediaPipe hand model from the official documentation:

[👉 MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

📁 Place the File

Save the downloaded .task file exactly in this directory:

```bash
models/hand_landmarker.task
```

# 🧠 Train Your Custom AI Models

Before running the main controller, you must teach the system your specific hand geometry and motion styles.

## 📸 Step A: Collect Spatial Data (Static Poses)
Run the pose collection script:

```bash
python collect_poses.py
```

- Strike a static pose (e.g., "Fist", "Point", "Open Palm", "Peace").
- Press `s`, type the label in the terminal, and press Enter.
- Capture approximately 30 varied samples per gesture (altering distance and angle slightly).

## 🎬 Step B: Collect Time-Series Data (Dynamic Motions)
Run the motion collection script:

```bash
python collect_motions.py
```

- Press `r` to start a 30-frame recording.
- Perform a continuous motion (e.g., "Swipe Right", "Zoom In").
- Type the label to save.

> ⚠️ **Crucial:** Record at least 20 samples labeled `None` where your hand is simply resting or doing random background tasks to prevent false positives.

## ⚙️ Step C: Train the Models
Run the training script:

```bash
python train_model.py
```

- Follow the interactive terminal menu to train both the Static Pose AI and Dynamic Motion AI.
- This will output your `custom_gesture_model.pkl` and `custom_motion_model.pkl` files into your project directory.

# 🚀 Phase 2: Run the OS Controller
Once the models are trained and saved, launch the live dashboard:

```bash
python main.py
```
- **Activate:** Hold up a trigger gesture (like "Point" or "Open Palm") to activate the motion tracker.
- **Execute:** Perform your trained motion to execute OS commands.
- **Exit:** Press `ESC` to safely exit the application.
Execute: Perform your trained motion to execute OS commands.

Exit: Press ESC to safely exit the application.

# 👨‍💻 **Author:** Omkar
📄 **License:** This project is open-source and free to use.
