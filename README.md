# 🏏 KhiladiPro - Cricket Grip AI Coach

> **AI-powered real-time cricket bowling grip recognition and coaching assistant using Computer Vision and Machine Learning.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8_Custom-red.svg?logo=ultralytics&logoColor=white)](https://ultralytics.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random_Forest-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Model_Accuracy-95.67%25-brightgreen.svg)](https://github.com/Dinil007/ball)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [AI & Vision Pipeline](#-ai--vision-pipeline)
- [Technologies Used](#-technologies-used)
- [Computer Vision Models](#-computer-vision-models)
  - [MediaPipe Hands](#1-mediapipe-hands-tracking)
  - [YOLO Ball Detector](#2-yolo-cricket-ball-detector)
- [Machine Learning Model](#-machine-learning-model)
- [Feature Engineering (24 Kinematic Features)](#-feature-engineering-24-kinematic-features)
- [Dataset & Data Collection](#-dataset--data-collection)
- [Training Process](#-training-process)
- [Evaluation & Results](#-evaluation--results)
- [Real-Time Inference & Gating](#-real-time-inference--gating)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Future Improvements](#-future-improvements)

---

## 🌟 Overview

In cricket bowling, subtle finger adjustments on the seam and ball surface dictate trajectory, swing, and deviation off the pitch. Traditional bowling coaching relies heavily on naked-eye observation or post-session video review, making real-time micro-correction difficult for aspiring bowlers.

**KhiladiPro – Cricket Grip AI Coach** solves this challenge by combining modern edge Computer Vision and Machine Learning into a real-time feedback assistant. The system tracks 21 hand landmarks per hand, localizes the cricket ball with a custom-trained YOLO model, computes 24 biomechanical/kinematic features, and classifies the bowling grip with **95.67% accuracy** across 5 bowling styles in a fullscreen coaching interface.

---

## 🚀 Key Features

- 🎥 **Real-time Camera Analysis**: Low-latency video pipeline processing live webcam streams at 30+ FPS.
- ✋ **21 Hand Landmark Tracking**: 3D keypoint tracking powered by MediaPipe Hands with hierarchical One Euro landmark smoothing.
- 🏏 **Custom YOLO Ball Detection**: Detects and localizes the cricket ball bounding box in varied lighting and occlusion scenarios.
- 🖐️ **Palm / Back Hand Orientation**: 3D palm plane normal and 2D chirality determinant distinguishing palm-facing from back-of-hand postures.
- 🎯 **Weighted Hand-Ball Association**: Correlates the cricket ball with fingertip (70%) and MCP joint (30%) proximity to identify the active holding hand.
- 👥 **Multi-Hand Intelligence**: Tracks 0, 1, or 2 hands simultaneously with independent chirality, state smoothing, and dedicated HUD status cards.
- 🧠 **24-Feature Kinematic Extraction**: Captures joint angles, individual finger curls, fingertip spreads, ball distances, wrist rotation angles, and ball coverage ratios.
- ⚡ **Random Forest Classification**: Classifies 5 cricket grips with probability mapping and 10-frame majority voting stabilization.
- 📊 **Broadcast Coaching Dashboard**: 1080p fullscreen display separating the live camera stream from an analytics panel with confidence meters and dynamic coaching cues.

---

## 🏗️ System Architecture

```
                                    ┌───────────────────────┐
                                    │    Webcam Capture     │
                                    │  (OpenCV 30+ FPS HD)  │
                                    └───────────┬───────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
        ┌─────────────────────────┐                           ┌─────────────────────────┐
        │  MediaPipe Hand Engine  │                           │   Custom YOLOv8 Model   │
        │  (21 3D Hand Landmarks) │                           │  (Cricket Ball Detect)  │
        └────────────┬────────────┘                           └────────────┬────────────┘
                     │                                                     │
                     ▼                                                     │
        ┌─────────────────────────┐                                        │
        │ One Euro Filter Smoother│                                        │
        │  (Per-side Jitter Drop) │                                        │
        └────────────┬────────────┘                                        │
                     │                                                     │
                     ▼                                                     │
        ┌─────────────────────────┐                                        │
        │ Hand Orientation Module │                                        │
        │ (3D Normal: PALM / BACK)│                                        │
        └────────────┬────────────┘                                        │
                     │                                                     │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                                                ▼
                                   ┌─────────────────────────┐
                                   │  Hand-Ball Association  │
                                   │ (Weighted Finger + MCP) │
                                   └────────────┬────────────┘
                                                │
                                                ▼
                                   ┌─────────────────────────┐
                                   │ 24 Kinematic Extractor  │
                                   │ (Angles, Curls, Spreads)│
                                   └────────────┬────────────┘
                                                │
                                                ▼
                                   ┌─────────────────────────┐
                                   │ Random Forest Classifier│
                                   │ (5 Grip Classes @ 60%)  │
                                   └────────────┬────────────┘
                                                │
                                                ▼
                                   ┌─────────────────────────┐
                                   │ 10-Frame Temporal Vote  │
                                   │  (Majority Smoothing)   │
                                   └────────────┬────────────┘
                                                │
                                                ▼
                                   ┌─────────────────────────┐
                                   │ Fullscreen Coaching UI  │
                                   │ (1080p Split Dashboard) │
                                   └─────────────────────────┘
```

---

## 🔄 AI & Vision Pipeline

1. **Frame Ingestion**: Captures live camera frames using OpenCV and converts them to RGB color space.
2. **Dual-Model Inference**:
   - **Hand Tracking**: MediaPipe Hands detects multi-hand keypoints ($21 \times (x, y, z)$).
   - **Ball Detection**: YOLOv8 predicts bounding boxes (`x1, y1, x2, y2`) for the cricket ball.
3. **One Euro Smoothing**: Hierarchical velocity-adaptive filtering stabilizes fingertip and joint coordinates to eliminate jitter.
4. **Orientation Classification**: Evaluates palm surface normals $\mathbf{N} = \mathbf{v}_1 \times \mathbf{v}_2$ and 2D anatomical chirality to determine `PALM` vs `BACK`.
5. **Hand-Ball Association**: Calculates weighted Euclidean distances from the ball center to fingertips (70%) and MCP joints (30%) to identify the bowler's holding hand.
6. **Kinematic Feature Extraction**: Extracts 24 normalized geometric and distance features invariant to camera distance.
7. **Inference & Gating**: The Random Forest classifier predicts grip probabilities only when the holding hand is in `BACK` view with the ball held.
8. **Temporal Smoothing**: A 10-frame sliding window with majority voting ensures flicker-free real-time HUD output.

---

## 🛠️ Technologies Used

| Technology | Purpose in Project | Key Reason Selected |
| :--- | :--- | :--- |
| **Python 3.10+** | Core programming language | Comprehensive AI/CV library ecosystem and fast prototyping |
| **OpenCV (`cv2`)** | Video capture, drawing, and UI canvas | High-performance frame rendering, window scaling, and image processing |
| **MediaPipe Hands** | 21 3D hand landmark localization | Real-time landmark regression with sub-millisecond CPU execution |
| **Ultralytics YOLOv8** | Custom cricket ball object detector | High precision detection under motion blur, lighting changes, and hand occlusion |
| **NumPy** | Vector and matrix mathematics | Fast 2D/3D cross products, Euclidean distances, and vector norms |
| **Pandas** | Tabular feature management | Structured CSV data processing, one-hot encoding, and feature alignment |
| **Scikit-Learn** | Model training and evaluation | Robust `RandomForestClassifier`, `train_test_split`, and metrics |
| **Joblib** | Model serialization | High-speed binary persistence for `.pkl` models, encoders, and schema |
| **Git / GitHub** | Version control and collaboration | Structured branch tracking and artifact management |

---

## 👁️ Computer Vision Models

### 1. MediaPipe Hands Tracking
- **Landmark Topology**: 21 3D anatomical points per hand (Wrist, Thumb, Index, Middle, Ring, Pinky joints and tips).
- **Tracking Optimization**:
  - `static_image_mode = False` for video tracking continuity.
  - `model_complexity = 1` for balanced accuracy and speed.
  - `min_detection_confidence = 0.5`, `min_tracking_confidence = 0.6`.
- **Temporal Stabilization**: Custom `OneEuroFilter` implementation applies adaptive cutoff frequencies based on instantaneous velocity, eliminating stationary jitter while preserving responsive movement.

### 2. YOLO Cricket Ball Detector
- **Model**: Custom-trained YOLOv8 detector saved at `models/best.pt`.
- **Functionality**: Localizes cricket balls under dynamic angles, seam orientations, finger occlusions, and background variations.
- **Output**: Bounding box coordinates (`x1, y1, x2, y2`) and confidence score.

---

## 🧠 Machine Learning Model

- **Algorithm**: `RandomForestClassifier(n_estimators=200, random_state=42)`
- **Why Random Forest was Selected**:
  - **Non-Linear Boundaries**: Seamlessly models complex combinations of finger curls and wrist angles.
  - **Tabular Efficiency**: Ideal for normalized kinematic feature matrices.
  - **Low Latency**: Sub-millisecond CPU inference per frame during live camera loops.
  - **Overfitting Resistance**: Ensemble bagging prevents overfitting across diverse bowler hand sizes.

### Target Grip Classes (5 Categories):
1. 🔴 **`seam_grip`**: Standard upright seam grip (index and middle fingers parallel across the seam).
2. 🔄 **`off_spin_grip`**: Wide index-to-middle spread with index finger gripping across the seam for spin revolution.
3. ↩️ **`inswing`**: Inward wrist rotation with seam tilted toward fine leg.
4. ↪️ **`outswing`**: Outward wrist rotation with seam tilted toward first slip.
5. 🤏 **`knuckle_ball`**: Folded/curled index and middle knuckles resting on the ball surface for pace reduction.

---

## 📐 Feature Engineering (24 Kinematic Features)

All intra-hand distances are normalized by **Hand Scale** ($\text{Wrist [0]} \rightarrow \text{Middle MCP [9]}$), ensuring total scale invariance regardless of hand distance from the camera.

```
       Thumb [4]      Index [8]      Middle [12]     Ring [16]     Pinky [20]
          \              |               |              |             /
           \           (DIP 7)         (DIP 11)       (DIP 15)     (DIP 19)
            (IP 3)     (PIP 6)         (PIP 10)       (PIP 14)     (PIP 18)
              \          |               |              |             /
             (MCP 2)   (MCP 5) —————— (MCP 9) —————— (MCP 13) ———— (MCP 17)
                 \        \              |              /          /
                  (CMC 1)  \             |             /          /
                     \______\____________|____________/__________/
                                         │
                                     Wrist [0]
```

### 1. Hand Identity & Orientation Features (2)
- `hand_side`: Categorical (`LEFT` / `RIGHT`).
- `orientation`: Geometric orientation (`BACK` / `PALM`).

### 2. Finger Joint Angles (3)
- `index_angle`: Angle at Index PIP joint ($5 \rightarrow 6 \rightarrow 8$).
- `middle_angle`: Angle at Middle PIP joint ($9 \rightarrow 10 \rightarrow 12$).
- `ring_angle`: Angle at Ring PIP joint ($13 \rightarrow 14 \rightarrow 16$).

### 3. Finger Curl Angles (5)
- `thumb_angle`: Thumb bend angle ($2 \rightarrow 3 \rightarrow 4$).
- `index_curl`: Index bend angle ($5 \rightarrow 6 \rightarrow 8$).
- `middle_curl`: Middle bend angle ($9 \rightarrow 10 \rightarrow 12$).
- `ring_curl`: Ring bend angle ($13 \rightarrow 14 \rightarrow 16$).
- `pinky_curl`: Pinky bend angle ($17 \rightarrow 18 \rightarrow 20$).

### 4. Finger Spread & Distance Features (4)
- `thumb_index_distance`: Normalized distance from Thumb Tip (4) to Index Tip (8).
- `index_middle_distance`: Normalized spread from Index Tip (8) to Middle Tip (12).
- `middle_ring_distance`: Normalized spread from Middle Tip (12) to Ring Tip (16).
- `ring_pinky_distance`: Normalized spread from Ring Tip (16) to Pinky Tip (20).

### 5. Ball-Relative Proximity Features (8)
- `ball_thumb_distance`: Ball center to Thumb Tip (4).
- `ball_index_distance`: Ball center to Index Tip (8).
- `ball_middle_distance`: Ball center to Middle Tip (12).
- `ball_ring_distance`: Ball center to Ring Tip (16).
- `ball_index_mcp_distance`: Ball center to Index MCP joint (5).
- `ball_middle_mcp_distance`: Ball center to Middle MCP joint (9).
- `ball_ring_mcp_distance`: Ball center to Ring MCP joint (13).
- `ball_coverage_ratio`: Average fingertip-to-ball distance normalized by hand scale.

### 6. Wrist Rotation & Alignment (2)
- `wrist_angle`: Angle of Middle MCP relative to horizontal wrist axis.
- `wrist_rotation_angle`: Signed rotation angle of the palm plane vector relative to horizontal axis (differentiates inswing vs outswing).

---

## 📊 Dataset & Data Collection

The dataset was curated using the custom validation collector [`scripts/collect_data.py`](scripts/collect_data.py):
- **Total Dataset Size**: **1,500 balanced samples**
- **Samples Per Class**: **300 samples** for each of the 5 grip classes.
- **Pre-Save Integrity Checks**: Rejects any sample if orientation is not `BACK`, if the ball is not detected, or if any of the 24 kinematic features is missing.

```
Class Distribution in data/features.csv:
├── seam_grip      : 300 samples (20.0%)
├── off_spin_grip  : 300 samples (20.0%)
├── inswing        : 300 samples (20.0%)
├── outswing       : 300 samples (20.0%)
└── knuckle_ball   : 300 samples (20.0%)
Total              : 1,500 samples (100.0%)
```

---

## 🏋️ Training Process

The training pipeline in [`scripts/train_classifier.py`](scripts/train_classifier.py) executes as follows:

1. **Data Ingestion**: Loads `data/features.csv` via Pandas and validates all columns.
2. **Categorical Encoding**: One-hot encodes `hand_side` and `orientation`.
3. **Target Encoding**: Uses `LabelEncoder` to encode target strings to integers.
4. **Stratified Split**: 80% Training (1,200 samples) and 20% Testing (300 samples) stratified equally across all 5 classes (60 samples per class).
5. **Model Fitting**: Trains a 200-tree Random Forest with reproducible seeds (`random_state=42`).
6. **Serialization**: Saves model artifacts in `models/`:
   - `models/grip_classifier.pkl`: Trained Random Forest model.
   - `models/label_encoder.pkl`: Label encoder class mapping.
   - `models/feature_columns.pkl`: Exact 25-column feature schema.

---

## 📈 Evaluation & Results

The classifier achieves an overall test accuracy of **95.67%** (0.9567) on unseen test samples:

```
               precision    recall  f1-score   support

      inswing       0.91      0.97      0.94        60
 knuckle_ball       1.00      0.95      0.97        60
off_spin_grip       1.00      1.00      1.00        60
     outswing       0.95      1.00      0.98        60
    seam_grip       0.93      0.87      0.90        60

     accuracy                           0.96       300
    macro avg       0.96      0.96      0.96       300
 weighted avg       0.96      0.96      0.96       300
```

---

## 🖥️ Real-Time Inference & Gating

- **Orientation Gating**: Predictions trigger only when the holding hand is in `BACK` view. `PALM` views prompt `"Show back of hand"` to avoid false predictions.
- **Confidence Thresholding**: Set to `0.60` (60%). Predictions below 60% are marked as `UNCERTAIN`.
- **Temporal Majority Voting**: Maintains a sliding buffer of the last 10 frames and selects the dominant class.
- **1080p Split Dashboard**:
  - **Left Area (1400x1080)**: Live camera feed with aspect-ratio preservation.
  - **Right Sidebar (520x1080)**: Dark slate coaching panel showing detection cards, identified grip titles, confidence meters, and contextual coaching advice.

---

## 📁 Project Structure

```
KhiladiPro/
├── app.py                     # Main real-time coaching application & fullscreen UI
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
│
├── ball/
│   └── ball_detector.py       # Custom YOLOv8 cricket ball detector wrapper
│
├── camera/
│   └── webcam.py              # Thread-safe OpenCV video capture wrapper
│
├── classifier/
│   ├── __init__.py
│   └── grip_classifier.py     # Random Forest grip inference & schema alignment
│
├── data/
│   └── features.csv           # 1,500 balanced 24-feature dataset samples
│
├── features/
│   ├── __init__.py
│   └── feature_extractor.py   # 24-feature kinematic and distance extraction engine
│
├── hand/
│   ├── hand_detector.py       # MediaPipe Hands detector & persistent tracking
│   ├── hand_orientation.py    # 3D palm normal & chirality orientation classifier
│   └── landmark_smoother.py   # One Euro hierarchical landmark smoothing filter
│
├── models/
│   ├── best.pt                # Trained YOLOv8 cricket ball weights
│   ├── grip_classifier.pkl    # Trained Random Forest classifier
│   ├── label_encoder.pkl      # Sklearn LabelEncoder artifact
│   └── feature_columns.pkl    # Saved feature schema list
│
├── scripts/
│   ├── collect_data.py        # Validated data collection script with HUD telemetry
│   ├── train_classifier.py    # Stratified model training and evaluation script
│   ├── collect_ball_images.py # Helper script for ball image harvesting
│   ├── collect_negative_images.py # Helper script for background harvesting
│   ├── prepare_ball_dataset.py# Helper script for YOLO dataset structuring
│   └── add_negative_images.py # Helper script for background augmentation
│
└── tracking/
    └── hand_ball_association.py # Weighted fingertip + MCP ball-holder assignment
```

---

## 💻 Installation & Setup

### 1. Clone Repository
```powershell
git clone https://github.com/Dinil007/ball.git
cd ball
```

### 2. Set Up Virtual Environment
```powershell
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
pip install ultralytics torch torchvision pandas scikit-learn joblib
```

---

## 🎮 Usage Guide

### 1. Run Live Real-time AI Coach
```powershell
python app.py
```
- The application will launch in **Fullscreen 1080p mode**.
- Hold your hand in front of the webcam with a cricket ball.
- Press **`Q`** or **`ESC`** to exit.

### 2. Collect Additional Training Data
```powershell
python scripts/collect_data.py
```
- Enter target grip name (e.g. `seam_grip`, `off_spin_grip`, `inswing`, `outswing`, `knuckle_ball`).
- Press **`SPACE`** to record validated 24-feature samples.

### 3. Retrain the Classifier
```powershell
python scripts/train_classifier.py
```
- Reads `data/features.csv`, evaluates test accuracy, and exports updated `.pkl` models to `models/`.

---

## 🔮 Future Improvements

- [ ] **Full Body Biomechanics**: Integrate MediaPipe Pose to analyze bowling run-up, arm angle, and release point.
- [ ] **Temporal Deep Learning**: Explore LSTM / Temporal Convolutional Networks (TCN) for continuous release-phase action recognition.
- [ ] **Mobile & Edge Deployment**: Export models to ONNX / TFLite for iOS and Android mobile coaching applications.
- [ ] **Additional Bowling Variations**: Add support for Leg-spin (Googly, Flipper, Leg-break), Cross-seam, and Carrom ball grips.
- [ ] **Audio Feedback**: Voice-assisted audio cues (`"Tilt seam outward"`, `"Fold knuckles deeper"`) for hands-free practice.

---

## 👨‍💻 Author & Maintainer

**Dinil Raj M R**  
- GitHub: [@Dinil007](https://github.com/Dinil007)
- Project Repository: [KhiladiPro - CricketGrip AI](https://github.com/Dinil007/ball)
