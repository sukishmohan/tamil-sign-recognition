# Tamil Sign Language Recognition System - Complete Overview

## 🎯 PROJECT SUMMARY

This is a **deep-learning computer vision system** that recognizes **247 Tamil alphabet hand signs + 1 background class (total 248 classes)** from a live webcam feed using:
- **Transfer Learning** with MobileNetV2 (pretrained on ImageNet)
- **TensorFlow/Keras** for model building and training
- **OpenCV** for image processing
- **Flask** for web interface

---

## 📚 TABLE OF CONTENTS

1. [What This Project Does](#what-this-project-does)
2. [Technology Stack (From Grassroot to Height)](#technology-stack)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Complete Workflow](#complete-workflow)
8. [File Descriptions & Purposes](#file-descriptions)
9. [How to Run Everything](#how-to-run-everything)
10. [Outputs & Results](#outputs--results)

---

## 🎬 What This Project Does

### Core Functionality:
1. **Recognizes Tamil hand signs** - Detects hand gestures representing Tamil alphabets from a camera
2. **Builds words** - Accumulates recognized letters into words
3. **Provides live feedback** - Shows confidence scores and real-time predictions
4. **Web interface** - Flask app with webcam streaming and UI
5. **Offline recognition** - Can work without internet after model training

### Real-World Application:
- Assist deaf/hard-of-hearing people in Tamil sign language communication
- Convert Tamil sign language to text
- Build a bridge between sign language users and non-users

---

## 🔧 TECHNOLOGY STACK (From Grassroot to Height)

### **LEVEL 1: Foundation (OS & Runtime)**
```
Windows OS
    ↓
Python 3.9+ (Programming Language)
    ↓
Virtual Environment (venv)
```

### **LEVEL 2: Scientific Computing & Data Processing**
```
NumPy          → Numerical array operations
Pillow (PIL)   → Image manipulation
scikit-learn   → Data splitting, evaluation metrics
```

### **LEVEL 3: Deep Learning Framework**
```
TensorFlow 2.12+
    ├── Keras (High-level API)
    │   ├── Model building
    │   ├── Layer definitions
    │   ├── Loss functions
    │   └── Callbacks (early stopping, checkpoints)
    └── tf.keras.applications
        └── MobileNetV2 (Pretrained weights from ImageNet)
```

### **LEVEL 4: Computer Vision**
```
OpenCV 4.7+   → Real-time video capture, image processing, drawing
                (frame reading, resizing, color conversion, overlays)
```

### **LEVEL 5: Web Framework**
```
Flask 2.3+    → Web server
                ├── Route handling (/video_feed, /predict, etc.)
                ├── Template rendering (HTML/CSS/JS)
                └── JSON API endpoints
```

### **LEVEL 6: Visualization**
```
Matplotlib    → Training curves, confusion matrices, evaluation plots
```

### **TECHNOLOGY DEPENDENCIES SUMMARY (requirements.txt)**
```
Flask>=2.3.0              # Web framework
tensorflow>=2.12.0        # Deep learning
opencv-python>=4.7.0      # Computer vision
numpy>=1.23.0             # Numerical computing
matplotlib>=3.6.0         # Plotting & visualization
scikit-learn>=1.2.0       # ML utilities, metrics
Pillow>=9.4.0             # Image processing
```

---

## 📁 PROJECT STRUCTURE

```
tamil_sign_recognition/
│
├── 📄 CORE PYTHON MODULES (Processing & ML)
│   ├── utils.py                    # Label mappings, helper functions
│   ├── preprocessing.py            # Dataset loading, splitting, augmentation
│   ├── train_model.py              # Model building, two-phase training
│   ├── evaluate_model.py           # Test set evaluation, metrics
│   ├── real_time_prediction.py     # Live webcam inference
│   └── app.py                      # Flask web application
│
├── 📊 MODELS & ARTIFACTS (Saved Models & Weights)
│   └── models/
│       ├── tamil_sign_model.keras      # Final trained model
│       ├── best_model.keras            # Best checkpoint during training
│       ├── class_indices.npy           # Class index mappings
│       ├── class_names.npy             # Sorted class names
│       ├── test_paths.npy              # Test image file paths
│       ├── test_labels.npy             # Test image labels
│       ├── evaluation_report.txt       # Classification metrics
│       ├── training_curves.png         # Accuracy/loss plots
│       └── confusion_matrix_top30.png  # Confusion matrix visualization
│
├── 📂 TRAINING LOGS
│   └── logs/
│       ├── Phase-1 (Frozen Base)_<timestamp>/
│       │   └── TensorBoard logs for phase 1
│       └── Phase-2 (Fine-tuning)_<timestamp>/
│           └── TensorBoard logs for phase 2
│
├── 🖼️ DATASET (Auto-generated split, non-copying)
│   └── dataset_split/
│       ├── train/              (80% of data)
│       │   ├── 1-247/          (Integer folder names = Tamil chars)
│       │   └── Background/     (Non-hand images)
│       ├── val/                (10% of data)
│       └── test/               (10% of data)
│
├── 🌐 WEB INTERFACE
│   ├── static/
│   │   ├── style.css           # Styling
│   │   └── app.js              # Frontend JavaScript
│   └── templates/
│       └── index.html          # HTML template
│
├── 📖 DOCUMENTATION
│   ├── README.md               # Quick start guide
│   ├── requirements.txt        # Dependencies
│   ├── training_log.txt        # Training history
│   └── training_log_full.txt   # Detailed training logs
│
└── 📝 CONFIG FILES
    └── .gitignore, __pycache__/
```

---

## 📊 DATASET

### **Source**
- **Name**: TLFS23 - Tamil Language Finger Spelling Image Dataset
- **Location**: External directory (referenced in preprocessing.py)

### **Composition**
| Property | Value |
|----------|-------|
| **Total Classes** | 248 |
| **Tamil Characters** | 247 (unique hand signs) |
| **Background Class** | 1 (same folder → different hand positions) |
| **Images per Class** | ~1,000 images |
| **Total Images** | ~248,000 images |
| **Image Format** | JPEG/PNG |
| **Image Resolution** | Variable (resized to 224×224) |
| **Folder Naming** | Numeric: "1", "2", ..., "247", "Background" |

### **Folder Structure Example**
```
TLFS23 - Tamil Language Finger Spelling Image Dataset/
Dataset Folders/
├── 1/          → அ (a-vowel)
├── 2/          → ஆ (aa-vowel)
├── 3/          → இ (i-vowel)
├── ...
├── 247/        → னௌ (nu-consonant vowel)
└── Background/ → Non-hand images
```

### **Dataset Split Strategy**
- **Training**: 80% (for learning)
- **Validation**: 10% (for tuning hyperparameters)
- **Testing**: 10% (for final evaluation)

This split is **stratified** (maintains class proportions) and **shuffled** deterministically for reproducibility.

---

## 🧠 MODEL ARCHITECTURE

### **High-Level Overview**
```
Input Image (224×224×3 RGB)
    ↓
┌─────────────────────────────────────────┐
│   MobileNetV2 (ImageNet Pretrained)    │
│   • 88 layers                           │
│   • 3.5M parameters                     │
│   • Frozen base (Phase-1) or            │
│     Unfrozen last 40 layers (Phase-2)  │
└─────────────────────────────────────────┘
    ↓ (Output: 1280×7×7 feature map)
Global Average Pooling
    ↓ (Output: 1280 features)
Batch Normalization
    ↓
Dense Layer (256 units) + ReLU
    ↓
Dropout (40%)
    ↓
Dense Layer (248 units) + Softmax
    ↓
Output: Class Probabilities (248 values)
```

### **Model Details**
- **Base Model**: MobileNetV2 
  - Lightweight (~3.5M parameters)
  - Trained on ImageNet (1000 classes)
  - Good for real-time inference on CPU
  
- **Custom Classification Head**:
  - Global Average Pooling (reduces spatial dimensions)
  - Batch Normalization (stabilizes training)
  - Dense Layer (256 units) (learns class-specific features)
  - Dropout (40%) (prevents overfitting)
  - Output Layer (248 units) (one per class)

### **Why Transfer Learning?**
- **Problem**: Training from scratch requires millions of images & GPUs
- **Solution**: Use MobileNetV2 (already learned visual features like edges, shapes, textures)
- **Benefit**: Faster training, better accuracy with limited data

---

## 🚀 TRAINING PIPELINE

### **Phase 1: Frozen Base Training**
```
Locked:    MobileNetV2 Base (88 layers)
Trainable: Custom classification head only

┌─────────────────────────────────────────┐
│ Objective: Learn Tamil-specific patterns│
│ Learning Rate: 1e-3 (higher)            │
│ Epochs: ≤20 (early stopping)            │
│ Patience: 6 epochs without improvement  │
└─────────────────────────────────────────┘
```

### **Phase 2: Fine-tuning**
```
Locked:    First 48 layers of MobileNetV2
Trainable: Last 40 layers + classification head

┌─────────────────────────────────────────┐
│ Objective: Refine all layers            │
│ Learning Rate: 5e-5 (lower)             │
│ Epochs: ≤20 (early stopping)            │
│ Patience: 6 epochs without improvement  │
└─────────────────────────────────────────┘
```

### **Why Two Phases?**
1. **Phase 1**: Fast convergence, prevents base from forgetting ImageNet knowledge
2. **Phase 2**: Fine-tune low-level features for better accuracy

### **Loss & Metrics**
- **Loss Function**: Categorical Cross-Entropy (multi-class classification)
- **Optimizer**: Adam (adaptive learning rates)
- **Metrics Tracked**:
  - Training Accuracy
  - Validation Accuracy
  - Training Loss
  - Validation Loss

### **Callbacks (Automatic Actions)**
```
1. Early Stopping
   └── Stop if validation accuracy stops improving for 6 epochs
   
2. Model Checkpoint
   └── Save best model when validation accuracy improves
   
3. Reduce LR on Plateau
   └── Lower learning rate if stuck
   
4. TensorBoard
   └── Log training curves for visualization
```

---

## 🔄 COMPLETE WORKFLOW

### **STEP 1: DATA PREPARATION**
```
Raw Dataset (248 folders × ~1000 images)
    ↓ (preprocessing.py)
├── Validate dataset exists
├── Load image paths (non-copying, memory efficient)
├── Split into train/val/test (stratified)
├── Apply augmentation (rotation, shift, zoom, flip)
└── Create tf.data pipelines for batching & prefetching
    ↓
Training Dataset: 178,400 images
Validation Dataset: 22,300 images
Testing Dataset: 22,300 images
```

### **STEP 2: MODEL BUILDING**
```
MobileNetV2 Base (ImageNet weights)
    ↓ (train_model.py)
├── Freeze base layers
├── Add custom head (Dense → Batch Norm → Dense → Dropout → Dense)
├── Compile with Adam optimizer & categorical cross-entropy
└── Ready for training
```

### **STEP 3: PHASE-1 TRAINING (Frozen Base)**
```
For each epoch (≤20):
    For each batch (223 batches/epoch, 178K/32):
        ├── Forward pass (inference)
        ├── Compute loss
        ├── Backward pass (backpropagation)
        ├── Update classification head weights
        └── Update metrics
    ├── Validate on val set
    ├── Check early stopping
    └── Log to TensorBoard
```

### **STEP 4: PHASE-2 TRAINING (Fine-tuning)**
```
Unfreeze last 40 layers of MobileNetV2
    ↓
For each epoch (≤20):
    ├── Similar to Phase-1 but with lower learning rate
    └── Update ALL trainable layers
```

### **STEP 5: EVALUATION**
```
Test Set (22,300 images)
    ↓ (evaluate_model.py)
├── Load best model
├── Predict all test images
├── Compute metrics:
│   ├── Accuracy (% correct)
│   ├── Precision (false positives)
│   ├── Recall (false negatives)
│   ├── F1-Score (harmonic mean)
│   └── Per-class metrics
├── Generate confusion matrix (top 30 classes)
└── Save classification report
```

### **STEP 6: REAL-TIME INFERENCE**
```
Webcam Stream (30 FPS)
    ↓ (real_time_prediction.py)
├── Capture frame
├── Resize to 224×224
├── Normalize pixel values
├── Predict with loaded model
├── Apply majority voting (smooth predictions)
├── Check confidence threshold
├── Accumulate letters into words
├── Display overlay with letter + confidence
└── Log output
```

---

## 📄 FILE DESCRIPTIONS & PURPOSES

### **CORE PYTHON MODULES**

#### **1. `utils.py`** - Foundations & Mappings
**Purpose**: Label mapping & helper functions
**Key Contents**:
- `LABEL_MAP`: Complete mapping of folder numbers (1-247) to Tamil Unicode characters
- `IDX_TO_CHAR`: Maps 0-based class indices to Tamil characters
- `SORTED_FOLDERS`: Class folder names in alphabetical order (as Keras sees them)
- `is_background()`: Checks if prediction is background
- `build_index_to_label()`: Reconstructs class ordering (handles numeric string sorting quirk)

**Why Important**: Converts raw model outputs (class indices 0-247) to readable Tamil characters.

---

#### **2. `preprocessing.py`** - Dataset Handling
**Purpose**: Load, split, augment dataset without copying all files
**Key Functions**:
- `verify_dataset_root()`: Checks if dataset exists
- `collect_file_paths()`: Walks dataset, collects image paths & labels
- `split_paths()`: Stratified train/val/test split using scikit-learn
- `build_dataset_from_paths()`: Creates tf.data pipeline with augmentation
- `_make_decode_fn()`: Decodes JPEG, normalizes, augments images

**Key Parameters**:
- `IMG_SIZE = (224, 224)`: MobileNetV2 expects this size
- `BATCH_SIZE = 32`: Images processed together
- `SEED = 42`: Reproducible splits

**Why Important**: Efficient dataset pipeline. Instead of copying 250K+ images, it reads from original folders on-the-fly.

---

#### **3. `train_model.py`** - Model Training
**Purpose**: Build, train, and save model across 2 phases
**Key Functions**:
- `build_model()`: Constructs MobileNetV2 + custom head
- `train_phase_1()`: Freeze base, train head
- `train_phase_2()`: Unfreeze last 40 layers, fine-tune
- `main()`: Orchestrates full pipeline

**Key Hyperparameters**:
```python
PHASE1_EPOCHS = 20      # Phase 1 epochs
PHASE2_EPOCHS = 20      # Phase 2 epochs
PHASE1_LR = 1e-3        # Learning rate phase 1 (higher)
PHASE2_LR = 5e-5        # Learning rate phase 2 (lower)
FINETUNE_LAYERS = 40    # Layers to unfreeze
DROPOUT = 0.40          # Dropout rate (regularization)
DENSE_UNITS = 256       # Hidden layer size
```

**Callbacks Used**:
```python
EarlyStopping()          # Stop if val_acc plateaus
ModelCheckpoint()        # Save best model
ReduceLROnPlateau()      # Lower LR if stuck
TensorBoard()            # Log training curves
```

**Outputs**:
- `models/best_model.keras` - Best checkpoint
- `models/tamil_sign_model.keras` - Final model
- `models/class_indices.npy` - Class mapping
- `logs/<Phase>_<timestamp>/` - TensorBoard logs

---

#### **4. `evaluate_model.py`** - Test Set Evaluation
**Purpose**: Assess model performance on held-out test set
**Key Functions**:
- `load_class_names()`: Load ordered class names
- `evaluate()`: Run inference on test set, compute metrics
- `plot_confusion_matrix()`: Visualize top 30 classes

**Outputs**:
- `models/evaluation_report.txt` - Classification metrics table
- `models/confusion_matrix_top30.png` - Confusion matrix heatmap

**Metrics Computed**:
- Accuracy (% predictions correct)
- Precision (true positives / all positives predicted)
- Recall (true positives / all actual positives)
- F1-Score (harmonic mean of precision & recall)
- Support (samples per class)

---

#### **5. `real_time_prediction.py`** - Live Webcam Inference
**Purpose**: Real-time Tamil sign recognition from webcam
**Key Features**:
- Captures video frames (30 FPS default)
- Resizes to 224×224 normalize
- Predicts class with model
- Applies majority voting (smooth predictions over 15 frames)
- Confidence thresholding (ignores uncertain predictions)
- Accumulates letters into words
- Displays live overlay (letter + confidence + ring progress)

**Key Configurations**:
```python
IMG_SIZE = (224, 224)
CONFIDENCE_THRES = 0.65     # Min confidence to accept
VOTE_WINDOW = 15            # Smooth over 15 frames
ADD_LETTER_DELAY = 1.5      # Seconds to hold sign before adding
```

**Outputs**:
- OpenCV window with live video + overlays
- Console output (letter, confidence, word formed)
- `live_output.log` (tailable log file)

---

#### **6. `app.py`** - Flask Web Application
**Purpose**: Web interface for real-time recognition
**Key Routes**:
```
GET  /                     → Render index.html
GET  /video_feed           → MJPEG stream (video)
POST /predict              → Process frame, return JSON prediction
GET  /get_status           → Poll current state
```

**Features**:
- Webcam video streaming (MJPEG)
- Real-time predictions via JavaScript polling
- Dynamic UI updates
- Letter accumulation into words
- Confidence visualization
- Threading for concurrent operations

**Shared State** (thread-safe):
```python
_state = {
    "letter": "அ",              # Current letter
    "confidence": 0.92,         # Prediction confidence
    "is_background": False,     # Is it background?
    "sequence": ["அ", "ன"],    # Letters accumulated
    "collect_step": 5,          # Visual progress (1/7)
    "fps": 28.5,                # Frames per second
    "status": "Recognizing…"    # Status message
}
```

**Front-end** (HTML/CSS/JS):
- Displays video stream
- Shows current letter & confidence
- Displays accumulated word
- Ring progress indicator (visual feedback)

---

### **MODELS & WEIGHTS**

| File | Purpose | Size |
|------|---------|------|
| `best_model.keras` | Best checkpoint during training | ~14-15 MB |
| `tamil_sign_model.keras` | Final trained model | ~14-15 MB |
| `class_indices.npy` | Class index to folder mapping | Small |
| `class_names.npy` | Sorted class folder names | Small |
| `test_paths.npy` | Test image file paths | ~2-5 MB |
| `test_labels.npy` | Test image true labels | Small |

---

### **OUTPUTS & VISUALIZATIONS**

#### **evaluation_report.txt**
```
Classification Report
======================
             precision    recall  f1-score   support

    அ         0.95      0.93      0.94       120
    ஆ         0.92      0.91      0.91       115
    ...
weighted avg 0.89      0.89      0.89      22300
```

#### **training_curves.png**
Plots showing:
- Training Accuracy (upward trend)
- Validation Accuracy (follows training)
- Training Loss (downward trend)
- Validation Loss (follows training)

#### **confusion_matrix_top30.png**
Heatmap showing:
- Rows = True class
- Columns = Predicted class
- Darker colors = more predictions
- Diagonal = correct predictions

---

## 🎮 HOW TO RUN EVERYTHING

### **SETUP (One-time)**

#### 1. Install Python 3.9+
```bash
# Check Python version
python --version
```

#### 2. Create Virtual Environment
```bash
python -m venv tamil_env
tamil_env\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### **TRAINING PIPELINE**

#### **Full Training (Recommended)**
```bash
cd tamil_sign_recognition
python train_model.py
```
**Output**: 
- Best model: `models/best_model.keras`
- Final model: `models/tamil_sign_model.keras`
- Logs: `logs/Phase-1.../` and `logs/Phase-2.../`
- Duration: 30-60 min (GPU), 2-4 hours (CPU)

#### **Quick Test (Few Epochs)**
```bash
python train_model.py --quick
```
**Output**: Same as above but faster (5 epochs each phase)

#### **Skip Fine-tuning**
```bash
python train_model.py --no-finetune
```
**Output**: Only Phase-1 training, saves time

#### **Force Rebuild Dataset Split**
```bash
python train_model.py --force-split
```
**Output**: Recreates train/val/test split from scratch

---

### **EVALUATION**

#### **Test on Held-out Test Set**
```bash
python evaluate_model.py
```
**Output**:
- `models/evaluation_report.txt` (metrics table)
- `models/confusion_matrix_top30.png` (heatmap)
- Console output (accuracy per class)

#### **Use Custom Model**
```bash
python evaluate_model.py --model models/best_model.keras
```

---

### **REAL-TIME WEBCAM INFERENCE**

#### **Default Webcam (OpenCV Window)**
```bash
python real_time_prediction.py
```
**Controls**:
- `Q` key = Quit
- `SPACE` = Reset accumulated word
- `BACKSPACE` = Delete last letter

#### **Different Camera**
```bash
python real_time_prediction.py --camera 1
```

#### **Adjust Confidence Threshold**
```bash
python real_time_prediction.py --threshold 0.75
```
**Higher threshold** = More confident predictions, fewer letters

#### **Adjust Voting Window**
```bash
python real_time_prediction.py --window 20
```
**Larger window** = Smoother predictions, less responsive

#### **Terminal Only (No GUI)**
```bash
python real_time_prediction.py --no-display
```

---

### **WEB INTERFACE**

#### **Start Flask Server**
```bash
python app.py
```
**Output**:
```
 * Running on http://localhost:5000
 * Press CTRL+C to quit
```

#### **Access in Browser**
- Open: `http://localhost:5000`
- See live video feed
- Watch live predictions
- Accumulated word displayed

---

### **TENSORBOARD VISUALIZATION** (Advanced)

#### **View Training Curves**
```bash
# From project directory:
tensorboard --logdir=logs/
```
**Then open**: `http://localhost:6006`

**View**:
- Accuracy curves (training vs validation)
- Loss curves
- Timing information
- Histogram of weights

---

## 📊 OUTPUTS & RESULTS

### **Training Outputs**
| Output | Location | Purpose |
|--------|----------|---------|
| Best Model | `models/best_model.keras` | Checkpoint with highest val accuracy |
| Final Model | `models/tamil_sign_model.keras` | Final trained weights |
| Training Curves | `logs/<Phase>_<timestamp>/` | Accuracy/loss plots |
| Training Log | `training_log_full.txt` | Detailed epoch-by-epoch results |

### **Evaluation Outputs**
| Output | Location | Interpretation |
|--------|----------|-----------------|
| Classification Report | `models/evaluation_report.txt` | Per-class precision, recall, F1 |
| Confusion Matrix | `models/confusion_matrix_top30.png` | Which classes confuse the model |

### **Inference Outputs**
| Output | Type | Format |
|--------|------|--------|
| Live Predictions | Real-time | OpenCV window or web page |
| Log File | File | `live_output.log` |
| Accumulated Word | Display | "அனல்" (accumulated Tamil word) |
| Confidence Score | Display | 0.0 - 1.0 (certainty of prediction) |

---

## 🎓 EXAMPLE WORKFLOW: From Start to Finish

```
1. DATA PREPARATION
   └─ Dataset already organized in folders (1-247 + Background)
      └ preprocessing.py validates & splits it (80/10/10)

2. TRAINING
   └─ train_model.py
      ├─ Phase 1: Freeze base, train head (≤20 epochs)
      ├─ Phase 2: Unfreeze last 40 layers (≤20 epochs)
      └─ Save best_model.keras & tamil_sign_model.keras

3. EVALUATION
   └─ evaluate_model.py
      ├─ Load trained model
      ├─ Test on 22,300 test images
      └─ Report accuracy, confusion matrix, metrics

4. REAL-TIME USE (Choice of 2)
   ├─ Option A: OpenCV Window
   │  └─ real_time_prediction.py
   │     ├─ Capture webcam frames
   │     ├─ Predict each frame
   │     ├─ Smooth with majority voting
   │     └─ Accumulate into words
   │
   └─ Option B: Web Interface
      └─ app.py
         ├─ Flask server on localhost:5000
         ├─ Stream video via MJPEG
         ├─ Real-time AJAX predictions
         └─ Display accumulated word

5. OUTPUT
   └─ Recognized Tamil word displayed!
      (e.g., "அனல்" = fire/flame)
```

---

## 🔍 KEY TECHNICAL INSIGHTS

### **Why MobileNetV2?**
- **Small**: 3.5M parameters (vs ResNet: 25M)
- **Fast**: Runs on CPU in real-time
- **Accurate**: ImageNet pretrained (transfer learning)
- **Mobile-friendly**: Can deploy to phones

### **Why Transfer Learning?**
- Normal training: Need 1M+ images, months of GPU time
- Transfer learning: Use pretrained base, train head (days or hours)
- Result: Better accuracy with less data

### **Why Two Training Phases?**
- Phase 1: Quickly adapt base features to Tamil signs (faster convergence)
- Phase 2: Fine-tune everything while preserving base knowledge (better accuracy)

### **Why Majority Voting?**
- Single frame predictions: Noisy, fluctuate between similar classes
- Voting over 15 frames: Smooths predictions, more stable
- Result: Better user experience (letters don't flicker)

### **Why Confidence Threshold?**
- Low confidence predictions: Model is uncertain
- Threshold (0.65): Only accept if model is >65% confident
- Result: Fewer false positives, better accuracy

---

## 📈 EXPECTED PERFORMANCE

Based on TLFS23 dataset & two-phase training:
- **Validation Accuracy**: ~88-92%
- **Test Accuracy**: ~85-89%
- **Inference Speed**: 30+ FPS on CPU, 60+ FPS on GPU
- **Per-class Accuracy**: Range from 70% (hardest) to 99% (easiest)

---

## 🚀 NEXT STEPS TO IMPROVE

1. **Data Augmentation**: Add rotation, perspective warping
2. **Ensemble Models**: Combine MobileNetV2 + EfficientNet
3. **Fine-grained Features**: Use attention mechanisms
4. **Word Database**: Add spell-check for formed words
5. **Deployment**: Convert to ONNX or TFLite for mobile
6. **Real-time OCR**: Integrate with text-to-speech (IPC)

---

## 📚 REFERENCES

- **TensorFlow Docs**: https://tensorflow.org/api
- **Keras Documentation**: https://keras.io
- **MobileNetV2 Paper**: https://arxiv.org/abs/1801.04381
- **TLFS23 Dataset**: https://www.kaggle.com/datasets/

---

**Created**: March 2026  
**Framework**: TensorFlow 2.12+  
**Language**: Python 3.9+  
**Status**: Production-Ready
