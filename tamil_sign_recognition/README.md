# Tamil Hand Sign Recognition System

A deep-learning computer vision system that recognises **247 Tamil alphabet hand signs + background** from a live webcam feed using Transfer Learning (MobileNetV2) and TensorFlow/Keras.

## Project Structure

```
tamil_sign_recognition/
├── utils.py                  # Label map (folder → Tamil character), helpers
├── preprocessing.py          # Dataset loading, splitting, augmentation
├── train_model.py            # Model building and two-phase training
├── evaluate_model.py         # Test-set evaluation & confusion matrix
├── real_time_prediction.py   # Live webcam inference with word formation
├── models/                   # Saved models and evaluation artefacts
│   ├── best_model.keras
│   ├── tamil_sign_model.keras
│   ├── class_indices.npy
│   ├── training_curves.png
│   ├── confusion_matrix_top30.png
│   └── evaluation_report.txt
└── dataset_split/            # Auto-generated train/val/test split
    ├── train/
    ├── val/
    └── test/
```

## Dataset

| Property | Value |
|---|---|
| Source | TLFS23 – Tamil Language Finger Spelling Image Dataset |
| Classes | 248 (247 Tamil characters + 1 Background) |
| Images per class | ~1 000 |
| Image format | JPEG |
| Folder naming | Numeric (1–247 = Tamil chars, 248 = Background) |

## Requirements

```
pip install -r requirements.txt
```

Python ≥ 3.9, TensorFlow ≥ 2.12, OpenCV 4.x, NumPy, Matplotlib, scikit-learn.

## Quick Start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### 2 – Train the model

```bash
# Full training (recommended, GPU advised)
python train_model.py

# Quick smoke-test (fewer epochs)
python train_model.py --quick

# Skip fine-tuning phase
python train_model.py --no-finetune

# Force-rebuild the train/val/test split
python train_model.py --force-split
```

Training runs two phases:

| Phase | Description | Epochs |
|---|---|---|
| Phase 1 | Freeze MobileNetV2 base, train classification head | ≤ 20 |
| Phase 2 | Unfreeze last 40 layers, fine-tune end-to-end | ≤ 20 |

Early stopping (patience = 6) stops each phase early when validation accuracy plateaus.

### 3 – Evaluate on test set

```bash
python evaluate_model.py
```

Outputs:
- Per-class accuracy table
- Confusion matrix image (`models/confusion_matrix_top30.png`)
- Full classification report (`models/evaluation_report.txt`)

### 4 – Real-time webcam prediction

```bash
# Default camera (index 0), with OpenCV window
python real_time_prediction.py

# Use a different camera
python real_time_prediction.py --camera 1

# Adjust confidence threshold (default 0.65)
python real_time_prediction.py --threshold 0.70

# Larger smoothing window (default 15 frames)
python real_time_prediction.py --window 20

# Terminal only, no GUI window
python real_time_prediction.py --no-display
```

#### Terminal output example

```
Predicted Letter: அ  (conf: 0.93)
Predicted Letter: க  (conf: 0.88)
  >> Letter Added  : க
  >> Current Word  : அக
Predicted Letter: ம  (conf: 0.95)
  >> Letter Added  : ம
  >> Current Word  : அகம
────────────────────────────────────────
  Word Formed: அகம
────────────────────────────────────────
```

## Model Architecture

```
Input (224 × 224 × 3)
   │
MobileNetV2 backbone (ImageNet pretrained, ~2.2M params)
   │
GlobalAveragePooling2D
   │
BatchNormalization
   │
Dense(256, ReLU)
   │
Dropout(0.40)
   │
Dense(248, Softmax)   ← 247 Tamil chars + Background
```

## Prediction Smoothing

Two mechanisms reduce noise:

1. **Majority voting** – The predicted class over the last *N* frames (default 15) is selected via majority vote.
2. **Confidence threshold** – Predictions below the threshold (default 0.65) are treated as Background.

## Word Formation Logic

| Event | Action |
|---|---|
| Same letter stable for ≥ 1.5 s | Letter appended to current word |
| Background detected for ≥ 2.5 s | Current word finalised; new word starts |
| Press **C** in webcam window | Clear current word |
| Press **Q** or **ESC** | Exit; print session word summary |

## Training Tips

- A **NVIDIA GPU** will reduce training time from hours to ~30 min.
- If training on CPU: consider `--quick` first to verify the pipeline runs.
- Adjust `BATCH_SIZE` in `preprocessing.py` (or via `--batch-size`) if you run out of memory.
- The dataset split (80 % / 10 % / 10 %) is created once under `dataset_split/` using symlinks (Windows may copy files instead if symlinks are unavailable).

## Windows Notes

- Tamil characters may not display in the default Windows CMD. Use **Windows Terminal** (with a Unicode font such as Nirmala UI) for correct rendering.
- The code calls `SetConsoleOutputCP(65001)` automatically to set UTF-8 output.
- Run CMD or PowerShell **as Administrator** if symlinks fail (requires Developer Mode or admin rights).

## Technology Stack

| Component | Library |
|---|---|
| Deep learning | TensorFlow / Keras |
| Webcam capture | OpenCV |
| Data processing | NumPy |
| Visualisation | Matplotlib |
| Metrics | scikit-learn |
