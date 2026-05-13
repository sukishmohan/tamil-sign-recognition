# Tamil Sign Language Recognition - Architecture & Data Flow

## 🏗️ SYSTEM ARCHITECTURE DIAGRAM

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE SYSTEM ARCHITECTURE                         │
└──────────────────────────────────────────────────────────────────────────────┘

                              DATA PIPELINE
                              ════════════

    ORIGINAL DATASET                    PREPROCESSING
    ────────────────                    ─────────────
    
    TLFS23 Dataset                      🐍 preprocessing.py
    248 Folders                         ├─ Load images from disk
    ~248K Images                        ├─ Collect file paths (non-copy)
          │                             ├─ Stratified split:
          │                             │  ├─ Train (80%) = 178,400
          │                             │  ├─ Val   (10%) = 22,300
          │                             │  └─ Test  (10%) = 22,300
          │                             └─ Create tf.data pipeline
          │                                ├─ Batch = 32
          │                                ├─ Resize = 224×224
          │                                └─ Augment (rotate, flip, zoom)
          │
          └────────────────────────────────┬─────────────────────────────────→
                                           │
                              ┌────────────▼────────────┐
                              │   Dataset Pipelines     │
                              │  (tf.data.Dataset)      │
                              ├────────────────────────┤
                              │ train_ds (223 batches)  │
                              │ val_ds   (28 batches)   │
                              │ test_ds  (28 batches)   │
                              └────────────────────────┘


═════════════════════════════════════════════════════════════════════════════════
                              TRAINING PIPELINE
                              ═════════════════

    ┌─────────────────────────────────────┐
    │   🐍 train_model.py                 │
    │   build_model()                     │
    └──────────────┬──────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────────┐
    │        Model Architecture                    │
    ├──────────────────────────────────────────────┤
    │  Input: 224×224×3 RGB Image                 │
    │          │                                   │
    │          ▼                                   │
    │  ┌──────────────────────────────────┐       │
    │  │ MobileNetV2 (Frozen in Phase-1) │       │
    │  │ • 88 layers                      │       │
    │  │ • ImageNet weights               │       │
    │  │ • Output: 1280 features (7×7)    │       │
    │  └──────────────────────────────────┘       │
    │          │                                   │
    │          ▼                                   │
    │  Global Average Pooling                     │
    │          │                                   │
    │          ▼  (1280 values)                    │
    │  Batch Normalization                        │
    │          │                                   │
    │          ▼                                   │
    │  Dense(256) + ReLU                          │
    │          │                                   │
    │          ▼                                   │
    │  Dropout(0.40)                              │
    │          │                                   │
    │          ▼                                   │
    │  Dense(248) + Softmax                       │
    │          │                                   │
    │          ▼                                   │
    │  Output: 248 class probabilities            │
    │  (0.0 to 1.0, sum = 1.0)                    │
    └──────────────────────────────────────────────┘


    PHASE-1 TRAINING (Frozen Base)
    ──────────────────────────────
    
    Trainable Layers:
    └─ Custom Classification Head Only
       (Dense + Batch Norm + Dropout + Output)
    
    For each epoch (≤20):
        ├─ Load training batch (32 images)
        ├─ Forward pass (compute output)
        ├─ Compute loss (categorical cross-entropy)
        ├─ Backward pass (backpropagation)
        ├─ Update head weights (Adam optimizer)
        ├─ Validate on val set
        ├─ Early stopping check (patience=6)
        └─ Save checkpoint if val_acc improves
    
    Hyperparameters:
    ├─ Learning Rate: 1e-3 (higher = bigger steps)
    ├─ Batch Size: 32
    ├─ Loss: Categorical Cross-Entropy
    └─ Optimizer: Adam
    
    Duration: ~1-3 hours (CPU), ~20 min (GPU)
    Expected Val Accuracy: ~85-88%


    PHASE-2 TRAINING (Fine-tuning)
    ──────────────────────────────
    
    Trainable Layers:
    └─ Last 40 layers of MobileNetV2 + Head
    
    For each epoch (≤20):
        ├─ Same as Phase-1
        └─ But with LOWER learning rate (5e-5)
    
    Hyperparameters:
    ├─ Learning Rate: 5e-5 (lower = smaller steps)
    ├─ Batch Size: 32 (same)
    ├─ Loss: Categorical Cross-Entropy (same)
    └─ Optimizer: Adam (same)
    
    Duration: ~1-3 hours (CPU), ~20 min (GPU)
    Expected Val Accuracy: ~88-92%
    
    Output → best_model.keras, tamil_sign_model.keras


═════════════════════════════════════════════════════════════════════════════════
                              EVALUATION PIPELINE
                              ═════════════════════

    Test Set Input
    22,300 images
         │
         ▼
    ┌───────────────────┐
    │ evaluate_model.py │
    └─────────┬─────────┘
              │
              ▼
    Load trained model
    (best_model.keras or tamil_sign_model.keras)
              │
              ▼
    For each test image:
        ├─ Preprocess (resize, normalize)
        ├─ Predict (model inference)
        ├─ Get predicted class
        ├─ Get true class (from test_labels.npy)
        └─ Compare for metrics
              │
              ▼
    Compute Metrics:
    ├─ Overall Accuracy: (correct / total) × 100
    ├─ Per-class Metrics:
    │  ├─ Precision: True Positive / (True Pos + False Pos)
    │  ├─ Recall:    True Positive / (True Pos + False Neg)
    │  └─ F1-Score:  2 × (Precision × Recall) / (Precision + Recall)
    └─ Confusion Matrix: 248×248 (what was predicted vs. true)
              │
              ▼
    Output:
    ├─ evaluation_report.txt (metrics table)
    ├─ confusion_matrix_top30.png (heatmap)
    └─ Console output (summary)

    Expected Test Accuracy: 85-89%


═════════════════════════════════════════════════════════════════════════════════
                          INFERENCE PIPELINE
                          ═══════════════════

    TWO OPTIONS:


    OPTION A: REAL-TIME WEBCAM (real_time_prediction.py)
    ──────────────────────────────────────────────────────
    
    Webcam
       │ (30 frames/sec)
       │
       ▼
    ┌────────────────────────┐
    │ Read frame (640×480)   │
    └─────────────┬──────────┘
                  │
                  ▼
    ┌────────────────────────┐
    │ Preprocess             │
    │ ├─ Resize to 224×224   │
    │ ├─ Normalize (±1 std)  │
    │ └─ Convert to tensor   │
    └─────────────┬──────────┘
                  │
                  ▼
    ┌────────────────────────┐
    │ Model Inference        │
    │ → 248 probabilities    │
    └─────────────┬──────────┘
                  │
                  ▼
    ┌────────────────────────┐
    │ Majority Voting        │
    │ (over 15-frame window) │
    │ Smooth predictions     │
    │ & reduce noise         │
    └─────────────┬──────────┘
                  │
                  ▼
    ┌────────────────────────┐
    │ Confidence Check       │
    │ if score ≥ 0.65        │
    │    Accept              │
    │ else                   │
    │    Reject (Background) │
    └─────────────┬──────────┘
                  │
                  ▼
    ┌────────────────────────┐
    │ Hold Time Check        │
    │ (1.5 sec required)     │
    │ to avoid noise         │
    └─────────────┬──────────┘
                  │
                  ▼
    ┌────────────────────────┐
    │ Add to Word            │
    │ sequence += letter     │
    │ e.g., ["அ"] →         │
    │       ["அ", "ன"]      │
    └─────────────┬──────────┘
                  │
                  ▼
    ┌────────────────────────┐
    │ Display Overlay        │
    │ ├─ Letter (அ)          │
    │ ├─ Confidence (92%)     │
    │ ├─ Ring progress       │
    │ └─ Accumulated word    │
    └─────────────┬──────────┘
                  │
                  ▼
    OpenCV Window + Console Output
    "அனல்" (word formed!)


    OPTION B: WEB INTERFACE (app.py)
    ────────────────────────────────
    
    Webcam
       │
       ▼
    ┌─────────────────────────────────┐
    │ Flask Server (localhost:5000)   │
    │                                 │
    │ GET  / → Serve HTML/CSS/JS      │
    │ GET  /video_feed → MJPEG stream │
    │ POST /predict → JSON response   │
    │ GET  /get_status → Current state│
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │ Threading                        │
    │ ├─ Main thread: Flask server     │
    │ ├─ Camera thread: Read frames    │
    │ └─ Prediction thread: Inference  │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │ Same inference logic as Option A │
    │ + Thread-safe state sharing      │
    └──────────────┬───────────────────┘
                   │
                   ▼
    Browser Window @ localhost:5000
    ├─ Live MJPEG video stream
    ├─ Current letter (அ)
    ├─ Confidence bar
    ├─ Ring progress indicator
    └─ Accumulated word display


═════════════════════════════════════════════════════════════════════════════════
                         LABEL MAPPING (utils.py)
                         ═════════════════════════

    Folder Number (1-248) 
            │
            ▼
    ┌──────────────────────────────┐
    │ LABEL_MAP Dictionary         │
    │ {folder: tamil_character}    │
    │ 1 → 'அ'                       │
    │ 2 → 'ஆ'                       │
    │ ...                          │
    │ 247 → 'னௌ'                    │
    │ 248 → 'Background'           │
    └──────────────────┬───────────┘
                       │
                       ▼
                Keras sorts folders
                alphabetically
                "1", "10", "11", ..., "2", "20", ...
                       │
                       ▼
    ┌──────────────────────────────┐
    │ SORTED_FOLDERS               │
    │ (0-based indices)            │
    │ 0   → "1"  (அ)               │
    │ 1   → "10" (ஒ)               │
    │ 2   → "100" (Error!)         │
    │ ...                          │
    │ 247 → "Background"           │
    └──────────────────┬───────────┘
                       │
                       ▼
    ┌──────────────────────────────┐
    │ IDX_TO_CHAR                  │
    │ (0-based → Tamil char)       │
    │ 0   → 'அ'                     │
    │ 1   → 'ஒ'                     │
    │ 2   → 'ஒ'                     │
    │ ... (handles numeric ordering)│
    │ 247 → 'Background'           │
    └──────────────────────────────┘


═════════════════════════════════════════════════════════════════════════════════
                            FILE ORGANIZATION
                            ═════════════════

    📁 tamil_sign_recognition/
    │
    ├─ 🐍 PYTHON MODULES (Processing & Training)
    │  ├─ utils.py → Label mappings, helpers
    │  ├─ preprocessing.py → Dataset pipeline
    │  ├─ train_model.py → Training orchestration
    │  ├─ evaluate_model.py → Test evaluation
    │  ├─ real_time_prediction.py → Live inference
    │  └─ app.py → Flask web interface
    │
    ├─ 📊 MODELS (Trained Weights)
    │  └─ models/
    │     ├─ tamil_sign_model.keras (final)
    │     ├─ best_model.keras (checkpoint)
    │     ├─ class_*.npy (mappings)
    │     ├─ test_*.npy (evaluation data)
    │     ├─ evaluation_report.txt
    │     └─ confusion_matrix_*.png
    │
    ├─ 📈 LOGS (Training History)
    │  └─ logs/
    │     ├─ Phase-1 (Frozen Base)_<timestamp>/
    │     └─ Phase-2 (Fine-tuning)_<timestamp>/
    │
    ├─ 🎯 DATASET (Virtual, Non-copying)
    │  └─ dataset_split/
    │     ├─ train/ (file paths, not copies)
    │     ├─ val/
    │     └─ test/
    │
    ├─ 🌐 WEB (Frontend)
    │  ├─ templates/
    │  │  └─ index.html (UI)
    │  └─ static/
    │     ├─ style.css (styling)
    │     └─ app.js (JavaScript)
    │
    ├─ 📄 DOCS
    │  ├─ README.md (quick start)
    │  ├─ requirements.txt (dependencies)
    │  ├─ training_log.txt
    │  └─ training_log_full.txt
    │
    └─ __pycache__/ (compiled Python files)


═════════════════════════════════════════════════════════════════════════════════
                            EXECUTION FLOW
                            ══════════════

    USER COMMAND
         │
         ├─── python train_model.py ─────────┐
         │                                     │
         │    1. verify_dataset_root()         │
         │    2. collect_file_paths()          │
         │    3. split_paths()                 │
         │    4. build_model()                 │
         │    5. train_phase_1()               │
         │    6. train_phase_2()               │
         │    7. Save models                   │
         │                                     └─ *.keras files
         │
         ├─── python evaluate_model.py ───────┐
         │                                     │
         │    1. Load best model               │
         │    2. Load test set                 │
         │    3. Predict on all samples        │
         │    4. Compute metrics               │
         │    5. Generate plots                │
         │                                     └─ *.txt, *.png
         │
         ├─── python real_time_prediction.py ┐
         │                                     │
         │    1. Load model + class map        │
         │    2. Open webcam                   │
         │    3. For each frame:               │
         │       ├─ Preprocess                 │
         │       ├─ Predict                    │
         │       ├─ Majority vote              │
         │       ├─ Check confidence           │
         │       ├─ Accumulate letter          │
         │       └─ Display overlay            │
         │                                     └─ OpenCV window
         │
         └─── python app.py ─────────────────┐
                                              │
             1. Load model + class map        │
             2. Start Flask server            │
             3. Open browser (localhost:5000)│
             4. Same logic as Option A       │
                but via HTTP/MJPEG            └─ Web interface


═════════════════════════════════════════════════════════════════════════════════
                        HYPERPARAMETER SUMMARY
                        ═══════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ TRAINING CONFIGURATION                                                      │
├──────────────────────────────┬───────────────┬──────────────────────────────┤
│ Parameter                    │ Value         │ Purpose                      │
├──────────────────────────────┼───────────────┼──────────────────────────────┤
│ Image Size                   │ 224×224       │ MobileNetV2 input dimension  │
│ Batch Size                   │ 32            │ Images per gradient update   │
│ Phase-1 LR                   │ 1e-3          │ Head-only learning rate      │
│ Phase-2 LR                   │ 5e-5          │ Fine-tuning learning rate    │
│ Phase-1 Epochs               │ ≤20           │ Max epochs (early stopping)  │
│ Phase-2 Epochs               │ ≤20           │ Max epochs (early stopping)  │
│ Fine-tune Layers             │ 40            │ Layers to unfreeze in Ph2    │
│ Dropout Rate                 │ 0.40          │ Regularization strength      │
│ Dense Units (hidden)         │ 256           │ Hidden layer size            │
│ Early Stopping Patience      │ 6 epochs      │ Stop if no improvement       │
│ Confidence Threshold         │ 0.65          │ Min confidence to accept     │
│ Voting Window                │ 15 frames     │ For smooth predictions       │
│ Hold Time                    │ 1.5 seconds   │ Before adding letter         │
└──────────────────────────────┴───────────────┴──────────────────────────────┘
```

---

## 🔗 MODULE DEPENDENCY GRAPH

```
                        External Libraries
                        ══════════════════
    numpy ────────┐
    tensorflow ───┼────────┐
    opencv ───────┤        │
    pillow ────────        │
    matplotlib ────        │
    scikit-learn ──        │
                           │
                           ▼
    ┌──────────────────────────────┐
    │  PREPROCESSING.PY            │◄─── Raw Dataset
    │  • Load images               │
    │  • Split train/val/test      │
    │  • Augmentation              │
    │  • tf.data pipeline          │
    │  Output: Dataset objects     │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  UTILS.PY                    │
    │  • Label mapping             │
    │  • Index ↔ Char conversion   │
    │  • Helper functions          │
    │  Imports: None local modules │
    └──────────────┬───────────────┘
                   │
           (used by all modules)
                   │
    ┌──────┬──────┬──────┬────────┬────────┐
    │      │      │      │        │        │
    ▼      ▼      ▼      ▼        ▼        ▼
┌─────┐┌──────┐┌──────┐┌────────┐┌──────┐┌──────┐
│TM   ││EVA   ││RTP   ││APP     ││      ││      │
│TRAIN││EVAL  ││REAL_ ││FLASK  ││      ││      │
└─────┘└──────┘└──────┘└────────┘└──────┘└──────┘

Where:
- TRAIN = train_model.py (imports preprocessing + utils)
- EVAL = evaluate_model.py (imports preprocessing + utils)
- RTP = real_time_prediction.py (imports utils)
- APP = app.py (imports utils)
```

---

This architecture ensures:
✓ **Modularity**: Each component has a single responsibility
✓ **Reusability**: Utils used by all modules
✓ **Maintainability**: Easy to modify individual components
✓ **Scalability**: Can add new inference backends easily
