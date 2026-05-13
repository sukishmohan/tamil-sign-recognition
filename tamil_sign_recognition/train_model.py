# train_model.py
# Build, train, and save the Tamil Sign Recognition model.
# Uses MobileNetV2 (transfer learning) + optional fine-tuning phase.
#
# Usage:
#   python train_model.py                 # full training pipeline
#   python train_model.py --quick         # fewer epochs (smoke test)
#   python train_model.py --force-split   # recreate train/val/test split

import os
import sys
import argparse
import datetime

# Force UTF-8 output on Windows to handle Tamil chars in print statements
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for all OS)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import (
    DATASET_ROOT, IMG_SIZE, BATCH_SIZE,
    verify_dataset_root, get_datasets,
)
from utils import NUM_CLASSES, IDX_TO_CHAR, SORTED_FOLDERS

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR  = os.path.dirname(__file__)
MODELS_DIR   = os.path.join(PROJECT_DIR, "models")
LOG_DIR      = os.path.join(PROJECT_DIR, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

BEST_MODEL_PATH  = os.path.join(MODELS_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "tamil_sign_model.keras")
CLASS_MAP_PATH   = os.path.join(MODELS_DIR, "class_indices.npy")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
PHASE1_EPOCHS = 20      # Top-layer training (base frozen)
PHASE2_EPOCHS = 20      # Fine-tuning (last N base layers unfrozen)
FINETUNE_LAYERS = 40    # Number of MobileNetV2 layers to unfreeze for fine-tune
PHASE1_LR = 1e-3
PHASE2_LR = 5e-5
DROPOUT   = 0.40
DENSE_UNITS = 256

QUICK_PHASE1 = 5
QUICK_PHASE2 = 5


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(num_classes: int, img_size: tuple = IMG_SIZE) -> Model:
    """
    MobileNetV2 backbone with custom classification head.
    Phase-1: base is frozen.
    """
    base = MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False   # Freeze for Phase-1

    inputs = keras.Input(shape=(*img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(DENSE_UNITS, activation='relu')(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='TamilSignMobileNetV2')
    return model, base


def unfreeze_base(model: Model, base_model, num_layers: int = FINETUNE_LAYERS):
    """Unfreeze the last `num_layers` layers of the backbone for fine-tuning."""
    base_model.trainable = True
    # Freeze all layers except the last `num_layers`
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    print(f"[INFO] Unfrozen last {num_layers} layers of {base_model.name}.")
    return model


# ── Training routine ──────────────────────────────────────────────────────────

def compile_and_train(
    model, train_gen, val_gen,
    epochs: int,
    lr: float,
    phase_name: str,
    checkpoint_path: str,
):
    """Compile and run one training phase; returns the history object."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')],
    )
    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"  LR={lr}  Epochs≤{epochs}  Classes={model.output_shape[-1]}")
    print(f"{'='*60}")
    model.summary(print_fn=lambda s: None)   # suppress verbose summary to stdout

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(
            log_dir=os.path.join(LOG_DIR, f"{phase_name}_{timestamp}"),
            histogram_freq=0,
        ),
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_history(histories: list, labels: list, out_path: str):
    """Save accuracy & loss curves to a PNG file."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hist, label in zip(histories, labels):
        acc     = hist.history.get('accuracy', [])
        val_acc = hist.history.get('val_accuracy', [])
        loss    = hist.history.get('loss', [])
        val_loss= hist.history.get('val_loss', [])
        epochs  = range(1, len(acc) + 1)

        axes[0].plot(epochs, acc,     label=f'{label} Train')
        axes[0].plot(epochs, val_acc, linestyle='--', label=f'{label} Val')
        axes[1].plot(epochs, loss,     label=f'{label} Train')
        axes[1].plot(epochs, val_loss, linestyle='--', label=f'{label} Val')

    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Training curves saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train Tamil Sign Recognition model')
    parser.add_argument('--quick',       action='store_true',
                        help='Run with fewer epochs + 5%% data subset (CPU-friendly)')
    parser.add_argument('--force-split', action='store_true',
                        help='Force recreation of the train/val/test split')
    parser.add_argument('--no-finetune', action='store_true',
                        help='Skip Phase 2 (fine-tuning)')
    parser.add_argument('--batch-size',  type=int, default=BATCH_SIZE)
    parser.add_argument('--subset',      type=float, default=None,
                        help='Fraction of data to use (e.g. 0.05). '
                             'Defaults to 0.05 in --quick mode, 1.0 otherwise.')
    args = parser.parse_args()

    phase1_epochs = QUICK_PHASE1 if args.quick else PHASE1_EPOCHS
    phase2_epochs = QUICK_PHASE2 if args.quick else PHASE2_EPOCHS
    batch_size    = args.batch_size
    subset_ratio  = args.subset if args.subset is not None else (0.05 if args.quick else 1.0)

    # ── GPU setup ────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] GPU(s) available: {[g.name for g in gpus]}")
        except RuntimeError as e:
            print(f"[WARN] GPU config error: {e}")
    else:
        print("[INFO] No GPU detected - training on CPU (will be slow).")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if not verify_dataset_root():
        sys.exit(1)

    train_ds, val_ds, test_ds, class_names, num_classes, n_train, n_val = get_datasets(
        batch_size=batch_size,
        subset_ratio=subset_ratio,
    )
    print(f"[INFO] Number of classes for training: {num_classes}")

    # Save class name list so inference scripts can reload it
    class_names_path = os.path.join(MODELS_DIR, "class_names.npy")
    np.save(class_names_path, np.array(class_names))
    print(f"[INFO] Class names saved -> {class_names_path}")

    # ── Phase 1: Train classification head ───────────────────────────────────
    model, base_model = build_model(num_classes)
    hist1 = compile_and_train(
        model, train_ds, val_ds,
        epochs=phase1_epochs,
        lr=PHASE1_LR,
        phase_name='Phase-1 (Frozen Base)',
        checkpoint_path=BEST_MODEL_PATH,
    )

    histories = [hist1]
    labels    = ['Phase-1']

    # ── Phase 2: Fine-tune ───────────────────────────────────────────────────
    if not args.no_finetune:
        model = unfreeze_base(model, base_model, FINETUNE_LAYERS)
        hist2 = compile_and_train(
            model, train_ds, val_ds,
            epochs=phase2_epochs,
            lr=PHASE2_LR,
            phase_name='Phase-2 (Fine-tuning)',
            checkpoint_path=BEST_MODEL_PATH,
        )
        histories.append(hist2)
        labels.append('Phase-2')

    # ── Save final model ─────────────────────────────────────────────────────
    model.save(FINAL_MODEL_PATH)
    print(f"\n[OK] Final model saved -> {FINAL_MODEL_PATH}")
    print(f"[OK] Best model saved  -> {BEST_MODEL_PATH}")

    # ── Evaluate on test set ─────────────────────────────────────────────────
    print("\n[INFO] Evaluating on test set...")
    results = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names
    print("\n  Test Results:")
    for name, val in zip(metric_names, results):
        print(f"    {name:20s}: {val:.4f}")

    # ── Plot training curves ─────────────────────────────────────────────────
    plot_path = os.path.join(MODELS_DIR, "training_curves.png")
    plot_history(histories, labels, plot_path)

    print("\n[DONE] Training complete.")


if __name__ == '__main__':
    main()
