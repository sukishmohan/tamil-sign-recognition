# evaluate_model.py
# Evaluate a trained Tamil Sign Recognition model on the held-out test set.
# Generates classification report, confusion matrix (sampled), and per-class
# accuracy table. Saves results to models/evaluation_report.txt.
#
# Usage:
#   python evaluate_model.py
#   python evaluate_model.py --model models/best_model.keras

import os
import sys
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import IMG_SIZE, BATCH_SIZE, build_dataset_from_paths
from utils import IDX_TO_CHAR, SORTED_FOLDERS, NUM_CLASSES, LABEL_MAP

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR     = os.path.dirname(__file__)
MODELS_DIR      = os.path.join(PROJECT_DIR, "models")
DEFAULT_MODEL   = os.path.join(MODELS_DIR, "tamil_sign_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.npy")
TEST_PATHS_PATH  = os.path.join(MODELS_DIR, "test_paths.npy")
TEST_LABELS_PATH = os.path.join(MODELS_DIR, "test_labels.npy")
REPORT_PATH     = os.path.join(MODELS_DIR, "evaluation_report.txt")
CONFMAT_PATH    = os.path.join(MODELS_DIR, "confusion_matrix_top30.png")


def load_class_names() -> list:
    """Load the ordered list of class folder names saved during training."""
    if os.path.exists(CLASS_NAMES_PATH):
        return list(np.load(CLASS_NAMES_PATH, allow_pickle=True))
    return SORTED_FOLDERS


def folder_to_char(folder_name: str) -> str:
    if folder_name == 'Background':
        return 'Background'
    try:
        return LABEL_MAP.get(int(folder_name), folder_name)
    except ValueError:
        return folder_name


def evaluate(model_path: str, batch_size: int = BATCH_SIZE):
    # ── Check saved test split ────────────────────────────────────────────────
    for p in [TEST_PATHS_PATH, TEST_LABELS_PATH]:
        if not os.path.exists(p):
            print(f"[ERROR] Test split file not found: {p}")
            print("        Run  python train_model.py  first to generate it.")
            sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        best = os.path.join(MODELS_DIR, "best_model.keras")
        if os.path.exists(best):
            model_path = best
        else:
            print(f"[ERROR] Model not found: {model_path}")
            sys.exit(1)
    print(f"[INFO] Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("[OK] Model loaded.")

    # ── Load test split ───────────────────────────────────────────────────────
    te_paths  = list(np.load(TEST_PATHS_PATH,  allow_pickle=True))
    te_labels = list(np.load(TEST_LABELS_PATH, allow_pickle=True))
    class_names = load_class_names()
    num_classes = len(class_names)
    idx_to_char = {i: folder_to_char(f) for i, f in enumerate(class_names)}

    print(f"[INFO] Test samples : {len(te_paths)} | Classes : {num_classes}")
    test_ds = build_dataset_from_paths(
        te_paths, te_labels, num_classes,
        img_size=IMG_SIZE, batch_size=batch_size,
        augment=False, shuffle=False,
    )

    # ── Overall metrics ───────────────────────────────────────────────────────
    print("\n[INFO] Running inference on test set (this may take a few minutes)...")
    t0 = time.time()
    results = model.evaluate(test_ds, verbose=1)
    elapsed = time.time() - t0
    metric_names = model.metrics_names

    print(f"\n  Inference time : {elapsed:.1f}s")
    print("  Test Results:")
    summary_lines = [
        "=" * 60,
        "  Tamil Sign Recognition - Evaluation Report",
        "=" * 60,
        f"  Model : {model_path}",
        f"  Test samples : {len(te_paths)}",
        f"  Inference time : {elapsed:.1f}s",
        "",
        "  Overall Metrics:",
    ]
    for name, val in zip(metric_names, results):
        line = f"    {name:30s}: {val:.4f}"
        print(line)
        summary_lines.append(line)

    # ── Per-class accuracy (Top-1) ────────────────────────────────────────────
    print("\n[INFO] Computing per-class accuracy...")
    all_true, all_pred = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        all_true.extend(np.argmax(y_batch.numpy(), axis=1).tolist())
        all_pred.extend(np.argmax(preds,           axis=1).tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    per_class_acc = []
    for c in range(num_classes):
        mask    = all_true == c
        correct = int(np.sum(all_pred[mask] == c)) if mask.any() else 0
        total   = int(mask.sum())
        acc     = correct / total if total > 0 else 0.0
        char    = idx_to_char.get(c, class_names[c])
        per_class_acc.append((char, acc, total))

    # Top-10 worst classes
    per_class_acc.sort(key=lambda x: x[1])
    summary_lines += [
        "",
        "  10 Worst-Performing Classes:",
        f"  {'Character':>12}  {'Accuracy':>10}  {'Samples':>8}",
    ]
    for char, acc, total in per_class_acc[:10]:
        line = f"  {char:>12}  {acc:>10.2%}  {total:>8}"
        summary_lines.append(line)

    per_class_acc.sort(key=lambda x: x[1], reverse=True)
    summary_lines += [
        "",
        "  10 Best-Performing Classes:",
        f"  {'Character':>12}  {'Accuracy':>10}  {'Samples':>8}",
    ]
    for char, acc, total in per_class_acc[:10]:
        line = f"  {char:>12}  {acc:>10.2%}  {total:>8}"
        summary_lines.append(line)

    # ── Top-5 accuracy ────────────────────────────────────────────────────────
    # Already in model.evaluate if metric was added during training

    # ── Confusion matrix (top-30 most confused classes) ───────────────────────
    print("\n[INFO] Building confusion matrix...")
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(all_true, all_pred)

    # Identify the 30 classes with most errors
    diag       = np.diag(cm)
    errors     = cm.sum(axis=1) - diag
    top30_idx  = np.argsort(errors)[-30:][::-1]
    cm_sub     = cm[np.ix_(top30_idx, top30_idx)]
    labels_sub = [idx_to_char.get(i, str(i)) for i in top30_idx]

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm_sub, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels_sub)))
    ax.set_yticks(range(len(labels_sub)))
    ax.set_xticklabels(labels_sub, rotation=90, fontsize=8)
    ax.set_yticklabels(labels_sub, fontsize=8)
    ax.set_title('Confusion Matrix - Top-30 Most Confused Classes')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(CONFMAT_PATH, dpi=120)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved -> {CONFMAT_PATH}")

    # ── Classification report (first 10 classes for brevity) ─────────────────
    cr = classification_report(
        all_true[:], all_pred[:],
        target_names=[idx_to_char.get(c, str(c)) for c in range(num_classes)],
        zero_division=0,
        output_dict=False,
    )
    summary_lines += [
        "",
        "  Classification Report (per class):",
        cr,
    ]

    # ── Save report ───────────────────────────────────────────────────────────
    report_text = '\n'.join(summary_lines)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n[OK] Full report saved -> {REPORT_PATH}")

    # Also print summary to terminal
    for line in summary_lines[:30]:
        print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Tamil Sign model')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='Path to the trained .keras model file')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    try:
        from sklearn.metrics import confusion_matrix, classification_report
    except ImportError:
        print("[ERROR] scikit-learn is required for evaluation:")
        print("        pip install scikit-learn")
        sys.exit(1)

    evaluate(args.model, args.batch_size)
