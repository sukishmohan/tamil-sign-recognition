# preprocessing.py
# Data loading, augmentation, and dataset splitting for Tamil Sign Recognition
# Uses tf.keras.utils.image_dataset_from_directory directly on the original
# dataset to avoid copying 250K+ images. Splits are done in-memory via
# file-path lists, so dataset construction takes seconds not hours.

import os
import sys
import random
import glob

import numpy as np
import tensorflow as tf

# -- Configuration -------------------------------------------------------------
IMG_SIZE   = (224, 224)   # MobileNetV2 compatible
BATCH_SIZE = 32
SEED       = 42

# Dataset root - folder that contains the numbered sub-folders 1..247 + Background
DATASET_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "TLFS23 - Tamil Language Finger Spelling Image Dataset",
        "Dataset Folders",
    )
)

# Kept for backward compatibility
SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_split")


# -- Dataset verification ------------------------------------------------------

def verify_dataset_root(path=DATASET_ROOT):
    if not os.path.isdir(path):
        print(f"[ERROR] Dataset root not found:\n  {path}")
        return False
    folders = [d for d in os.listdir(path)
               if os.path.isdir(os.path.join(path, d))
               and (d.isdigit() or d == "Background")]
    if len(folders) < 10:
        print(f"[ERROR] Expected class sub-folders, found only {len(folders)}.")
        return False
    print(f"[OK] Dataset root verified - {len(folders)} class folders found.")
    return True


# -- In-memory split helper ----------------------------------------------------

def collect_file_paths(dataset_root=DATASET_ROOT):
    """
    Walk the dataset root and collect (filepath, class_index) pairs.
    Class names are sorted alphabetically (string sort) matching Keras.
    Returns: (all_paths, all_labels, sorted_class_names)
    """
    class_folders = sorted(
        [d for d in os.listdir(dataset_root)
         if os.path.isdir(os.path.join(dataset_root, d))
         and (d.isdigit() or d == "Background")],
        key=lambda x: x
    )

    all_paths  = []
    all_labels = []
    for label_idx, cls in enumerate(class_folders):
        cls_dir = os.path.join(dataset_root, cls)
        images  = (glob.glob(os.path.join(cls_dir, "*.jpg"))
                 + glob.glob(os.path.join(cls_dir, "*.jpeg"))
                 + glob.glob(os.path.join(cls_dir, "*.png")))
        for img_path in images:
            all_paths.append(img_path)
            all_labels.append(label_idx)

    return all_paths, all_labels, class_folders


def split_paths(all_paths, all_labels, val_ratio=0.10, test_ratio=0.10, seed=SEED):
    from sklearn.model_selection import train_test_split as sk_split
    tr_paths, te_paths, tr_labels, te_labels = sk_split(
        all_paths, all_labels,
        test_size=test_ratio,
        stratify=all_labels,
        random_state=seed,
    )
    val_frac = val_ratio / (1.0 - test_ratio)
    tr_paths, va_paths, tr_labels, va_labels = sk_split(
        tr_paths, tr_labels,
        test_size=val_frac,
        stratify=tr_labels,
        random_state=seed,
    )
    return tr_paths, tr_labels, va_paths, va_labels, te_paths, te_labels


# -- tf.data pipeline ----------------------------------------------------------

def _make_decode_fn(img_size, num_classes, augment=False):
    h, w = img_size

    @tf.function
    def decode(path, label):
        raw   = tf.io.read_file(path)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, [h, w])
        image = tf.cast(image, tf.float32) / 255.0
        if augment:
            image = tf.image.random_brightness(image, 0.20)
            image = tf.image.random_contrast(image, 0.80, 1.20)
            # No horizontal flip - hand signs are not symmetric
            image = tf.image.random_crop(
                tf.image.resize_with_crop_or_pad(image, h + 20, w + 20),
                [h, w, 3]
            )
        label_oh = tf.one_hot(label, num_classes)
        return image, label_oh

    return decode


def build_dataset_from_paths(paths, labels, num_classes,
                              img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                              augment=False, shuffle=False):
    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=SEED)
    decode_fn = _make_decode_fn(img_size, num_classes, augment=augment)
    ds = ds.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# -- High-level entry point (used by train_model.py) --------------------------

def _stratified_subsample(paths, labels, ratio, seed=SEED):
    """Return a stratified random subsample of (paths, labels)."""
    from collections import defaultdict
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for p, l in zip(paths, labels):
        buckets[l].append(p)
    num_classes = len(buckets)
    n_total = max(num_classes, int(len(paths) * ratio))
    n_per_class = max(1, n_total // num_classes)
    sub_p, sub_l = [], []
    for cls, cls_paths in sorted(buckets.items()):
        chosen = rng.sample(cls_paths, min(n_per_class, len(cls_paths)))
        sub_p.extend(chosen)
        sub_l.extend([cls] * len(chosen))
    return sub_p, sub_l


def get_datasets(
    dataset_root = DATASET_ROOT,
    img_size     = IMG_SIZE,
    batch_size   = BATCH_SIZE,
    val_ratio    = 0.10,
    test_ratio   = 0.10,
    subset_ratio = 1.0,   # <1.0 uses a stratified subset (for quick/CPU mode)
):
    print("[INFO] Collecting image paths...")
    all_paths, all_labels, class_names = collect_file_paths(dataset_root)
    num_classes = len(class_names)
    print(f"[INFO] Total images  : {len(all_paths)}")
    print(f"[INFO] Total classes : {num_classes}")

    print("[INFO] Splitting into train / val / test...")
    tr_p, tr_l, va_p, va_l, te_p, te_l = split_paths(
        all_paths, all_labels, val_ratio, test_ratio
    )

    if subset_ratio < 1.0:
        tr_p, tr_l = _stratified_subsample(tr_p, tr_l, subset_ratio)
        va_p, va_l = _stratified_subsample(va_p, va_l, subset_ratio)
        print(f"[INFO] Subset ({subset_ratio*100:.0f}%): {len(tr_p)} train / {len(va_p)} val images")
    else:
        print(f"[INFO] Train   : {len(tr_p)} images")
        print(f"[INFO] Val     : {len(va_p)} images")
    print(f"[INFO] Test    : {len(te_p)} images")

    train_ds = build_dataset_from_paths(
        tr_p, tr_l, num_classes, img_size, batch_size, augment=True,  shuffle=True
    )
    val_ds = build_dataset_from_paths(
        va_p, va_l, num_classes, img_size, batch_size, augment=False, shuffle=False
    )
    test_ds = build_dataset_from_paths(
        te_p, te_l, num_classes, img_size, batch_size, augment=False, shuffle=False
    )

    # Save split metadata for evaluate_model.py
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "test_paths.npy"),  np.array(te_p))
    np.save(os.path.join(cache_dir, "test_labels.npy"), np.array(te_l))
    np.save(os.path.join(cache_dir, "class_names.npy"), np.array(class_names))

    return train_ds, val_ds, test_ds, class_names, num_classes, len(tr_p), len(va_p)


# -- Compatibility stubs -------------------------------------------------------

def create_split_dataset(*args, **kwargs):
    train_dir = os.path.join(SPLIT_DIR, "train")
    val_dir   = os.path.join(SPLIT_DIR, "val")
    test_dir  = os.path.join(SPLIT_DIR, "test")
    return train_dir, val_dir, test_dir


# -- Standalone smoke test -----------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print(" Tamil Sign Recognition - Data Preprocessing Check")
    print("=" * 60)
    if not verify_dataset_root():
        sys.exit(1)
    train_ds, val_ds, test_ds, class_names, nc, n_tr, n_va = get_datasets(batch_size=4)
    print("\n[INFO] Fetching a test batch...")
    for images, labels in train_ds.take(1):
        print(f"  Image batch shape : {images.shape}")
        print(f"  Label batch shape : {labels.shape}")
    print("\n[DONE] Preprocessing check complete.")
