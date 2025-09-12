#!/usr/bin/env python3
"""
CNN+LSTM Video Emotion Classification (from scratch, no pretrained models)
— with strong regularization + dropout, class weighting, and data augmentation.

Usage (example):
    python train_cnn_lstm_video_emotion.py \
        --csv datatrain_clean.csv \
        --videos_dir videos \
        --id_col id \
        --label_col emotion_normalized \
        --frames 20 \
        --img_size 64 \
        --epochs 60 \
        --batch_size 8 \
        --val_split 0.2 \
        --use_class_weights \
        --augment \
        --l2 1e-3 \
        --cnn_dropout 0.4 \
        --lstm_dropout 0.5 \
        --dense_dropout 0.5 \
        --batchnorm \
        --cache_dir cache_npz

Notes:
- Expects a CSV with columns:
    - id (or set via --id_col): video file id (expects <id>.mp4 in --videos_dir)
    - emotion_normalized (or set via --label_col): string label
- Handles imbalance via --use_class_weights.
- Data augmentation happens on the fly in tf.data if --augment is set.
- Saves best model and classification report to disk.

if minority class tanks:
--oversample_minority

overfitting:
--cnn_width 24 --dropout 0.4 --augment

underfitting:
--cnn_width 48 --lstm_units 192 --frames 32

python train_cnn_lstm_video_emotion.py --csv datatrain_clean.csv --videos_dir videos --oversample_minority --frames 20 --img_size 64 --epochs 40 --batch_size 8 --use_class_weights --augment --l2 1e-3 --cnn_dropout 0.6 --lstm_dropout 0.6 --dense_dropout 0.6 --batchnorm --l2 0.01 --cnn_width 16 --lstm_units 64
"""

import os, sys, json, argparse, random
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# --------------------------
# Utilities
# --------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def sample_frame_indices(n_total: int, n_samples: int) -> np.ndarray:
    """Evenly sample n_samples indices from [0, n_total-1]."""
    if n_total <= 0:
        return np.array([], dtype=int)
    if n_total == n_samples:
        return np.arange(n_total)
    if n_total < n_samples:
        base = np.linspace(0, n_total - 1, num=n_total, dtype=int)
        pad = np.full((n_samples - n_total,), n_total - 1, dtype=int)
        return np.concatenate([base, pad], axis=0)
    return np.linspace(0, n_total - 1, num=n_samples, dtype=int)

def load_video_frames(video_path: str, num_frames: int = 20, target_size=(64, 64), bgr_to_rgb: bool = True) -> np.ndarray:
    """
    Load num_frames evenly-spaced frames, resize, normalize to [0,1].
    Returns shape: (num_frames, H, W, 3)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(max(0, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    idxs = sample_frame_indices(total, num_frames)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((*target_size, 3), dtype=np.float32))
            continue
        if bgr_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0)

def maybe_load_from_cache(cache_dir, vid, num_frames, img_size):
    if not cache_dir:
        return None
    path = os.path.join(cache_dir, f"{vid}__{num_frames}f_{img_size}sz.npz")
    if os.path.exists(path):
        try:
            data = np.load(path)
            return data["x"]
        except Exception:
            return None
    return None

def save_to_cache(cache_dir, vid, num_frames, img_size, arr):
    if not cache_dir:
        return
    ensure_dir(cache_dir)
    path = os.path.join(cache_dir, f"{vid}__{num_frames}f_{img_size}sz.npz")
    try:
        np.savez_compressed(path, x=arr)
    except Exception:
        pass

def load_dataset_from_csv(csv_path: str, videos_dir: str, id_col: str, label_col: str,
                          num_frames: int, img_size: int, cache_dir: str = None):
    df = pd.read_csv(csv_path)
    assert id_col in df.columns and label_col in df.columns, \
        f"CSV must contain '{id_col}' and '{label_col}'"

    X, y, kept_ids = [], [], []
    missing = 0
    for _, r in df.iterrows():
        vid = str(r[id_col])
        lab = str(r[label_col]).strip()
        mp4 = os.path.join(videos_dir, f"{vid}.mp4")
        if not os.path.exists(mp4):
            missing += 1
            continue

        cached = maybe_load_from_cache(cache_dir, vid, num_frames, img_size)
        if cached is None:
            arr = load_video_frames(mp4, num_frames=num_frames, target_size=(img_size, img_size))
            if arr is None or arr.shape[0] != num_frames:
                continue
            save_to_cache(cache_dir, vid, num_frames, img_size, arr)
        else:
            arr = cached

        X.append(arr)
        y.append(lab)
        kept_ids.append(vid)

    X = np.array(X, dtype=np.float32)  # (N, T, H, W, 3)
    y = np.array(y)
    print(f"[info] Loaded {len(X)} videos. Missing/failed: {missing}")
    return X, y, kept_ids

# --------------------------
# Model
# --------------------------

def conv_block(x, filters, l2, dropout_rate, use_bn):
    x = layers.TimeDistributed(
        layers.Conv2D(filters, 3, padding="same",
                      kernel_regularizer=regularizers.l2(l2),
                      use_bias=not use_bn)
    )(x)
    if use_bn:
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation("relu"))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    if dropout_rate and dropout_rate > 0:
        x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)
    return x

def build_cnn_lstm_model(num_classes: int, num_frames: int = 20, img_size: int = 64,
                         cnn_width: int = 32, lstm_units: int = 128, lr: float = 1e-4,
                         l2: float = 1e-3, cnn_dropout: float = 0.4,
                         lstm_dropout: float = 0.5, dense_dropout: float = 0.5,
                         use_bn: bool = True) -> tf.keras.Model:
    """
    From-scratch CNN+LSTM with strong regularization & dropout.
    - L2 applied to Conv/LSTM/Dense
    - Dropout after each block + LSTM + Dense
    - Optional BatchNorm in CNN
    """
    inputs = layers.Input(shape=(num_frames, img_size, img_size, 3))

    # CNN trunk (TimeDistributed)
    x = conv_block(inputs, cnn_width,     l2, cnn_dropout, use_bn)
    x = conv_block(x,      cnn_width * 2, l2, cnn_dropout, use_bn)
    x = conv_block(x,      cnn_width * 4, l2, cnn_dropout, use_bn)

    # Pool to (B, T, C)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    # Temporal modeling
    x = layers.LSTM(lstm_units, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2),
                    dropout=lstm_dropout, recurrent_dropout=0.0)(x)
    x = layers.LSTM(max(lstm_units // 2, 32),
                    kernel_regularizer=regularizers.l2(l2),
                    dropout=lstm_dropout, recurrent_dropout=0.0)(x)

    # Head
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    if dense_dropout and dense_dropout > 0:
        x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(l2))(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# --------------------------
# Data pipeline (tf.data) + augmentation
# --------------------------

def random_crop_and_resize(video, crop_frac_range=(0.9, 1.0), target_size=(64, 64)):
    """Random crop each frame by a small fraction then resize back — simulates zoom/translation."""
    h = tf.shape(video)[1]
    w = tf.shape(video)[2]
    frac = tf.random.uniform((), crop_frac_range[0], crop_frac_range[1])
    new_h = tf.cast(tf.cast(h, tf.float32) * frac, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * frac, tf.int32)

    # Center crop window start
    off_h = (h - new_h) // 2 + tf.random.uniform((), minval=-2, maxval=3, dtype=tf.int32)
    off_w = (w - new_w) // 2 + tf.random.uniform((), minval=-2, maxval=3, dtype=tf.int32)
    off_h = tf.clip_by_value(off_h, 0, h - new_h)
    off_w = tf.clip_by_value(off_w, 0, w - new_w)

    video = tf.image.crop_to_bounding_box(video, off_h, off_w, new_h, new_w)
    video = tf.image.resize(video, target_size, antialias=True)
    return video

def make_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int = 8, shuffle: bool = True,
                    augment: bool = False, img_size: int = 64) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)

    if augment:
        def _augment(video, label):
            # Random horizontal flip (whole sequence)
            do_flip = tf.less(tf.random.uniform(()), 0.5)
            video = tf.cond(do_flip,
                            lambda: tf.image.flip_left_right(video),
                            lambda: video)

            # Small random crop & resize back (simulates translation/zoom)
            video = random_crop_and_resize(video, (0.92, 1.0), (img_size, img_size))

            # Brightness & contrast jitter per frame
            noise = tf.random.uniform((), -0.05, 0.05)
            video = tf.clip_by_value(video + noise, 0.0, 1.0)
            video = tf.map_fn(lambda f: tf.image.random_contrast(f, 0.9, 1.1),
                              video, fn_output_signature=tf.float32)
            return video, label

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def oversample_minority_indices(y_train: np.ndarray, target_count: int = None) -> np.ndarray:
    label_to_idxs = defaultdict(list)
    for i, lab in enumerate(y_train):
        label_to_idxs[int(lab)].append(i)
    max_count = max(len(v) for v in label_to_idxs.values())
    if target_count is None:
        target_count = max_count
    new_indices = []
    for _, idxs in label_to_idxs.items():
        if len(idxs) >= target_count:
            new_indices.extend(random.sample(idxs, target_count))
        else:
            reps = target_count - len(idxs)
            new_indices.extend(idxs)
            new_indices.extend(np.random.choice(idxs, size=reps, replace=True).tolist())
    return np.array(new_indices, dtype=int)

# --------------------------
# Main
# --------------------------

def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--csv", required=True, default="datatrain_clean.csv", help="CSV with video ids + labels")
    p.add_argument("--videos_dir", required=True, default="videos", help="Directory with <id>.mp4 files")
    p.add_argument("--id_col", default="id", help="CSV column for video id")
    p.add_argument("--label_col", default="emotion_normalized", help="CSV column for label")
    p.add_argument("--frames", type=int, default=20, help="Frames per video")
    p.add_argument("--img_size", type=int, default=64, help="Square image size (pixels)")
    p.add_argument("--cache_dir", default=None, help="Cache extracted frames as npz here")
    # training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    # model sizes
    p.add_argument("--sdth", type=int, default=32)
    p.add_argument("--lstm_units", type=int, default=128)
    # regularization / dropout
    p.add_argument("--l2", type=float, default=1e-3, help="L2 weight decay")
    p.add_argument("--cnn_dropout", type=float, default=0.4)
    p.add_argument("--lstm_dropout", type=float, default=0.5)
    p.add_argument("--dense_dropout", type=float, default=0.5)
    p.add_argument("--batchnorm", action="store_true", help="Use BatchNorm in CNN blocks")
    # imbalance + augmentation
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--oversample_minority", action="store_true", help="Random oversample TRAIN split")
    p.add_argument("--augment", action="store_true", help="Apply data augmentation")
    # out
    p.add_argument("--out_dir", default="runs_cnnlstm", help="Outputs (model, reports)")
    args = p.parse_args()

    ensure_dir(args.out_dir)
    set_seed(args.seed)

    # 1) Load videos
    print("[step] Loading videos...")
    X, y_str, kept_ids = load_dataset_from_csv(
        csv_path=args.csv,
        videos_dir=args.videos_dir,
        id_col=args.id_col,
        label_col=args.label_col,
        num_frames=args.frames,
        img_size=args.img_size,
        cache_dir=args.cache_dir
    )
    if len(X) == 0:
        print("[error] No videos loaded. Check paths/CSV.")
        sys.exit(1)

    # 2) Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    class_names = list(le.classes_)
    num_classes = len(class_names)
    print(f"[info] Classes ({num_classes}): {class_names}")

    # 3) Train/val split
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X, y, kept_ids,
        test_size=args.val_split, random_state=args.seed, stratify=y
    )
    print(f"[info] Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")
    print("[info] Train distribution:", Counter(y_train))
    print("[info] Val distribution:", Counter(y_val))

    # 4) Optional oversampling (video-level)
    if args.oversample_minority:
        print("[step] Oversampling minority classes...")
        idxs = oversample_minority_indices(y_train)
        X_train, y_train = X_train[idxs], y_train[idxs]
        print("[info] New Train distribution:", Counter(y_train))

    # 5) Datasets
    ds_train = make_tf_dataset(X_train, y_train, batch_size=args.batch_size,
                               shuffle=True, augment=args.augment, img_size=args.img_size)
    ds_val   = make_tf_dataset(X_val,   y_val,   batch_size=args.batch_size,
                               shuffle=False, augment=False, img_size=args.img_size)

    # 6) Class weights
    class_weight = None
    if args.use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        print("[info] Class weights:", class_weight)

    # 7) Model
    print("[step] Building model...")
    model = build_cnn_lstm_model(
        num_classes=num_classes,
        num_frames=args.frames,
        img_size=args.img_size,
        cnn_width=args.cnn_width,
        lstm_units=args.lstm_units,
        lr=args.lr,
        l2=args.l2,
        cnn_dropout=args.cnn_dropout,
        lstm_dropout=args.lstm_dropout,
        dense_dropout=args.dense_dropout,
        use_bn=args.batchnorm
    )
    model.summary()

    # 8) Callbacks
    ckpt_path = os.path.join(args.out_dir, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", mode="max",
            save_best_only=True, save_weights_only=False, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, verbose=1, min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
    ]

    # 9) Train
    print("[step] Training...")
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs,
        class_weight=class_weight,
        verbose=1,
        callbacks=callbacks
    )

    # Save history
    hist_path = os.path.join(args.out_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    print(f"[info] Saved history: {hist_path}")

    # 10) Evaluation (VAL) — report + confusion matrix
    print("[step] Evaluating on validation set...")
    y_val_pred_probs = model.predict(ds_val)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)

    report = classification_report(y_val, y_val_pred, target_names=class_names, digits=4)
    print("\n=== Classification Report (VAL) ===")
    print(report)

    rep_path = os.path.join(args.out_dir, "classification_report_val.txt")
    with open(rep_path, "w") as f:
        f.write(report)
    print(f"[info] Saved report: {rep_path}")

    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(1.5 + 0.6*len(class_names), 1.5 + 0.6*len(class_names)))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Validation)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, "confusion_matrix_val.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[info] Saved confusion matrix: {cm_path}")

    # Save label classes
    le_path = os.path.join(args.out_dir, "label_encoder_classes.json")
    with open(le_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[info] Saved label classes: {le_path}")

    print(f"[done] Best model saved to: {ckpt_path}")

if __name__ == "__main__":
    main()
