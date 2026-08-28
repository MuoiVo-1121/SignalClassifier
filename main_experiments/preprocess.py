"""
preprocess.py
-------------
Load raw photocurrent CSVs, apply temporal splitting (no data leakage),
sliding-window segmentation, global normalization, and save processed arrays.

Key design decisions
--------------------
1. TEMPORAL split (per-class, before windowing) — eliminates data leakage from
   overlapping windows. Test windows are always strictly later in time than any
   training window.
2. Asymmetric step sizes:
     Training   : step = 25 (75 % overlap) — more training windows from same data
     Val / Test : step = 50 (50 % overlap) — standard, no artificial inflation
3. Global min-max normalization across ALL samples (preserves DC offset).
4. Noise augmentation on training set (×2) — additive Gaussian noise simulating
   measurement variability across sessions.

Expected CSV format:
    time_s,current_nA,label
    0.00,0.12,0
    0.01,0.09,0
    ...

Label convention: 0 = dark, 1 = UVC lamp, 2 = methanol flame
"""

import os
import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
WINDOW_SIZE   = 100          # 1 second at 100 Hz
TRAIN_STEP    = 25           # 75 % overlap → more training samples
EVAL_STEP     = 50           # 50 % overlap → standard for val / test
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
# TEST_FRAC   = 0.15 (remainder)
AUG_NOISE_STD = 0.05         # additive Gaussian noise for augmentation (nA)
AUG_COPIES    = 1            # number of augmented copies per original window
RANDOM_SEED   = 42

CLASS_FILES = {
    0: "dark_recording.csv",
    1: "lamp_recording.csv",
    2: "flame_recording.csv",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def sliding_window(signal: np.ndarray, window_size: int,
                   step: int) -> np.ndarray:
    """Extract overlapping windows from a 1-D signal array."""
    n_windows = (len(signal) - window_size) // step + 1
    return np.array(
        [signal[i * step: i * step + window_size] for i in range(n_windows)],
        dtype=np.float32,
    )


def temporal_split(signal: np.ndarray):
    """Split a signal array into train / val / test by time index."""
    n = len(signal)
    train_end = int(n * TRAIN_FRAC)
    val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))
    return signal[:train_end], signal[train_end:val_end], signal[val_end:]


def augment(X: np.ndarray, y: np.ndarray,
            n_copies: int, noise_std: float,
            rng: np.random.Generator) -> tuple:
    """Create augmented copies of (X, y) with additive Gaussian noise."""
    X_parts = [X]
    y_parts = [y]
    for _ in range(n_copies):
        noise   = rng.normal(0.0, noise_std, size=X.shape).astype(np.float32)
        X_parts.append(X + noise)
        y_parts.append(y.copy())
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# ── Main preprocessing pipeline ───────────────────────────────────────────────
def preprocess(raw_dir: str       = RAW_DIR,
               processed_dir: str = PROCESSED_DIR):

    os.makedirs(processed_dir, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    X_train_list, X_val_list, X_test_list = [], [], []
    y_train_list, y_val_list, y_test_list = [], [], []

    print("=" * 62)
    print("Preprocessing pipeline (temporal split + augmentation)")
    print("=" * 62)

    # ── Step 1: Load, split, and window each class independently ────────────
    print(f"\n[1] Loading and windowing (T={WINDOW_SIZE})…")
    for label, fname in CLASS_FILES.items():
        path = os.path.join(raw_dir, fname)
        df   = pd.read_csv(path)
        sig  = df["current_nA"].values.astype(np.float32)

        tr_sig, va_sig, te_sig = temporal_split(sig)

        tr_X = sliding_window(tr_sig, WINDOW_SIZE, TRAIN_STEP)
        va_X = sliding_window(va_sig, WINDOW_SIZE, EVAL_STEP)
        te_X = sliding_window(te_sig, WINDOW_SIZE, EVAL_STEP)

        X_train_list.append(tr_X);  y_train_list.append(np.full(len(tr_X), label, dtype=np.int64))
        X_val_list.append(va_X);    y_val_list.append(np.full(len(va_X),   label, dtype=np.int64))
        X_test_list.append(te_X);   y_test_list.append(np.full(len(te_X),  label, dtype=np.int64))

        print(f"  Class {label} ({fname}): "
              f"train={len(tr_X):>4} | val={len(va_X):>3} | test={len(te_X):>3} windows")

    X_train = np.concatenate(X_train_list)
    X_val   = np.concatenate(X_val_list)
    X_test  = np.concatenate(X_test_list)
    y_train = np.concatenate(y_train_list)
    y_val   = np.concatenate(y_val_list)
    y_test  = np.concatenate(y_test_list)

    print(f"\n  Raw totals — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # ── Step 2: Global min-max normalization (fit on train only) ────────────
    print("\n[2] Global min-max normalization (train statistics)…")
    global_min = float(X_train.min())
    global_max = float(X_train.max())
    scale      = global_max - global_min
    if scale == 0:
        scale = 1.0

    def normalize(X):
        return (X - global_min) / scale

    X_train = normalize(X_train)
    X_val   = normalize(X_val)
    X_test  = normalize(X_test)

    np.save(os.path.join(processed_dir, "global_stats.npy"),
            np.array([global_min, global_max], dtype=np.float32))
    print(f"  Raw range  : [{global_min:.4f}, {global_max:.4f}] nA")
    print(f"  Norm range : [{X_train.min():.4f}, {X_train.max():.4f}]")

    # ── Step 3: Noise augmentation on training set only ─────────────────────
    print(f"\n[3] Noise augmentation on training set (×{AUG_COPIES + 1}, "
          f"σ={AUG_NOISE_STD} nA)…")
    X_train, y_train = augment(X_train, y_train, AUG_COPIES, AUG_NOISE_STD, rng)
    X_train = np.clip(X_train, 0.0, 1.0).astype(np.float32)
    print(f"  Augmented training set: {len(X_train)} windows")

    # ── Step 4: Shuffle training set ────────────────────────────────────────
    perm    = rng.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # ── Step 5: Reshape to (N, T, 1) ────────────────────────────────────────
    X_train = X_train[:, :, np.newaxis]
    X_val   = X_val[:, :, np.newaxis]
    X_test  = X_test[:, :, np.newaxis]

    # ── Step 6: Print split summary ─────────────────────────────────────────
    print(f"\n[4] Final split summary:")
    print(f"  Train : {X_train.shape}  — {X_train.shape[0]} windows (post-augmentation)")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")

    for split_name, yy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        uq, cts = np.unique(yy, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(uq, cts)}
        print(f"  {split_name} class distribution: {dist}")

    # ── Step 7: Save ─────────────────────────────────────────────────────────
    print("\n[5] Saving…")
    for fname, arr in [
        ("X_train.npy", X_train), ("X_val.npy",  X_val),  ("X_test.npy",  X_test),
        ("y_train.npy", y_train), ("y_val.npy",  y_val),  ("y_test.npy",  y_test),
    ]:
        np.save(os.path.join(processed_dir, fname), arr)
        print(f"  Saved {fname}  shape={arr.shape}")

    print(f"\n✓ Preprocessing complete → '{processed_dir}/'")
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    preprocess()
