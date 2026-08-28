"""
common_runner.py
----------------
Shared training/eval/resume machinery for the added experiments.
Protocol is the faithful replication of the original multiseed_train.py +
train.py (see capacity_compare.py docstring).
"""

import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import preprocess as pre_mod

WEIGHTS_DIR = "weights_added"
EXPORTS_DIR = "exports_added"
SEEDS       = [42, 123, 7, 2024, 99]
TRAIN_SEED  = 42
EPOCHS      = 80
BATCH_SIZE  = 64
LR          = 1e-3


def load_splits(processed_dir):
    def load(name):
        return np.load(os.path.join(processed_dir, name))
    return (load("X_train.npy"), load("X_val.npy"), load("X_test.npy"),
            load("y_train.npy"), load("y_val.npy"), load("y_test.npy"))


def run_inference(model, X, warmup: int = 5):
    """compare.py timing method: batch predict over the test set / N windows."""
    _ = model.predict(X[:warmup], verbose=0)
    t0    = time.perf_counter()
    probs = model.predict(X, verbose=0)
    ms    = (time.perf_counter() - t0) / len(X) * 1000
    return np.argmax(probs, axis=1), ms


def run_training(builder, run_key: str, seed: int,
                 processed_dir: str = "data/processed",
                 window_size: int | None = None,
                 train_step: int | None = None,
                 eval_step: int | None = None,
                 lr: float = LR) -> dict:
    """One preprocess+train+eval cycle. Returns a result dict."""
    print(f"\n{'-'*55}\n  run={run_key}  data_seed={seed}\n{'-'*55}", flush=True)

    # Patch preprocess globals (seed always; window geometry when given)
    saved = {k: getattr(pre_mod, k)
             for k in ("RANDOM_SEED", "WINDOW_SIZE", "TRAIN_STEP", "EVAL_STEP")}
    pre_mod.RANDOM_SEED = seed
    if window_size is not None:
        pre_mod.WINDOW_SIZE = window_size
    if train_step is not None:
        pre_mod.TRAIN_STEP = train_step
    if eval_step is not None:
        pre_mod.EVAL_STEP = eval_step
    try:
        pre_mod.preprocess(raw_dir="data/raw", processed_dir=processed_dir)
    finally:
        for k, v in saved.items():
            setattr(pre_mod, k, v)

    tf.random.set_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(processed_dir)

    model = builder(timesteps=X_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=20,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=8, min_lr=1e-6, verbose=0),
    ]
    t0 = time.perf_counter()
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=callbacks, verbose=0)
    train_s = time.perf_counter() - t0

    wdir = os.path.join(WEIGHTS_DIR, f"{run_key}_seed{seed}")
    os.makedirs(wdir, exist_ok=True)
    model.save_weights(os.path.join(wdir, "best.weights.h5"))

    y_pred, ms = run_inference(model, X_test)
    result = {
        "seed":          seed,
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "macro_f1":      round(float(f1_score(y_test, y_pred, average="macro")), 6),
        "best_val_acc":  round(float(max(history.history["val_accuracy"])), 6),
        "stopped_epoch": int(len(history.history["loss"])),
        "inference_ms":  round(float(ms), 4),
        "n_errors":      int((y_pred != y_test).sum()),
        "n_test":        int(len(y_test)),
        "train_seconds": round(train_s, 1),
    }
    print(f"  -> test_acc={result['test_accuracy']*100:.2f}%  "
          f"macro_f1={result['macro_f1']:.4f}  epochs={result['stopped_epoch']}  "
          f"train={train_s:.0f}s  ms/window={ms:.3f}", flush=True)
    return result


def summarize(params: int, runs: list, extra: dict | None = None) -> dict:
    accs = [r["test_accuracy"] for r in runs]
    f1s  = [r["macro_f1"] for r in runs]
    out = {
        "params":       params,
        "acc_mean":     round(float(np.mean(accs)), 6),
        "acc_std":      round(float(np.std(accs)), 6),
        "acc_min":      round(float(np.min(accs)), 6),
        "acc_max":      round(float(np.max(accs)), 6),
        "f1_mean":      round(float(np.mean(f1s)), 6),
        "f1_std":       round(float(np.std(f1s)), 6),
        "inference_ms": round(float(np.median([r["inference_ms"] for r in runs])), 4),
        "per_seed":     runs,
    }
    if extra:
        out.update(extra)
    return out


def resume_map(out_path: str) -> dict:
    """Load previously saved results (if any) for skip-completed resume."""
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return {}


def save(out_path: str, results: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
