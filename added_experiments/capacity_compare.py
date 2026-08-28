"""
capacity_compare.py
-------------------
Capacity ablation: Tiny-CNN vs Proposed 1D-CNN vs Deep-CNN.

Faithfully replicates the original multiseed protocol
(signal_classifier/multiseed_train.py + train.py):
  - per seed: re-run preprocess with that seed (augmentation noise + shuffle;
    raw traces and the temporal test split are fixed), then train
  - training seed fixed at 42 inside the training routine (as in train.py)
  - Adam lr=1e-3, batch 64, up to 80 epochs
  - EarlyStopping(val_accuracy, patience 20, restore best)
  - ReduceLROnPlateau(val_loss, factor 0.5, patience 8, min_lr 1e-6)
  - seeds {42, 123, 7, 2024, 99}

Reports per model: test accuracy / macro-F1 (mean +/- std over seeds),
parameter count, inference ms/window (compare.py timing method).

Usage:
    python capacity_compare.py
    python capacity_compare.py --models tiny proposed deep --seeds 42 123
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import preprocess as pre_mod
from models.cnn_model import build_cnn
from models.capacity_variants import build_tiny_cnn, build_deep_cnn

PROCESSED_DIR = "data/processed"
EXPORTS_DIR   = "exports_added"
WEIGHTS_DIR   = "weights_added"
SEEDS         = [42, 123, 7, 2024, 99]
TRAIN_SEED    = 42          # fixed inside train.py in the original protocol
EPOCHS        = 80
BATCH_SIZE    = 64
LR            = 1e-3

BUILDERS = {
    "tiny":     build_tiny_cnn,
    "proposed": build_cnn,
    "deep":     build_deep_cnn,
}


def load_splits():
    def load(name):
        return np.load(os.path.join(PROCESSED_DIR, name))
    return (load("X_train.npy"), load("X_val.npy"), load("X_test.npy"),
            load("y_train.npy"), load("y_val.npy"), load("y_test.npy"))


def run_inference(model, X, warmup: int = 5):
    """Same timing method as compare.py: batch predict / N windows."""
    _ = model.predict(X[:warmup], verbose=0)
    t0    = time.perf_counter()
    probs = model.predict(X, verbose=0)
    ms    = (time.perf_counter() - t0) / len(X) * 1000
    return np.argmax(probs, axis=1), ms


def run_one(model_key: str, seed: int) -> dict:
    print(f"\n{'-'*55}\n  model={model_key}  data_seed={seed}\n{'-'*55}", flush=True)

    # Re-preprocess with this seed (multiseed_train.py protocol)
    orig_seed = pre_mod.RANDOM_SEED
    pre_mod.RANDOM_SEED = seed
    pre_mod.preprocess(raw_dir="data/raw", processed_dir=PROCESSED_DIR)
    pre_mod.RANDOM_SEED = orig_seed

    tf.random.set_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    model = BUILDERS[model_key](timesteps=X_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=20,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=8, min_lr=1e-6, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )

    os.makedirs(os.path.join(WEIGHTS_DIR, f"{model_key}_seed{seed}"), exist_ok=True)
    model.save_weights(os.path.join(WEIGHTS_DIR, f"{model_key}_seed{seed}",
                                    "best.weights.h5"))

    y_pred, ms = run_inference(model, X_test)
    result = {
        "seed":          seed,
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "macro_f1":      round(float(f1_score(y_test, y_pred, average="macro")), 6),
        "best_val_acc":  round(float(max(history.history["val_accuracy"])), 6),
        "stopped_epoch": int(len(history.history["loss"])),
        "inference_ms":  round(float(ms), 4),
        "n_errors":      int((y_pred != y_test).sum()),
    }
    print(f"  -> test_acc={result['test_accuracy']*100:.2f}%  "
          f"macro_f1={result['macro_f1']:.4f}  "
          f"epochs={result['stopped_epoch']}  ms/window={ms:.3f}", flush=True)
    return result


def summarize(model_key: str, params: int, runs: list) -> dict:
    accs = [r["test_accuracy"] for r in runs]
    f1s  = [r["macro_f1"] for r in runs]
    return {
        "params":        params,
        "acc_mean":      round(float(np.mean(accs)), 6),
        "acc_std":       round(float(np.std(accs)), 6),
        "acc_min":       round(float(np.min(accs)), 6),
        "acc_max":       round(float(np.max(accs)), 6),
        "f1_mean":       round(float(np.mean(f1s)), 6),
        "f1_std":        round(float(np.std(f1s)), 6),
        "inference_ms":  round(float(np.median([r["inference_ms"] for r in runs])), 4),
        "per_seed":      runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(BUILDERS),
                        choices=list(BUILDERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    os.makedirs(EXPORTS_DIR, exist_ok=True)
    out_path = os.path.join(EXPORTS_DIR, "capacity_results.json")

    # Resume support: reload finished (model, seed) runs and skip them,
    # so the script can be safely re-invoked until all runs are complete.
    all_results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)

    for model_key in args.models:
        params = BUILDERS[model_key](timesteps=100).count_params()
        print(f"\n{'='*55}\n  {model_key.upper()}  ({params:,} params)\n{'='*55}",
              flush=True)
        done = {r["seed"]: r
                for r in all_results.get(model_key, {}).get("per_seed", [])}
        runs = []
        for seed in args.seeds:
            if seed in done:
                print(f"  [skip] seed={seed} already done "
                      f"(test_acc={done[seed]['test_accuracy']*100:.2f}%)",
                      flush=True)
                runs.append(done[seed])
                continue
            runs.append(run_one(model_key, seed))
            all_results[model_key] = summarize(model_key, params, runs)
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
        all_results[model_key] = summarize(model_key, params, runs)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  {'Model':<10} {'Params':>8} {'Test acc (mean+/-std)':>24} "
          f"{'Range':>18} {'ms/win':>7}")
    print(f"{'='*72}")
    for key, s in all_results.items():
        print(f"  {key:<10} {s['params']:>8,} "
              f"{s['acc_mean']*100:>10.2f}% +/- {s['acc_std']*100:.2f}pp "
              f"  [{s['acc_min']*100:.2f}, {s['acc_max']*100:.2f}]% "
              f"{s['inference_ms']:>7.3f}")
    print(f"{'='*72}")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
