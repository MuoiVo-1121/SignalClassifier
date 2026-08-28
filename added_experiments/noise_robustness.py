"""
noise_robustness.py
-------------------
Experiment 3: test-time noise robustness of the proposed 1D-CNN.

Loads the five per-seed proposed-CNN models trained in capacity_compare.py
(weights_added/proposed_seed*/best.weights.h5) and evaluates each on the test
set with additive Gaussian noise of increasing sigma (normalized units; the
data range is ~10.22 nA, so sigma_nA = sigma * range). Noise is applied after
normalization and clipped to [0, 1], mirroring the training augmentation
(which used sigma = 0.05).

No retraining — evaluation only. Run this BEFORE matched_compare.py touches
data/processed, or at least not concurrently (it reads X_test from there).

Usage:
    python noise_robustness.py
"""

import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score

from models.cnn_model import build_cnn
from common_runner import SEEDS, EXPORTS_DIR, WEIGHTS_DIR

PROCESSED_DIR = "data/processed"
SIGMAS        = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
CLASS_NAMES   = ["dark", "lamp", "flame"]
OUT_PATH      = os.path.join(EXPORTS_DIR, "noise_results.json")


def main():
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    gmin, gmax = np.load(os.path.join(PROCESSED_DIR, "global_stats.npy"))
    rng_range = float(gmax - gmin)
    print(f"Test set: {X_test.shape}, data range = {rng_range:.3f} nA")

    results = {"data_range_nA": round(rng_range, 4),
               "sigmas_normalized": SIGMAS,
               "sigmas_nA": [round(s * rng_range, 3) for s in SIGMAS],
               "per_seed": {}}

    for seed in SEEDS:
        wpath = os.path.join(WEIGHTS_DIR, f"proposed_seed{seed}",
                             "best.weights.h5")
        model = build_cnn(timesteps=X_test.shape[1])
        model.load_weights(wpath)

        rows = []
        for i, sigma in enumerate(SIGMAS):
            rng = np.random.default_rng(1000 * seed + i)   # reproducible
            if sigma == 0.0:
                Xn = X_test
            else:
                noise = rng.normal(0.0, sigma, size=X_test.shape).astype(np.float32)
                Xn = np.clip(X_test + noise, 0.0, 1.0)
            y_pred = np.argmax(model.predict(Xn, verbose=0), axis=1)
            rec = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2],
                               zero_division=0)
            rows.append({
                "sigma":     sigma,
                "sigma_nA":  round(sigma * rng_range, 3),
                "accuracy":  round(float(accuracy_score(y_test, y_pred)), 6),
                "macro_f1":  round(float(f1_score(y_test, y_pred,
                                                  average="macro")), 6),
                "recall":    {c: round(float(r), 4)
                              for c, r in zip(CLASS_NAMES, rec)},
            })
            print(f"  seed={seed}  sigma={sigma:<5} ({sigma*rng_range:5.2f} nA)"
                  f"  acc={rows[-1]['accuracy']*100:6.2f}%  "
                  f"recall d/l/f = {rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f}",
                  flush=True)
        results["per_seed"][str(seed)] = rows

    # Aggregate mean/std per sigma
    agg = []
    for i, sigma in enumerate(SIGMAS):
        accs = [results["per_seed"][str(s)][i]["accuracy"] for s in SEEDS]
        agg.append({"sigma": sigma,
                    "sigma_nA": round(sigma * rng_range, 3),
                    "acc_mean": round(float(np.mean(accs)), 6),
                    "acc_std":  round(float(np.std(accs)), 6),
                    "acc_min":  round(float(np.min(accs)), 6)})
    results["aggregate"] = agg

    os.makedirs(EXPORTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*64}")
    print(f"  {'sigma':>6} {'sigma(nA)':>10} {'acc mean':>10} {'std':>8} {'min':>8}")
    print(f"{'='*64}")
    for a in agg:
        print(f"  {a['sigma']:>6} {a['sigma_nA']:>10} "
              f"{a['acc_mean']*100:>9.2f}% {a['acc_std']*100:>7.2f}pp "
              f"{a['acc_min']*100:>7.2f}%")
    print(f"{'='*64}")
    print(f"\nResults saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
