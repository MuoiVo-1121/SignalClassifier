"""
compare.py
----------
Side-by-side comparison of 1D CNN vs LSTM on the test set.
Produces a printed summary table and saves comparison_bar.png.

Usage:
    python compare.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)

from models.cnn_model import build_cnn
from models.lstm_model import build_lstm

PROCESSED_DIR = "data/processed"
WEIGHTS_DIR   = "weights"
EXPORTS_DIR   = "exports"
CLASS_NAMES   = ["Dark", "UVC Lamp", "Flame"]


def load_test():
    X = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    return X, y


def run_inference(model, X, warmup: int = 5):
    _ = model.predict(X[:warmup], verbose=0)
    t0    = time.perf_counter()
    probs = model.predict(X, verbose=0)
    ms    = (time.perf_counter() - t0) / len(X) * 1000
    return np.argmax(probs, axis=1), ms


def compare():
    X_test, y_test = load_test()
    timesteps = X_test.shape[1]

    results = {}
    for label, builder, key in [("1D CNN", build_cnn, "cnn"), ("LSTM", build_lstm, "lstm")]:
        model = builder(timesteps)
        model.load_weights(os.path.join(WEIGHTS_DIR, f"{key}_best.h5"))
        y_pred, ms = run_inference(model, X_test)
        results[label] = {
            "accuracy":   accuracy_score(y_test, y_pred),
            "f1_macro":   f1_score(y_test, y_pred, average="macro"),
            "precision":  precision_score(y_test, y_pred, average="macro"),
            "recall":     recall_score(y_test, y_pred, average="macro"),
            "f1_per_cls": f1_score(y_test, y_pred, average=None),
            "params":     model.count_params(),
            "ms":         ms,
        }

    # --- printed table ---
    print("\n" + "=" * 62)
    print(f"  {'Metric':<22}  {'1D CNN':>12}  {'LSTM':>12}")
    print("=" * 62)
    for lbl, k in [("Accuracy", "accuracy"), ("F1 macro", "f1_macro"),
                   ("Precision", "precision"), ("Recall", "recall")]:
        c, l = results["1D CNN"][k], results["LSTM"][k]
        win  = "  <-- winner" if c >= l else ""
        print(f"  {lbl:<22}  {c:>11.4f}  {l:>11.4f}{win}")
    print("-" * 62)
    for i, cls in enumerate(CLASS_NAMES):
        c = results["1D CNN"]["f1_per_cls"][i]
        l = results["LSTM"]["f1_per_cls"][i]
        print(f"  F1 {cls:<18}  {c:>11.4f}  {l:>11.4f}")
    print("-" * 62)
    print(f"  {'Parameters':<22}  {results['1D CNN']['params']:>12,}  {results['LSTM']['params']:>12,}")
    print(f"  {'Inference ms/window':<22}  {results['1D CNN']['ms']:>11.3f}  {results['LSTM']['ms']:>11.3f}")
    print("=" * 62)

    # --- bar chart ---
    metric_keys  = ["accuracy", "f1_macro", "precision", "recall"]
    metric_lbls  = ["Accuracy", "F1 Macro", "Precision", "Recall"]
    cnn_scores   = [results["1D CNN"][k] for k in metric_keys]
    lstm_scores  = [results["LSTM"][k]   for k in metric_keys]

    x, w = range(len(metric_lbls)), 0.35
    _, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar([i - w / 2 for i in x], cnn_scores,  w, label="1D CNN", color="#4C72B0")
    b2 = ax.bar([i + w / 2 for i in x], lstm_scores, w, label="LSTM",   color="#DD8452")
    ax.set_ylim(0.80, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("1D CNN vs LSTM - Test Set Performance")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_lbls)
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    out_path = os.path.join(EXPORTS_DIR, "comparison_bar.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    compare()
