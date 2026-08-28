"""
visualize_training.py
---------------------
Load saved training histories and produce publication-quality learning-curve plots.

Prerequisites: train.py must have been run for both models so that
  exports/history_cnn.json  and  exports/history_lstm.json  exist.

Exports
-------
  exports/training_curves_cnn.png   — loss + accuracy curves for 1D CNN
  exports/training_curves_lstm.png  — loss + accuracy curves for LSTM
  exports/training_curves_combined.png — side-by-side comparison

Usage:
    python visualize_training.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

EXPORTS = "exports"
os.makedirs(EXPORTS, exist_ok=True)

MODEL_META = {
    "cnn":  {"label": "1D CNN",  "color": "#4C72B0"},
    "lstm": {"label": "LSTM",    "color": "#DD8452"},
}


# ── Helper ────────────────────────────────────────────────────────────────────
def load_history(model_key: str) -> dict:
    path = os.path.join(EXPORTS, f"history_{model_key}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"History not found at {path}. Run: python train.py --model {model_key}"
        )
    with open(path) as f:
        return json.load(f)


def smooth(values, window=3):
    """Simple moving-average smoother for display clarity."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ── Per-model curve ───────────────────────────────────────────────────────────
def plot_single(model_key: str, history: dict):
    meta   = MODEL_META[model_key]
    color  = meta["color"]
    label  = meta["label"]
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{label} — Training History", fontsize=13, fontweight="bold")

    # ── Loss ─────────────────────────────────────────────────────────────────
    ax_loss.plot(epochs, history["loss"],     color=color,   alpha=0.35, linewidth=1.2)
    ax_loss.plot(epochs, history["val_loss"], color="tomato", alpha=0.35, linewidth=1.2)
    ax_loss.plot(epochs, smooth(history["loss"]),     color=color,   linewidth=2.0, label="Train loss")
    ax_loss.plot(epochs, smooth(history["val_loss"]), color="tomato", linewidth=2.0,
                 linestyle="--", label="Val loss")
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Sparse Categorical Cross-Entropy", fontsize=9)
    ax_loss.set_title("Loss Curves", fontsize=11)
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax_acc.plot(epochs, [v * 100 for v in history["accuracy"]],     color=color,   alpha=0.35, linewidth=1.2)
    ax_acc.plot(epochs, [v * 100 for v in history["val_accuracy"]], color="tomato", alpha=0.35, linewidth=1.2)
    ax_acc.plot(epochs, smooth([v * 100 for v in history["accuracy"]]),
                color=color, linewidth=2.0, label="Train accuracy")
    ax_acc.plot(epochs, smooth([v * 100 for v in history["val_accuracy"]]),
                color="tomato", linewidth=2.0, linestyle="--", label="Val accuracy")

    best_val_acc = max(history["val_accuracy"]) * 100
    best_ep      = np.argmax(history["val_accuracy"]) + 1
    ax_acc.axhline(best_val_acc, color="green", linewidth=1.0, linestyle=":",
                   label=f"Best val acc: {best_val_acc:.1f}% (ep {best_ep})")
    ax_acc.set_xlabel("Epoch", fontsize=10)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=9)
    ax_acc.set_title("Accuracy Curves", fontsize=11)
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(EXPORTS, f"training_curves_{model_key}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Combined comparison ───────────────────────────────────────────────────────
def plot_combined(histories: dict):
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)
    fig.suptitle("1D CNN vs LSTM — Training History Comparison",
                 fontsize=13, fontweight="bold")

    ax_loss = fig.add_subplot(gs[0])
    ax_acc  = fig.add_subplot(gs[1])

    for key, history in histories.items():
        meta   = MODEL_META[key]
        color  = meta["color"]
        label  = meta["label"]
        epochs = range(1, len(history["loss"]) + 1)

        # Loss
        ax_loss.plot(epochs, smooth(history["val_loss"]), color=color,
                     linewidth=2.2, label=f"{label} val loss")
        ax_loss.plot(epochs, smooth(history["loss"]), color=color,
                     linewidth=1.2, linestyle="--", alpha=0.55,
                     label=f"{label} train loss")

        # Accuracy
        ax_acc.plot(epochs, smooth([v * 100 for v in history["val_accuracy"]]),
                    color=color, linewidth=2.2, label=f"{label} val acc")
        ax_acc.plot(epochs, smooth([v * 100 for v in history["accuracy"]]),
                    color=color, linewidth=1.2, linestyle="--", alpha=0.55,
                    label=f"{label} train acc")

    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Loss", fontsize=10)
    ax_loss.set_title("Validation Loss", fontsize=11)
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.set_xlabel("Epoch", fontsize=10)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10)
    ax_acc.set_title("Validation Accuracy", fontsize=11)
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(EXPORTS, "training_curves_combined.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating training curve visualizations…\n")
    histories = {}
    for key in ["cnn", "lstm"]:
        try:
            h = load_history(key)
            histories[key] = h
            plot_single(key, h)
            stopped = h.get("stopped_epoch", len(h["loss"]))
            best_acc = max(h["val_accuracy"]) * 100
            print(f"  {MODEL_META[key]['label']:8s} — "
                  f"epochs: {stopped}, best val acc: {best_acc:.2f}%")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    if len(histories) == 2:
        plot_combined(histories)

    print("\nDone.")
