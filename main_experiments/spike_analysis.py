"""
spike_analysis.py
-----------------
Checks whether CNN classification of Flame windows relies on the 25 sparse
spike events injected into the Flame signal, or genuinely on AR(1) roughness.

Method:
  - For each Flame test window, detect spike presence:
      spike present if max(window) > global_flame_mean + 3 * global_flame_std
  - Split Flame test windows into spike_windows and no_spike_windows
  - Compute CNN accuracy for each group separately

If both groups have high accuracy → CNN does not rely on spikes.
If no_spike accuracy is much lower → spikes are a discriminative cue.

Usage:
    python spike_analysis.py
"""

import os, json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from models.cnn_model import build_cnn

PROCESSED_DIR = "data/processed"
WEIGHTS_PATH  = "weights/cnn_best.h5"
EXPORTS_DIR   = "exports"
CLASS_NAMES   = ["Dark", "Lamp", "Flame"]
FLAME_CLASS   = 2


def detect_spikes(X_flame_norm: np.ndarray,
                  global_min: float, global_max: float,
                  z_thresh: float = 3.0) -> np.ndarray:
    """
    Return boolean mask: True where a window contains a spike.
    Spike defined as: any sample > mean + z_thresh * std of all Flame values.
    Works in normalized space.
    """
    X2 = X_flame_norm[:, :, 0]           # (N, T)
    flame_mean = X2.mean()
    flame_std  = X2.std()
    spike_mask = (X2.max(axis=1) > flame_mean + z_thresh * flame_std)
    return spike_mask


def main():
    print("\n" + "="*60)
    print("  SPIKE CONFOUND ANALYSIS")
    print("="*60)

    # ── Load data and model ───────────────────────────────────────────────────
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    gs      = np.load(os.path.join(PROCESSED_DIR, "global_stats.npy"))
    global_min, global_max = float(gs[0]), float(gs[1])

    cnn = build_cnn(timesteps=X_test.shape[1])
    cnn.load_weights(WEIGHTS_PATH)
    y_pred_all = np.argmax(cnn.predict(X_test, verbose=0), axis=1)

    # ── Isolate Flame test windows ────────────────────────────────────────────
    flame_mask   = (y_test == FLAME_CLASS)
    X_flame      = X_test[flame_mask]
    y_flame_true = y_test[flame_mask]
    y_flame_pred = y_pred_all[flame_mask]

    print(f"\n  Total Flame test windows: {len(X_flame)}")

    # ── Spike detection ───────────────────────────────────────────────────────
    spike_mask    = detect_spikes(X_flame, global_min, global_max, z_thresh=3.0)
    n_spike       = spike_mask.sum()
    n_no_spike    = (~spike_mask).sum()

    print(f"  Windows WITH spike    : {n_spike}  ({n_spike/len(X_flame)*100:.1f}%)")
    print(f"  Windows WITHOUT spike : {n_no_spike}  ({n_no_spike/len(X_flame)*100:.1f}%)")

    # ── Accuracy per group ────────────────────────────────────────────────────
    results = {}

    if n_spike > 0:
        acc_spike = accuracy_score(y_flame_true[spike_mask],
                                   y_flame_pred[spike_mask])
        results["spike_windows"] = {
            "n": int(n_spike),
            "accuracy": round(float(acc_spike), 6),
        }
        print(f"\n  CNN accuracy on spike windows    : {acc_spike*100:.2f}%")
    else:
        print("\n  No spike windows found in test set.")
        results["spike_windows"] = {"n": 0, "accuracy": None}

    acc_no_spike = accuracy_score(y_flame_true[~spike_mask],
                                  y_flame_pred[~spike_mask])
    results["no_spike_windows"] = {
        "n": int(n_no_spike),
        "accuracy": round(float(acc_no_spike), 6),
    }
    print(f"  CNN accuracy on no-spike windows : {acc_no_spike*100:.2f}%")

    # Overall Flame accuracy
    acc_all = accuracy_score(y_flame_true, y_flame_pred)
    results["all_flame_windows"] = {
        "n": int(len(X_flame)),
        "accuracy": round(float(acc_all), 6),
    }
    print(f"  CNN accuracy on ALL Flame windows: {acc_all*100:.2f}%")

    # ── Per-window amplitude stats for context ────────────────────────────────
    X2         = X_flame[:, :, 0]
    win_max    = X2.max(axis=1)
    print(f"\n  Flame window max amplitude (normalized):")
    print(f"    mean={win_max.mean():.4f}  std={win_max.std():.4f}  "
          f"min={win_max.min():.4f}  max={win_max.max():.4f}")

    # ── Conclusion ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    if n_spike == 0:
        conclusion = "no_spikes_in_test"
        print("  CONCLUSION: No spike windows in test set.")
        print("  CNN accuracy is entirely from non-spike Flame windows.")
        print("  → CNN learns smooth AR(1) structure, not spikes.")
    elif acc_no_spike >= 0.90:
        conclusion = "spikes_not_needed"
        print(f"  CONCLUSION: CNN achieves {acc_no_spike*100:.1f}% on non-spike Flame.")
        print("  → CNN does NOT rely on spikes; it detects smooth AR(1) texture.")
    else:
        conclusion = "spikes_may_help"
        print(f"  CONCLUSION: CNN drops to {acc_no_spike*100:.1f}% without spikes.")
        print("  → Spikes may be contributing as a discriminative cue.")
    print("="*60)

    results["conclusion"] = conclusion

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(EXPORTS_DIR, "spike_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    return results


if __name__ == "__main__":
    main()
