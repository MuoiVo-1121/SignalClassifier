"""
baseline.py
-----------
Simple hand-crafted and classical ML baselines for photocurrent classification.
Answers reviewer question: "Why use deep learning when a simple statistic suffices?"

Three baselines:
  1. Threshold  — two handcrafted rules (window_mean, mean_abs_diff)
  2. LogReg     — Logistic Regression on 3 engineered features
  3. RandomForest — ensemble on the same 3 features

Features used (computed per window):
  - window_mean    : mean amplitude  → separates Dark from UV-active (Level 1)
  - mean_abs_diff  : mean |x[t]-x[t-1]|  → separates Lamp from Flame (Level 2)
  - std_diff       : std  |x[t]-x[t-1]|  → additional roughness descriptor

Usage:
    python baseline.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = "data/processed"
EXPORTS_DIR   = "exports"
CLASS_NAMES   = ["Dark", "Lamp", "Flame"]


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Extract 3 scalar features per window.
    X shape: (N, T, 1)  — already globally normalized
    Returns: (N, 3)
    """
    X2 = X[:, :, 0]                                   # (N, T)
    diff = np.abs(np.diff(X2, axis=1))                # (N, T-1)

    window_mean   = X2.mean(axis=1)                   # Level-1 feature
    mean_abs_diff = diff.mean(axis=1)                 # Level-2 feature (roughness)
    std_diff      = diff.std(axis=1)                  # Level-2 feature (roughness var)

    return np.stack([window_mean, mean_abs_diff, std_diff], axis=1)


# ── Baseline 1: Hand-crafted threshold ───────────────────────────────────────

def threshold_classifier(X: np.ndarray,
                         amp_thresh: float = 0.43,
                         diff_thresh: float = 0.045) -> np.ndarray:
    """
    Rule-based classifier using two thresholds:
      - amp_thresh  : separates Dark (below) from UV-active (above)
      - diff_thresh : separates Lamp (above) from Flame (below)

    Thresholds derived analytically from AR(1) signal statistics:
      Dark  normalized mean ≈ 0.23,  Lamp/Flame ≈ 0.62  → midpoint 0.43
      Lamp  mean|diff| ≈ 0.070,      Flame       ≈ 0.023 → midpoint 0.045
    """
    feats = extract_features(X)
    window_mean   = feats[:, 0]
    mean_abs_diff = feats[:, 1]

    y_pred = np.full(len(X), 2, dtype=np.int64)       # default: Flame
    y_pred[mean_abs_diff > diff_thresh] = 1            # high roughness → Lamp
    y_pred[window_mean   < amp_thresh]  = 0            # low amplitude  → Dark
    return y_pred


# ── Baseline 2 & 3: ML classifiers ───────────────────────────────────────────

def run_ml_baselines(X_train, y_train, X_test, y_test):
    feats_train = extract_features(X_train)
    feats_test  = extract_features(X_test)

    scaler = StandardScaler()
    feats_train_sc = scaler.fit_transform(feats_train)
    feats_test_sc  = scaler.transform(feats_test)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(feats_train_sc, y_train)
    results["LogReg"] = {
        "train_acc": accuracy_score(y_train, lr.predict(feats_train_sc)),
        "test_acc":  accuracy_score(y_test,  lr.predict(feats_test_sc)),
        "y_pred":    lr.predict(feats_test_sc),
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(feats_train, y_train)
    results["RandomForest"] = {
        "train_acc": accuracy_score(y_train, rf.predict(feats_train)),
        "test_acc":  accuracy_score(y_test,  rf.predict(feats_test)),
        "y_pred":    rf.predict(feats_test),
    }

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(name: str, y_test: np.ndarray, y_pred: np.ndarray,
                 train_acc: float = None):
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    if train_acc is not None:
        print(f"  Train accuracy : {train_acc*100:.2f}%")
    print(f"  Test  accuracy : {acc*100:.2f}%")
    print()
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    return acc


def save_confusion(name: str, y_test: np.ndarray, y_pred: np.ndarray):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    _, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    path = os.path.join(EXPORTS_DIR, f"confusion_matrix_{name.lower().replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Feature distribution analysis ─────────────────────────────────────────────

def show_feature_stats(X_train, y_train):
    feats = extract_features(X_train)
    print("\n── Feature statistics per class (training set) ──────────────────")
    print(f"  {'Class':<8} {'window_mean':>14} {'mean_abs_diff':>15} {'std_diff':>10}")
    print(f"  {'-'*50}")
    for cls, name in enumerate(CLASS_NAMES):
        mask = y_train == cls
        f    = feats[mask]
        print(f"  {name:<8} "
              f"  {f[:,0].mean():>8.4f}±{f[:,0].std():.4f}"
              f"  {f[:,1].mean():>8.4f}±{f[:,1].std():.4f}"
              f"  {f[:,2].mean():>6.4f}±{f[:,2].std():.4f}")
    lamp_diff  = feats[y_train == 1, 1].mean()
    flame_diff = feats[y_train == 2, 1].mean()
    print(f"\n  Lamp/Flame mean_abs_diff ratio: {lamp_diff/flame_diff:.2f}×")


# ── Summary comparison table ───────────────────────────────────────────────────

def summary_table(results: dict):
    print("\n" + "="*62)
    print(f"  {'Method':<20}  {'Train Acc':>10}  {'Test Acc':>10}")
    print("="*62)
    rows = [
        ("Threshold",     results["Threshold"]["train_acc"],     results["Threshold"]["test_acc"]),
        ("LogReg",        results["LogReg"]["train_acc"],        results["LogReg"]["test_acc"]),
        ("RandomForest",  results["RandomForest"]["train_acc"],  results["RandomForest"]["test_acc"]),
        ("1D CNN*",       None,                                  0.9925),
        ("LSTM*",         None,                                  0.7154),
    ]
    for name, tr, te in rows:
        tr_str = f"{tr*100:>9.2f}%" if tr is not None else "       N/A"
        print(f"  {name:<20}  {tr_str}  {te*100:>9.2f}%")
    print("="*62)
    print("  * CNN and LSTM numbers from saved weights (seed=42)")
    print("\n  Key insight: if Threshold/LogReg approach CNN accuracy,")
    print("  deep learning value lies in generalization and ease of")
    print("  deployment — not raw accuracy on this synthetic dataset.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nLoading data…")
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Feature statistics
    show_feature_stats(X_train, y_train)

    results = {}

    # ── Baseline 1: Threshold ─────────────────────────────────────────────────
    y_thresh = threshold_classifier(X_test)
    y_thresh_tr = threshold_classifier(X_train)
    results["Threshold"] = {
        "train_acc": accuracy_score(y_train, y_thresh_tr),
        "test_acc":  accuracy_score(y_test,  y_thresh),
        "y_pred":    y_thresh,
    }
    print_report("Threshold Classifier (0 params, no training)",
                 y_test, y_thresh, results["Threshold"]["train_acc"])
    save_confusion("threshold", y_test, y_thresh)

    # ── Baselines 2 & 3: ML classifiers ──────────────────────────────────────
    ml = run_ml_baselines(X_train, y_train, X_test, y_test)
    results.update(ml)

    print_report("Logistic Regression (3 features)",
                 y_test, ml["LogReg"]["y_pred"], ml["LogReg"]["train_acc"])
    save_confusion("logreg", y_test, ml["LogReg"]["y_pred"])

    print_report("Random Forest (3 features, 100 trees)",
                 y_test, ml["RandomForest"]["y_pred"], ml["RandomForest"]["train_acc"])
    save_confusion("random_forest", y_test, ml["RandomForest"]["y_pred"])

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_table(results)

    # ── Save results to JSON for paper update ─────────────────────────────────
    import json
    out = {
        name: {
            "train_acc": round(float(v["train_acc"]), 6) if v["train_acc"] else None,
            "test_acc":  round(float(v["test_acc"]),  6),
        }
        for name, v in results.items()
    }
    json_path = os.path.join(EXPORTS_DIR, "baseline_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to: {json_path}")


if __name__ == "__main__":
    main()
