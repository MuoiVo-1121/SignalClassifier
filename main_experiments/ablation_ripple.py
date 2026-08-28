"""
ablation_ripple.py
------------------
Ablation study: does CNN rely on the 5 Hz power-supply ripple in the Lamp
signal, or does it genuinely learn short-range temporal roughness (AR ρ)?

Method:
  1. Regenerate Lamp signal WITHOUT the 0.08·sin(2π·5·t) ripple component.
  2. Rebuild the full dataset (preprocess) using the no-ripple Lamp recording.
  3. Retrain CNN from scratch on the no-ripple dataset.
  4. Compare test accuracy: with-ripple (original) vs without-ripple.

If accuracy stays near 99% → CNN learns AR roughness, not the ripple.
If accuracy drops significantly → ripple was a discriminative cue (confound).

Usage:
    python ablation_ripple.py
"""

import os, json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

# ── Import project modules ────────────────────────────────────────────────────
from preprocess import preprocess
from train import train

RAW_DIR          = "data/raw"
ABLATION_RAW_DIR = "data/raw_no_ripple"
ABLATION_PRO_DIR = "data/processed_no_ripple"
ABLATION_WGT_DIR = "weights/ablation_no_ripple"
EXPORTS_DIR      = "exports"
SAMPLE_RATE      = 100
DURATION_S       = 300
N_SAMPLES        = SAMPLE_RATE * DURATION_S
SEED             = 42


def ar1_process(n, mean, sigma, rho):
    """Stationary AR(1): x[t] = mean + rho*(x[t-1]-mean) + eps[t]."""
    eps_std = sigma * np.sqrt(max(1e-9, 1.0 - rho ** 2))
    x = np.empty(n)
    np.random.seed(SEED)
    x[0] = np.random.normal(mean, sigma)
    for i in range(1, n):
        x[i] = mean + rho * (x[i - 1] - mean) + np.random.normal(0.0, eps_std)
    return x


def generate_no_ripple_lamp(out_dir: str):
    """Generate Lamp signal identical to original but with ripple = 0."""
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(SEED)
    time_axis  = np.linspace(0, DURATION_S, N_SAMPLES, endpoint=False)

    # Copy Dark and Flame recordings unchanged
    for name in ["dark_recording.csv", "flame_recording.csv"]:
        src = os.path.join(RAW_DIR, name)
        dst = os.path.join(out_dir, name)
        df  = pd.read_csv(src)
        df.to_csv(dst, index=False)

    # Lamp: same AR(1) but NO ripple
    lamp_noise  = ar1_process(N_SAMPLES, mean=4.00, sigma=0.65, rho=0.06)
    lamp_signal = lamp_noise          # ripple removed

    df_lamp = pd.DataFrame({
        "time_s":     time_axis,
        "current_nA": lamp_signal,
        "label":      1,
    })
    df_lamp.to_csv(os.path.join(out_dir, "lamp_recording.csv"), index=False)

    orig_std  = pd.read_csv(os.path.join(RAW_DIR, "lamp_recording.csv"))["current_nA"].std()
    norl_std  = lamp_signal.std()
    print(f"  Lamp std  — original: {orig_std:.4f} nA  |  no-ripple: {norl_std:.4f} nA")
    print(f"  Ripple amplitude: 0.08 nA (removed)")
    print(f"  Files saved to: {out_dir}/")


def run_ablation():
    print("\n" + "="*60)
    print("  ABLATION: CNN accuracy with vs without 5 Hz lamp ripple")
    print("="*60)

    # ── Step 1: Generate no-ripple dataset ───────────────────────────────────
    print("\n[1] Generating no-ripple Lamp signal…")
    generate_no_ripple_lamp(ABLATION_RAW_DIR)

    # ── Step 2: Preprocess no-ripple dataset ─────────────────────────────────
    print("\n[2] Preprocessing no-ripple dataset…")
    preprocess(raw_dir=ABLATION_RAW_DIR, processed_dir=ABLATION_PRO_DIR)

    # ── Step 3: Train CNN on no-ripple data ──────────────────────────────────
    print("\n[3] Training CNN on no-ripple dataset…")
    os.makedirs(ABLATION_WGT_DIR, exist_ok=True)

    # Patch WEIGHTS_DIR and EXPORTS_DIR inside train module for this run
    import train as train_module
    orig_weights = train_module.WEIGHTS_DIR
    orig_exports = train_module.EXPORTS_DIR
    train_module.WEIGHTS_DIR = ABLATION_WGT_DIR
    train_module.EXPORTS_DIR = os.path.join(EXPORTS_DIR, "ablation_no_ripple")
    os.makedirs(train_module.EXPORTS_DIR, exist_ok=True)

    _, history = train(model_name="cnn", epochs=80, batch_size=64,
                       processed_dir=ABLATION_PRO_DIR)

    train_module.WEIGHTS_DIR = orig_weights
    train_module.EXPORTS_DIR = orig_exports

    # ── Step 4: Evaluate on no-ripple test set ────────────────────────────────
    print("\n[4] Evaluating no-ripple CNN on test set…")
    from models.cnn_model import build_cnn

    X_test_nr = np.load(os.path.join(ABLATION_PRO_DIR, "X_test.npy"))
    y_test_nr = np.load(os.path.join(ABLATION_PRO_DIR, "y_test.npy"))

    cnn_nr = build_cnn(timesteps=X_test_nr.shape[1])
    cnn_nr.load_weights(os.path.join(ABLATION_WGT_DIR, "cnn_best.h5"))
    y_pred_nr = np.argmax(cnn_nr.predict(X_test_nr, verbose=0), axis=1)
    acc_nr    = accuracy_score(y_test_nr, y_pred_nr)

    # ── Step 5: Compare ───────────────────────────────────────────────────────
    acc_orig = 0.9925   # from saved weights (seed=42)

    print("\n" + "="*60)
    print("  ABLATION RESULTS")
    print("="*60)
    print(f"  CNN with    5 Hz ripple (original) : {acc_orig*100:.2f}%")
    print(f"  CNN without 5 Hz ripple (ablation) : {acc_nr*100:.2f}%")
    delta = (acc_nr - acc_orig) * 100
    print(f"  Delta                              : {delta:+.2f} pp")
    print()
    if abs(delta) < 2.0:
        print("  CONCLUSION: accuracy unchanged → CNN does NOT rely on 5 Hz ripple.")
        print("  CNN learns AR(1) roughness (temporal autocorrelation), not the ripple.")
        conclusion = "ripple_not_used"
    else:
        print("  CONCLUSION: accuracy changed significantly → ripple IS a confound.")
        conclusion = "ripple_is_confound"
    print("="*60)

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        "acc_with_ripple":    round(float(acc_orig), 6),
        "acc_without_ripple": round(float(acc_nr),   6),
        "delta_pp":           round(float(delta),     4),
        "conclusion":         conclusion,
    }
    out_path = os.path.join(EXPORTS_DIR, "ablation_ripple_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    return results


if __name__ == "__main__":
    run_ablation()
