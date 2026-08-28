"""
generate_dummy_data.py
----------------------
Generate physics-inspired photocurrent recordings for three source classes.

Two-level discrimination challenge
------------------------------------
Level 1 — DC Amplitude  (easy, separates Dark from Lamp/Flame):
  Dark  : mean ≈ 0 nA,  std ≈ 0.55 nA,  AR ρ ≈ 0.25
  Lamp  : mean ≈ 4 nA,  std ≈ 0.65 nA,  AR ρ ≈ 0.06  (stable, nearly white)
  Flame : mean ≈ 4 nA,  std ≈ 0.65 nA,  AR ρ ≈ 0.90  (strongly correlated)

Level 2 — Temporal Autocorrelation  (hard, separates Lamp from Flame):
  Lamp  : nearly i.i.d. noise — each sample is almost independent of previous
  Flame : strongly AR(1) correlated — each sample carries long-range memory
          correlation length ≈ -1/ln(0.90) ≈ 9.5 samples ≈ 0.095 seconds

  Because Lamp and Flame have IDENTICAL means AND IDENTICAL variances, any
  classifier that relies only on amplitude statistics (mean, std) will fail.
  Temporal models (CNN via learned filters, LSTM via sequential state) must
  detect the correlation structure to discriminate Lamp from Flame.

Physical motivation
-------------------
  - UVC lamp photocurrent: driven by stable power supply → each photon
    absorption event is statistically independent (shot noise → white noise).
  - Methanol flame photocurrent: driven by turbulent combustion dynamics →
    turbulence has coherence times of 0.05–0.5 s, producing strongly
    correlated fluctuations at the 100 Hz sampling rate.

Expected model differentiation
-------------------------------
  - LSTM naturally detects AR correlation through cell state → strong on Flame
  - CNN Conv1D(k=5) detects 5-step lag patterns → weaker correlation detector
  → LSTM expected to outperform CNN for Lamp/Flame discrimination
  → CNN compensates with 3× fewer parameters and 2.8× faster inference

Usage:
    python generate_dummy_data.py
"""

import os
import numpy as np
import pandas as pd

RAW_DIR     = "data/raw"
SAMPLE_RATE = 100
DURATION_S  = 300
N_SAMPLES   = SAMPLE_RATE * DURATION_S   # 30,000
SEED        = 42

np.random.seed(SEED)
os.makedirs(RAW_DIR, exist_ok=True)

time_axis = np.linspace(0, DURATION_S, N_SAMPLES, endpoint=False)


def ar1_process(n: int, mean: float, sigma: float, rho: float) -> np.ndarray:
    """Stationary AR(1): x[t] = mean + rho*(x[t-1]-mean) + eps[t].
    sigma is the STATIONARY std (not the innovation std).
    """
    eps_std = sigma * np.sqrt(max(1e-9, 1.0 - rho ** 2))
    x = np.empty(n)
    x[0] = np.random.normal(mean, sigma)
    for i in range(1, n):
        x[i] = mean + rho * (x[i - 1] - mean) + np.random.normal(0.0, eps_std)
    return x


# ── Class 0: Dark ─────────────────────────────────────────────────────────────
# Weakly correlated electronic noise + slow thermal drift.
dark_noise  = ar1_process(N_SAMPLES, mean=0.0, sigma=0.55, rho=0.25)
dark_drift  = 0.15 * np.sin(2 * np.pi * 0.006 * time_axis)
dark_signal = dark_noise + dark_drift

# ── Class 1: UVC Lamp ─────────────────────────────────────────────────────────
# Stable 4 nA DC with:
#   - Very low AR correlation (ρ=0.06) → nearly independent samples (shot noise)
#   - Stationary std = 0.65 nA (same as Flame)
#   - 5 Hz ripple (0.08 nA) from power supply
# NOTE: warmup transient removed — the ramp-up from 0→4 nA created the first
# ~9 training windows with near-zero amplitude (labeled lamp=1 but looking like
# dark=0). After noise augmentation those copies fell squarely in the dark range,
# poisoning the training set with mislabeled examples.
lamp_noise  = ar1_process(N_SAMPLES, mean=4.00, sigma=0.65, rho=0.06)
lamp_ripple = 0.08 * np.sin(2 * np.pi * 5.0 * time_axis)
lamp_signal = lamp_noise + lamp_ripple

# ── Class 2: Methanol Flame ───────────────────────────────────────────────────
# SAME mean (4 nA) AND SAME stationary std (0.65 nA) as Lamp,
# but STRONGLY correlated: AR ρ=0.90 → correlation length ≈ 9.5 samples.
#
# Within any 100-sample window, flame samples are heavily dependent on each
# other (slow oscillations / drifts), while lamp samples are nearly i.i.d.
# This temporal correlation is the ONLY discriminative feature between Lamp
# and Flame — requiring the model to detect sequential dependence structure.
#
# Small sparse spikes (physically: occasional turbulent puffs).
flame_base = ar1_process(N_SAMPLES, mean=4.00, sigma=0.65, rho=0.90)

rng = np.random.default_rng(SEED + 13)
# Sparse individual spikes (not grouped — groups would reveal themselves via
# local variance, which would make the task amplitude-based again)
spike_locs = rng.choice(np.arange(500, N_SAMPLES - 500), size=25, replace=False)
for loc in spike_locs:
    amp = rng.uniform(1.5, 3.5)
    flame_base[loc] += amp
    if loc + 1 < N_SAMPLES:
        flame_base[loc + 1] += amp * 0.4  # short tail

flame_signal = flame_base

# ── Print summary ─────────────────────────────────────────────────────────────
for label, signal, name, rho in [
    (0, dark_signal,  "dark",  0.25),
    (1, lamp_signal,  "lamp",  0.06),
    (2, flame_signal, "flame", 0.90),
]:
    df   = pd.DataFrame({"time_s": time_axis, "current_nA": signal, "label": label})
    path = os.path.join(RAW_DIR, f"{name}_recording.csv")
    df.to_csv(path, index=False)
    print(f"  {name:5s} | mean={signal.mean():+.3f}  std={signal.std():.3f}  "
          f"[{signal.min():.2f}, {signal.max():.2f}]  AR ρ≈{rho}  class={label}")

print()
print("Key design properties:")
print("  Lamp mean ≈ Flame mean ≈ 4 nA  → amplitude alone CANNOT separate them")
print("  Lamp std  ≈ Flame std  ≈ 0.65 nA → variance alone CANNOT separate them")
print("  AR ρ(Lamp)=0.06 vs AR ρ(Flame)=0.90 → ONLY temporal structure can")
print("  → LSTM should outperform CNN on Lamp/Flame discrimination")
print(f"\nSaved 3 recordings to '{RAW_DIR}/'  ({N_SAMPLES:,} samples each)")
print("Next step:  python preprocess.py")
