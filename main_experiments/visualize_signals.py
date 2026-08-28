"""
visualize_signals.py
--------------------
Generate publication-quality figures of the raw photocurrent signals
and their statistical distributions.

Exports
-------
  exports/signals_raw.png        — 5-second time-domain waveform per class
  exports/signals_distribution.png — amplitude histogram + KDE per class
  exports/signals_window_examples.png — sample windows (one per class)

Usage:
    python visualize_signals.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

RAW_DIR    = "data/raw"
PROC_DIR   = "data/processed"
EXPORTS    = "exports"
os.makedirs(EXPORTS, exist_ok=True)

CLASS_NAMES  = ["Dark (Class 0)", "UVC Lamp (Class 1)", "Methanol Flame (Class 2)"]
CLASS_COLORS = ["#2196F3", "#FF9800", "#F44336"]
CLASS_FILES  = ["dark_recording.csv", "lamp_recording.csv", "flame_recording.csv"]

SAMPLE_RATE  = 100   # Hz
DISPLAY_S    = 10    # seconds to show in time-domain plot


# ── Helper ────────────────────────────────────────────────────────────────────
def load_signals():
    signals = []
    for fname in CLASS_FILES:
        df = pd.read_csv(os.path.join(RAW_DIR, fname))
        signals.append(df["current_nA"].values)
    return signals


# ── Figure 1: Time-domain waveforms ─────────────────────────────────────────
def plot_raw_signals(signals):
    n_show = DISPLAY_S * SAMPLE_RATE
    time   = np.linspace(0, DISPLAY_S, n_show)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Raw Photocurrent Signals — Time Domain (first 10 s)",
                 fontsize=13, fontweight="bold", y=0.98)

    for ax, sig, name, color in zip(axes, signals, CLASS_NAMES, CLASS_COLORS):
        seg = sig[:n_show]
        ax.plot(time, seg, color=color, linewidth=0.7, alpha=0.85)
        ax.axhline(seg.mean(), color="black", linewidth=1.2,
                   linestyle="--", alpha=0.6, label=f"mean = {seg.mean():.2f} nA")
        ax.fill_between(time,
                        seg.mean() - seg.std(),
                        seg.mean() + seg.std(),
                        color=color, alpha=0.15, label=f"±1σ ({seg.std():.2f} nA)")
        ax.set_ylabel("Current (nA)", fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold", color=color)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, DISPLAY_S)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()
    out = os.path.join(EXPORTS, "signals_raw.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 2: Amplitude distributions ───────────────────────────────────────
def plot_distributions(signals):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Photocurrent Amplitude Distributions per Class",
                 fontsize=13, fontweight="bold")

    for ax, sig, name, color in zip(axes, signals, CLASS_NAMES, CLASS_COLORS):
        # Histogram
        ax.hist(sig, bins=80, color=color, alpha=0.45, density=True,
                label="Histogram")
        # KDE
        kde_x = np.linspace(sig.min(), sig.max(), 400)
        kde   = gaussian_kde(sig, bw_method="silverman")
        ax.plot(kde_x, kde(kde_x), color=color, linewidth=2.0, label="KDE")
        # Stats annotation
        stats = (f"μ = {sig.mean():.2f} nA\n"
                 f"σ = {sig.std():.2f} nA\n"
                 f"min = {sig.min():.2f}\n"
                 f"max = {sig.max():.2f}")
        ax.text(0.97, 0.97, stats, transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.set_title(name, fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel("Current (nA)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(EXPORTS, "signals_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 3: Sample windows per class ──────────────────────────────────────
def plot_window_examples():
    try:
        X_test = np.load(os.path.join(PROC_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(PROC_DIR, "y_test.npy"))
    except FileNotFoundError:
        print("  [skip] Processed data not found — run preprocess.py first.")
        return

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.suptitle("Representative 1-Second Windows (Normalized, post-preprocessing)",
                 fontsize=12, fontweight="bold")

    t_axis = np.linspace(0, 1.0, 100)
    for cls, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        idx = np.where(y_test == cls)[0]
        ax  = fig.add_subplot(gs[cls])
        # Plot up to 5 example windows (light) + mean window (bold)
        examples = X_test[idx[:5], :, 0]
        for ex in examples:
            ax.plot(t_axis, ex, color=color, linewidth=0.6, alpha=0.4)
        mean_win = examples.mean(axis=0)
        ax.plot(t_axis, mean_win, color=color, linewidth=2.0, label="Mean window")
        ax.set_title(name, fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel("Time within window (s)", fontsize=9)
        ax.set_ylabel("Normalized amplitude", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(EXPORTS, "signals_window_examples.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 4: Class overlap illustration ────────────────────────────────────
def plot_class_overlap(signals):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Amplitude Distribution Overlap — Classification Challenge",
                 fontsize=12, fontweight="bold")

    all_min = min(s.min() for s in signals)
    all_max = max(s.max() for s in signals)
    kde_x   = np.linspace(all_min, all_max, 600)

    for sig, name, color in zip(signals, CLASS_NAMES, CLASS_COLORS):
        kde = gaussian_kde(sig, bw_method="silverman")
        y   = kde(kde_x)
        ax.plot(kde_x, y, color=color, linewidth=2.5, label=name)
        ax.fill_between(kde_x, 0, y, color=color, alpha=0.15)

    # Annotate overlap regions
    ax.axvspan(1.5, 3.0, alpha=0.08, color="gray",
               label="Dark–Lamp overlap region")
    ax.axvspan(3.0, 6.5, alpha=0.08, color="purple",
               label="Lamp–Flame overlap region")

    ax.set_xlabel("Photocurrent (nA)", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(EXPORTS, "signals_overlap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating signal visualizations…\n")
    signals = load_signals()
    plot_raw_signals(signals)
    plot_distributions(signals)
    plot_window_examples()
    plot_class_overlap(signals)
    print("\nDone.")
