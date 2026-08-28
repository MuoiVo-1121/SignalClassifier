"""
visualize_noise.py
------------------
Figure for the test-time noise robustness sweep (proposed 1D-CNN).

Panel (a): test accuracy vs noise sigma (nA) — mean line + std band over the
           5 per-seed models, individual seeds as faint lines.
Panel (b): mean per-class recall vs sigma — shows the failure mode
           (flame recall collapses first: added white noise masks the
           smooth AR(1) texture, so flame windows look lamp-like).

Output: exports_added/noise_robustness.png (300 dpi)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

EXPORTS_DIR = "exports_added"

ACCENT = "#2563EB"
INK    = "#374151"
MUTED  = "#9CA3AF"
CLASS_STYLE = {                      # color + marker (identity never color-alone)
    "dark":  ("#9CA3AF", "o"),
    "lamp":  ("#2563EB", "s"),
    "flame": ("#DC2626", "^"),
}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})

with open(os.path.join(EXPORTS_DIR, "noise_results.json")) as f:
    res = json.load(f)

sig_nA = [a["sigma_nA"] for a in res["aggregate"]]
mean   = np.array([a["acc_mean"] for a in res["aggregate"]]) * 100
std    = np.array([a["acc_std"] for a in res["aggregate"]]) * 100
AUG_NA = 0.05 * res["data_range_nA"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

# ── Panel (a): accuracy vs sigma ─────────────────────────────────────────────
for seed, rows in res["per_seed"].items():
    ax1.plot([r["sigma_nA"] for r in rows],
             [r["accuracy"] * 100 for r in rows],
             color=INK, alpha=0.20, linewidth=0.9, zorder=2)
ax1.fill_between(sig_nA, mean - std, mean + std,
                 color=ACCENT, alpha=0.15, linewidth=0, zorder=2)
ax1.plot(sig_nA, mean, color=ACCENT, linewidth=2, marker="o",
         markersize=4.5, zorder=4, label="mean (5 seed models)")
ax1.axvline(AUG_NA, color=MUTED, linewidth=1, linestyle="--", zorder=1)
ax1.annotate("training aug. σ", (AUG_NA, 44), rotation=90,
             textcoords="offset points", xytext=(-9, 0),
             fontsize=7.5, color=MUTED, ha="center")
ax1.axhline(100 / 3, color=MUTED, linewidth=0.8, linestyle=":", zorder=1)
ax1.annotate("chance (33.3%)", (2.05, 100 / 3), textcoords="offset points",
             xytext=(0, 4), fontsize=7.5, color=MUTED)

ax1.set_xlabel("Added test-time noise σ (nA)")
ax1.set_ylabel("Test accuracy (%)")
ax1.set_ylim(28, 103)
ax1.set_xlim(-0.08, 3.2)
ax1.yaxis.grid(True, color=MUTED, linewidth=0.4, alpha=0.5)
ax1.set_axisbelow(True)
ax1.spines[["top", "right"]].set_visible(False)
ax1.legend(loc="upper right", fontsize=7.5, frameon=False)
ax1.set_title("(a) Accuracy under additive noise", fontsize=9.5, loc="left")

# ── Panel (b): per-class recall (mean over seeds) ────────────────────────────
seeds = list(res["per_seed"])
for cls, (color, marker) in CLASS_STYLE.items():
    curves = np.array([[r["recall"][cls] for r in res["per_seed"][s]]
                       for s in seeds]) * 100
    m = curves.mean(axis=0)
    ax2.plot(sig_nA, m, color=color, linewidth=2, marker=marker,
             markersize=4.5, label=cls.capitalize(), zorder=3)

ax2.axvline(AUG_NA, color=MUTED, linewidth=1, linestyle="--", zorder=1)
ax2.set_xlabel("Added test-time noise σ (nA)")
ax2.set_ylabel("Mean recall (%)")
ax2.set_ylim(-4, 104)
ax2.set_xlim(-0.08, 3.2)
ax2.yaxis.grid(True, color=MUTED, linewidth=0.4, alpha=0.5)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(loc="center right", fontsize=7.5, frameon=False)
ax2.set_title("(b) Per-class recall — failure mode", fontsize=9.5, loc="left")

fig.tight_layout()
out = os.path.join(EXPORTS_DIR, "noise_robustness.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
