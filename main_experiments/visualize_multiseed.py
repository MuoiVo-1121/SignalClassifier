"""
visualize_multiseed.py
-----------------------
Figure for the primary multi-seed CNN vs LSTM comparison (5 seeds each).

Promotes the multiseed_train.py results to a headline comparison figure,
rather than relying solely on the single-run exports/comparison_bar.png
(from compare.py).

Single panel: per-seed dots + mean +/- std per model, on a categorical
x-axis (CNN, LSTM) — same visual language as
added_experiments/visualize_matched.py (color + marker: CNN blue circle,
LSTM orange triangle, never color alone).

Output: exports/multiseed_comparison.png (300 dpi)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

EXPORTS_DIR = "exports"
ORDER  = ["cnn", "lstm"]
LABELS = {"cnn": "CNN\n(10,755 params)", "lstm": "LSTM\n(30,723 params)"}
STYLE  = {"cnn": ("#2563EB", "o"), "lstm": ("#D97706", "^")}
INK    = "#374151"
MUTED  = "#9CA3AF"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})

with open(os.path.join(EXPORTS_DIR, "multiseed_results.json")) as f:
    results = json.load(f)

fig, ax = plt.subplots(figsize=(4.6, 3.2))
rng = np.random.default_rng(0)

for i, key in enumerate(ORDER):
    s = results[key]
    color, marker = STYLE[key]
    accs = np.array([r["test_accuracy"] for r in s["per_seed"]]) * 100
    jitter = rng.uniform(-0.10, 0.10, size=len(accs))
    ax.scatter(np.full(len(accs), float(i)) + jitter, accs,
               s=22, color=INK, alpha=0.55, zorder=3, edgecolors="none",
               label="per-seed run" if i == 0 else None)
    ax.errorbar(i, s["mean"] * 100, yerr=s["std"] * 100,
                fmt=marker, color=color, markersize=8, capsize=4,
                elinewidth=1.4, zorder=4,
                label="mean ± std (5 seeds)" if i == 0 else None)
    ax.annotate(f"{s['mean']*100:.1f}% ± {s['std']*100:.1f}pp",
                (i, s["mean"] * 100), textcoords="offset points",
                xytext=(12, 4), fontsize=8, color=color)

ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([LABELS[k] for k in ORDER])
ax.set_xlim(-0.5, len(ORDER) - 0.5)
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(60, 104)
ax.yaxis.grid(True, color=MUTED, linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower left", fontsize=7.5, frameon=False)
ax.set_title("Multi-seed robustness: CNN vs LSTM (5 seeds each)",
             fontsize=9.5, loc="left")

fig.tight_layout()
out = os.path.join(EXPORTS_DIR, "multiseed_comparison.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
