"""
visualize_matched.py
--------------------
Figure for the parameter-matched CNN vs LSTM experiment.

Single panel: test accuracy vs trainable parameters (log x), colored by
architecture (CNN blue circles, LSTM orange triangles — color + marker, never
color alone). Per-seed runs as faint dots, mean +/- std as solid markers.
If capacity matters, LSTM@30k and CNN@30k should agree; they don't.

Output: exports_added/matched_comparison.png (300 dpi)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

EXPORTS_DIR = "exports_added"

INK   = "#374151"
MUTED = "#9CA3AF"
ARCH_STYLE = {"CNN": ("#2563EB", "o"), "LSTM": ("#D97706", "^")}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})

with open(os.path.join(EXPORTS_DIR, "matched_results.json")) as f:
    matched = json.load(f)
with open(os.path.join(EXPORTS_DIR, "capacity_results.json")) as f:
    capacity = json.load(f)

# (label, arch, summary, annotation offset in points, ha)
GROUPS = [
    ("CNN (proposed)", "CNN",  capacity["proposed"],   (-14, -4),  "right"),
    ("CNN widened",    "CNN",  matched["cnn_30k"],     (-14, -14), "right"),
    ("LSTM narrowed",  "LSTM", matched["lstm_11k"],    (-16, -8),  "right"),
    ("LSTM (original)", "LSTM", matched["lstm_30k"],   (16, -8),   "left"),
]

fig, ax = plt.subplots(figsize=(4.6, 3.2))
rng = np.random.default_rng(0)
seen_arch = set()

for label, arch, s, (dx, dy), ha in GROUPS:
    color, marker = ARCH_STYLE[arch]
    p = s["params"]
    accs = np.array([r["test_accuracy"] for r in s["per_seed"]]) * 100
    jit = p * rng.uniform(-0.06, 0.06, size=len(accs))
    ax.scatter(p + jit, accs, s=18, color=color, alpha=0.35,
               marker=marker, edgecolors="none", zorder=3)
    ax.errorbar(p, s["acc_mean"] * 100, yerr=s["acc_std"] * 100,
                fmt=marker, color=color, markersize=8, capsize=4,
                elinewidth=1.4, zorder=4,
                label=arch if arch not in seen_arch else None)
    seen_arch.add(arch)
    ax.annotate(f"{label}\n{s['acc_mean']*100:.1f}%",
                (p, s["acc_mean"] * 100), textcoords="offset points",
                xytext=(dx, dy), ha=ha, fontsize=7.5, color=INK)

ax.set_xscale("log")
ax.set_xlim(6e3, 6e4)
ax.set_ylim(55, 104)
ax.set_xlabel("Trainable parameters")
ax.set_ylabel("Test accuracy (%)")
ax.yaxis.grid(True, color=MUTED, linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower left", fontsize=8, frameon=False, title=None)
ax.set_title("Parameter-matched CNN vs LSTM (5 seeds each)",
             fontsize=9.5, loc="left")

fig.tight_layout()
out = os.path.join(EXPORTS_DIR, "matched_comparison.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
