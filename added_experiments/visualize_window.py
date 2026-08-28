"""
visualize_window.py
-------------------
Figure for the window-length ablation (proposed 1D-CNN, 0.5/1/2 s windows).

Single panel: per-seed dots + mean +/- std per window length. The x-axis is
also the decision latency (one window = one decision).

Output: exports_added/window_ablation.png (300 dpi)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

EXPORTS_DIR = "exports_added"
ACCENT = "#2563EB"
INK    = "#374151"
MUTED  = "#9CA3AF"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})

with open(os.path.join(EXPORTS_DIR, "window_results.json")) as f:
    res = json.load(f)

order = sorted(res.values(), key=lambda s: s["window_samples"])

fig, ax = plt.subplots(figsize=(4.6, 3.2))
rng = np.random.default_rng(0)

for i, s in enumerate(order):
    accs = np.array([r["test_accuracy"] for r in s["per_seed"]]) * 100
    jit = rng.uniform(-0.08, 0.08, size=len(accs))
    ax.scatter(np.full(len(accs), float(i)) + jit, accs, s=22, color=INK,
               alpha=0.55, edgecolors="none", zorder=3,
               label="per-seed run" if i == 0 else None)
    ax.errorbar(i, s["acc_mean"] * 100, yerr=s["acc_std"] * 100,
                fmt="o", color=ACCENT, markersize=7, capsize=4,
                elinewidth=1.4, zorder=4,
                label="mean ± std (5 seeds)" if i == 0 else None)
    ax.annotate(f"{s['acc_mean']*100:.2f}%",
                (i, s["acc_mean"] * 100), textcoords="offset points",
                xytext=(12, 4), fontsize=8, color=ACCENT)

ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{s['window_seconds']:.1f} s\n({s['window_samples']} samples,"
                    f" n={s['per_seed'][0]['n_test']})" for s in order])
ax.set_xlim(-0.5, len(order) - 0.5)
ax.set_ylabel("Test accuracy (%)")
ax.set_xlabel("Window length = decision latency")
ax.yaxis.grid(True, color=MUTED, linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", fontsize=7.5, frameon=False)
ax.set_title("Window-length ablation (proposed 1D-CNN)",
             fontsize=9.5, loc="left")

ymin = min(min(r["test_accuracy"] for r in s["per_seed"]) for s in order) * 100
ax.set_ylim(min(90, ymin - 2), 101)

fig.tight_layout()
out = os.path.join(EXPORTS_DIR, "window_ablation.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
