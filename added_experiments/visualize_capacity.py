"""
visualize_capacity.py
---------------------
Figure for the capacity ablation (Tiny / Proposed / Deep 1D-CNN).

Panel (a): test accuracy — per-seed dots + mean +/- std (point plot, so the
           truncated y-axis is legitimate; bars would forbid it).
Panel (b): parameter count (log scale) and inference time per window.

Output: exports_added/capacity_comparison.png (300 dpi)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

EXPORTS_DIR = "exports_added"
ORDER       = ["tiny", "proposed", "deep"]
LABELS      = {"tiny": "Tiny-CNN", "proposed": "Proposed 1D-CNN", "deep": "Deep-CNN"}

ACCENT  = "#2563EB"   # mean marker / emphasis
INK     = "#374151"   # dots, text
MUTED   = "#9CA3AF"   # grid, secondary

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

with open(os.path.join(EXPORTS_DIR, "capacity_results.json")) as f:
    results = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ── Panel (a): accuracy ──────────────────────────────────────────────────────
rng = np.random.default_rng(0)
for i, key in enumerate(ORDER):
    s     = results[key]
    accs  = np.array([r["test_accuracy"] for r in s["per_seed"]]) * 100
    jitter = rng.uniform(-0.10, 0.10, size=len(accs))
    ax1.scatter(np.full(len(accs), float(i)) + jitter, accs,
                s=22, color=INK, alpha=0.55, zorder=3, edgecolors="none",
                label="per-seed run" if i == 0 else None)
    ax1.errorbar(i, s["acc_mean"] * 100, yerr=s["acc_std"] * 100,
                 fmt="o", color=ACCENT, markersize=7, capsize=4,
                 elinewidth=1.4, zorder=4,
                 label="mean ± std (5 seeds)" if i == 0 else None)
    ax1.annotate(f"{s['acc_mean']*100:.2f}%",
                 (i, s["acc_mean"] * 100), textcoords="offset points",
                 xytext=(10, 4), fontsize=8, color=ACCENT)

ax1.set_xticks(range(len(ORDER)))
ax1.set_xticklabels([f"{LABELS[k]}\n({results[k]['params']:,} params)"
                     for k in ORDER])
ax1.set_ylabel("Test accuracy (%)")
ax1.set_ylim(90, 101)
ax1.set_xlim(-0.5, 2.5)
ax1.yaxis.grid(True, color=MUTED, linewidth=0.4, alpha=0.5)
ax1.set_axisbelow(True)
ax1.spines[["top", "right"]].set_visible(False)
ax1.legend(loc="lower right", fontsize=7.5, frameon=False)
ax1.set_title("(a) Accuracy vs. model capacity", fontsize=9.5, loc="left")

# ── Panel (b): params (log) with inference time labels ───────────────────────
params = [results[k]["params"] for k in ORDER]
times  = [results[k]["inference_ms"] for k in ORDER]
acc    = [results[k]["acc_mean"] * 100 for k in ORDER]

ax2.scatter(params, acc, s=46, color=ACCENT, zorder=3)
OFFSETS = {"tiny":     ((0, 10), "center"),
           "proposed": ((8, -26), "left"),
           "deep":     ((0, -30), "center")}
for k, p, a, t in zip(ORDER, params, acc, times):
    (dx, dy), ha = OFFSETS[k]
    ax2.annotate(f"{LABELS[k]}\n{t:.2f} ms/window",
                 (p, a), textcoords="offset points", xytext=(dx, dy),
                 ha=ha, fontsize=7.5, color=INK)

ax2.set_xscale("log")
ax2.set_xlabel("Trainable parameters")
ax2.set_ylabel("Mean test accuracy (%)")
ax2.set_ylim(96.5, 100.6)
ax2.set_xlim(3e2, 1e6)
ax2.yaxis.grid(True, color=MUTED, linewidth=0.4, alpha=0.5)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_title("(b) Accuracy–size–latency trade-off", fontsize=9.5, loc="left")

fig.tight_layout()
out = os.path.join(EXPORTS_DIR, "capacity_comparison.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
