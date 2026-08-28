# README Wording Fix + Multi-seed Comparison Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a data-provenance-adjacent wording issue in `README.md` and add a missing multi-seed CNN-vs-LSTM comparison figure to `main_experiments`, so the code package's documentation and figures are consistent with the manuscript revision direction from Prof. Kim's 2026-08-28 feedback.

**Architecture:** Two independent, unrelated one-file changes. No shared code, no ordering dependency between them — either can be done first.

**Tech Stack:** Python 3, matplotlib, numpy, json (stdlib). No new dependencies.

**Spec:** N/A — this is a **bounded task** per `superpowers:brainstorming`'s bounded path (small, well-scoped changes to files that already exist in this repo). Per that skill's rules, bounded tasks get a short design presented and approved in chat, not a written spec doc. The design was presented and the two concrete changes were confirmed across this conversation (see the "help me verify" and "đào sâu hơn" turns on 2026-08-28): (1) fix `README.md`'s "shot-noise-dominated" wording, (2) add `main_experiments/visualize_multiseed.py`.

## Global Constraints

- No git repository exists in this working directory — **do not run `git add`/`git commit`**. Steps that would normally end in a commit instead end in a manual verification (diff/grep/visual check).
- No test framework (pytest, unittest, etc.) exists anywhere in this repo — do not introduce one for these two tasks. Verification is "run it and check the real output," matching how every other script in this repo is validated.
- Match existing code style exactly: the plotting style constants (`ACCENT`/`INK`/`MUTED` hex colors, `rcParams` block, spine/grid conventions) are copied verbatim from `added_experiments/visualize_matched.py` and `added_experiments/visualize_capacity.py` — do not invent a new visual style.
- Working directory for all Python commands in Task 2 is `main_experiments/` (scripts in this repo use relative paths like `exports/...`).

---

### Task 1: Fix README.md provenance-adjacent wording

**Files:**
- Modify: `README.md:9`

**Interfaces:** None — pure text edit, no code interfaces involved.

- [ ] **Step 1: Make the edit**

Current text at `README.md:9` (part of the sentence spanning lines 7–11):

```
The classifier distinguishes three photocurrent sources from 1-second
windows of simulated AR(1) time series: **Dark / No source**, **Artificial
UV lamp** (shot-noise-dominated, AR ρ ≈ 0.06), and **Real invisible flame**
(turbulence-correlated, AR ρ ≈ 0.90). Lamp and flame share the same mean and
standard deviation and differ only in temporal autocorrelation.
```

Change `shot-noise-dominated` → `idealized, weakly correlated` so the parenthetical reads `(idealized, weakly correlated, AR ρ ≈ 0.06)`. Rationale: "shot-noise-dominated" implies an observed physical noise mechanism from real acquired data; since this signal is simulated, the wording should describe it as an idealized/representative simulation choice instead (matches Prof. Kim's requested phrasing for the manuscript itself).

Full corrected sentence:

```
The classifier distinguishes three photocurrent sources from 1-second
windows of simulated AR(1) time series: **Dark / No source**, **Artificial
UV lamp** (idealized, weakly correlated, AR ρ ≈ 0.06), and **Real invisible flame**
(turbulence-correlated, AR ρ ≈ 0.90). Lamp and flame share the same mean and
standard deviation and differ only in temporal autocorrelation.
```

- [ ] **Step 2: Verify no remaining occurrences**

Run: `grep -in "shot.noise" README.md`
Expected: no output (empty result — grep exit code 1).

- [ ] **Step 3: Confirm the diff is exactly the intended one-word-phrase change**

Run: `grep -n "idealized, weakly correlated" README.md`
Expected: prints line 9 with the new wording, confirming the edit landed in the right place and nothing else on the line changed.

No commit (no git repo) — task is done once Steps 2–3 pass.

---

### Task 2: Add `main_experiments/visualize_multiseed.py`

**Why this task exists:** `main_experiments/exports/multiseed_results.json` (produced by `multiseed_train.py`) has never had a corresponding figure — the only existing CNN-vs-LSTM comparison figure, `exports/comparison_bar.png` (from `compare.py`), is a single-run result. Per Prof. Kim's feedback, the multi-seed result should be the headline model-comparison figure, so it needs its own plot.

**Files:**
- Create: `main_experiments/visualize_multiseed.py`
- Test/verification: none (no test framework in this repo — see Global Constraints)

**Interfaces:**
- Consumes: `main_experiments/exports/multiseed_results.json`, with this exact schema (confirmed by reading the file):
  ```json
  {
    "cnn":  {"mean": 1.0,   "std": 0.0,    "min": 1.0,    "max": 1.0,
             "per_seed": [{"seed": 42, "test_accuracy": 1.0, "best_val_acc": 1.0, "stopped_epoch": 32}, ...]},
    "lstm": {"mean": 0.797, "std": 0.1235, "min": 0.6667, "max": 0.9813,
             "per_seed": [{"seed": 42, "test_accuracy": 0.6667, "best_val_acc": 0.6667, "stopped_epoch": 21}, ...]}
  }
  ```
  Note the top-level keys are `mean`/`std`/`min`/`max` (NOT `acc_mean`/`acc_std` — that's the `added_experiments` schema, a different convention. Do not mix them up.)
- Produces: `main_experiments/exports/multiseed_comparison.png` (300 dpi). No other code in this repo imports this script, so no downstream consumers to keep in sync.

- [ ] **Step 1: Write the script**

Create `main_experiments/visualize_multiseed.py`:

```python
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
```

- [ ] **Step 2: Run it**

Run:
```bash
cd main_experiments
python visualize_multiseed.py
```
Expected stdout: `Saved exports/multiseed_comparison.png`

- [ ] **Step 3: Verify the output file exists and is non-trivial**

Run: `ls -la main_experiments/exports/multiseed_comparison.png`
Expected: file exists, size in the tens-to-hundreds of KB range (consistent with the other 300 dpi PNGs already in `exports/`, e.g. `comparison_bar.png`).

- [ ] **Step 4: Visually verify the figure is correct**

Open `main_experiments/exports/multiseed_comparison.png` (e.g. via the Read tool, which renders images) and confirm all of the following:
- Two x-axis categories: "CNN (10,755 params)" and "LSTM (30,723 params)".
- CNN: 5 tightly clustered gray dots at 100%, one blue circle marker at 100% with a zero-length error bar, annotated "100.0% ± 0.0pp".
- LSTM: 5 spread-out gray dots (roughly 66.7%–98.1%), one orange triangle marker near 79.7% with a large error bar (~±12.4pp), annotated "79.7% ± 12.4pp".
- No clipped labels, no dots outside the plot area (y-axis range 60–104% must contain the LSTM min of 66.67%).

If any of these don't match, the script has a bug — do not proceed until the figure looks like this description.

No commit (no git repo) — task is done once Step 4 passes.
