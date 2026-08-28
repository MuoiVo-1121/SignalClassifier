# Photocurrent Signal Classification

Code, data, trained weights, and results for the 1D-CNN classifier section of
*"Self-Powered Deep-UV Photodiodes for Intelligent Invisible Flame
Detection"* (ZnGa2O4/Cu2O photodiode).
This repository accompanies the manuscript by T. M. Vo, Y. Kim, and C. W. Bark. The DOI will be added upon publication.
The classifier distinguishes three photocurrent sources from 1-second
windows of simulated AR(1) time series: **Dark / No source**, **Artificial
UV lamp** (idealized, weakly correlated, AR ρ ≈ 0.06), and **Real invisible flame**
(turbulence-correlated, AR ρ ≈ 0.90). Lamp and flame share the same mean and
standard deviation and differ only in temporal autocorrelation.

```

├── main_experiments/    results reported in the manuscript (§2.3) and SI
└── added_experiments/   supplementary ablations (capacity, parameter-matched
                         LSTM, window length, noise robustness)
```

---

## 1. main_experiments

### Which script produces which reported number

| Script | Produces |
|---|---|
| `generate_dummy_data.py` | The three AR(1) recordings (300 s @ 100 Hz per class) → `data/raw/*.csv` (the exact realizations used are included) |
| `preprocess.py` | Temporal 70/15/15 split → 5,022 / 267 / 267 windows, global min-max normalization, ×2 noise augmentation |
| `models/cnn_model.py` | Proposed 1D-CNN, 10,755 parameters (SI Table S2) |
| `models/lstm_model.py` | Stacked LSTM with explicit diff channel, 30,723 parameters |
| `train.py` | Training (Adam 1e-3, batch 64, early stopping) → `weights/*_best.h5` |
| `evaluate.py` | Test accuracy **99.25%**, macro-F1 0.993, confusion matrix (Fig. 6e, SI Table S3) |
| `compare.py` | CNN vs LSTM side-by-side incl. inference time per window |
| `multiseed_train.py` | 5-seed robustness {42,123,7,2024,99} → `exports/multiseed_results.json` |
| `baseline.py` | Threshold (100%), logistic regression (94.38%), random forest (96.63%) — SI Fig. S18–S19 |
| `ablation_ripple.py` | 5 Hz supply-ripple confound ablation |
| `spike_analysis.py` | Flame-spike confound analysis (98.78% on spike-free windows) |
| `visualize_signals.py` / `visualize_features.py` / `visualize_training.py` / `visualize_architecture.py` | Fig. 6a,c,d,f,g and SI Fig. S15–S17 |

### Included data & results

- `data/raw/` and `data/raw_no_ripple/` — the exact generated CSV recordings
  used for all reported numbers (regenerating and retraining will give
  slightly different numbers; see reproducibility note below).
- `weights/` — trained weights: `cnn_best.h5`, `lstm_best.h5` (primary
  result), `seed_*/` (multiseed runs), `ablation_no_ripple/`.
- `exports/` — result JSONs (multiseed, baseline, ablation, spike, training
  histories) and generated figures (PNG).

### Run order

```bash
pip install -r requirements.txt
python generate_dummy_data.py   # optional — data/raw is already included
python preprocess.py            # -> data/processed/
python train.py --model cnn
python train.py --model lstm
python evaluate.py --model both
python compare.py
python multiseed_train.py       # ~1-2 h on CPU
python baseline.py
python ablation_ripple.py
python spike_analysis.py
```

To **verify the reported numbers without retraining**, run `evaluate.py` /
`compare.py` directly against the included `weights/` after `preprocess.py`.

---

## 2. Supplementary ablations

Four experiments, each following the same 5-seed protocol (per-seed
re-preprocessing; training seed fixed; identical test set per window length).

| Script | Experiment | Headline result (mean ± std over 5 seeds) | Reported in |
|---|---|---|---|
| `capacity_compare.py` | Tiny (771) / Proposed (10,755) / Deep (97,283) CNN | 97.75 ± 2.50 / 98.13 ± 2.53 / 99.78 ± 0.45 % — accuracy saturated across 126× params | SI Fig. S20 |
| `matched_compare.py` | Parameter-matched CNN vs LSTM (both at ~11k and ~31k) | CNN 98.13 / 99.33 % vs LSTM 75.88 / 75.43 % — accuracy tracks architecture, not capacity | SI Fig. S21 |
| `window_ablation.py` | Window length 0.5 / 1 / 2 s | 99.78 / 99.10 / 98.64 % — 0.5 s already suffices | SI Fig. S22 |
| `noise_robustness.py` | Additive test-time noise sweep (0–3.07 nA) | robust to σ ≈ 0.2 nA; flame recall collapses (flame→lamp) around σ ≈ 0.5–1 nA | **Not reported**, pending measured device data |
- `common_runner.py` — shared training/eval machinery; `models/` — the CNN/LSTM
  builders plus the scaled variants; `visualize_*.py` — the figures.
- `exports_added/` — per-seed result JSONs and figures; `weights_added/` —
  trained weights of every run.
- All training scripts are **resumable**: re-invoking skips already-completed
  (model, seed) runs recorded in the results JSON.

```bash
pip install -r requirements.txt      # exact versions: requirements-lock.txt
python capacity_compare.py
python matched_compare.py
python window_ablation.py
python noise_robustness.py           # evaluation-only; uses weights from capacity_compare.py
python visualize_capacity.py && python visualize_matched.py
python visualize_window.py  && python visualize_noise.py
```

---

## Reproducibility notes (please read)

1. **Two environments.** `main_experiments` results were produced with
   TensorFlow 2.13 (macOS/Colab); `added_experiments` with TensorFlow 2.21
   (Windows/CPU). Training dynamics differ slightly across framework
   versions — e.g., the proposed CNN reproduces at 98.13 ± 2.53% in the
   TF 2.21 environment versus 100% on all five seeds in the original one.
   Each folder's results are internally consistent; do not mix absolute
   numbers across the two folders.
2. **Training is not bit-deterministic on CPU/GPU** (thread scheduling).
   Retraining reproduces results in distribution (within the reported
   seed-to-seed std), not digit-for-digit. The included `weights/` are the
   models behind the reported numbers.
3. **Seeds.** The five "seeds" vary preprocessing (augmentation noise and
   shuffling); the training seed is fixed at 42, and the raw recordings and
   the temporal val/test split are identical across seeds, so all runs of a
   given window length share the same test set.
4. **All signals are simulated** AR(1) realizations designed to model the
   photodiode's operating regime (see SI); no measured photocurrent data is
   included in this package.
