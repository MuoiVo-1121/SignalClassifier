"""
window_ablation.py
------------------
Experiment 2: window-length ablation for the proposed 1D-CNN.

Windows of 0.5 s / 1 s / 2 s (50 / 100 / 200 samples at 100 Hz), keeping the
original overlap ratios: train step = W/4 (75% overlap), eval step = W/2
(50% overlap). Each window length gets its own processed dir, so this can run
concurrently with matched_compare.py (which uses data/processed).

Supports the manuscript's ~1 s decision-latency claim: does a shorter window
(faster decision) sacrifice accuracy, and does a longer window buy anything?

Note: the number of val/test windows changes with W (same temporal segments,
different slicing) — n_test is reported alongside accuracy.

Usage:
    python window_ablation.py            # resumable
    python window_ablation.py --windows 50 200 --seeds 42 123
"""

import argparse
import os

from models.cnn_model import build_cnn
from common_runner import (SEEDS, EXPORTS_DIR, run_training, summarize,
                           resume_map, save)

WINDOWS = {
    50:  {"train_step": 12, "eval_step": 25},
    100: {"train_step": 25, "eval_step": 50},
    200: {"train_step": 50, "eval_step": 100},
}
OUT_PATH = os.path.join(EXPORTS_DIR, "window_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", type=int,
                        default=list(WINDOWS), choices=list(WINDOWS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    all_results = resume_map(OUT_PATH)

    for w in args.windows:
        key = f"w{w}"
        params = build_cnn(timesteps=w).count_params()  # GAP -> same params
        print(f"\n{'='*55}\n  WINDOW {w} samples ({w/100:.1f} s)  "
              f"({params:,} params)\n{'='*55}", flush=True)
        done = {r["seed"]: r
                for r in all_results.get(key, {}).get("per_seed", [])}
        runs = []
        for seed in args.seeds:
            if seed in done:
                print(f"  [skip] seed={seed} already done "
                      f"(test_acc={done[seed]['test_accuracy']*100:.2f}%)",
                      flush=True)
                runs.append(done[seed])
                continue
            runs.append(run_training(
                build_cnn, key, seed,
                processed_dir=f"data/processed_{key}",
                window_size=w, **WINDOWS[w]))
            all_results[key] = summarize(params, runs,
                                         extra={"window_samples": w,
                                                "window_seconds": w / 100})
            save(OUT_PATH, all_results)
        all_results[key] = summarize(params, runs,
                                     extra={"window_samples": w,
                                            "window_seconds": w / 100})
    save(OUT_PATH, all_results)

    print(f"\n{'='*78}")
    print(f"  {'Window':<10} {'n_test':>6} {'Test acc (mean+/-std)':>24} {'Range':>20}")
    print(f"{'='*78}")
    for key, s in sorted(all_results.items(),
                         key=lambda kv: kv[1]["window_samples"]):
        n_test = s["per_seed"][0]["n_test"]
        print(f"  {s['window_seconds']:>5.1f} s   {n_test:>6} "
              f"{s['acc_mean']*100:>10.2f}% +/- {s['acc_std']*100:.2f}pp "
              f"  [{s['acc_min']*100:.2f}, {s['acc_max']*100:.2f}]%")
    print(f"{'='*78}")
    print(f"\nResults saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
