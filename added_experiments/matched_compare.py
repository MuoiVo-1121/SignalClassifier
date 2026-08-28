"""
matched_compare.py
------------------
Experiment 1: parameter-matched CNN vs LSTM — is the LSTM's failure a
capacity effect?

Runs (5 seeds each, same protocol/environment as capacity_compare.py):
  lstm_30k : original diff-channel LSTM      (30,723 params)  [models/lstm_model.py]
  cnn_30k  : proposed CNN widened to ~30k    (32,259 params, +5%)
  lstm_11k : LSTM narrowed to ~CNN size      (11,080 params, +3% vs 10,755)

The proposed CNN (10,755) reference row is read from
exports_added/capacity_results.json at report time.

Note on LR: the original multiseed_train.py computed lr=3e-4 for the LSTM but
never passed it (dead code) — the published multiseed LSTM numbers were
produced with lr=1e-3. We therefore use lr=1e-3 for all models here.

Usage:
    python matched_compare.py                # resumable
    python matched_compare.py --models lstm_30k --seeds 42
"""

import argparse
import json
import os

from models.lstm_model import build_lstm
from models.matched_variants import build_cnn_30k, build_lstm_11k
from common_runner import (SEEDS, EXPORTS_DIR, run_training, summarize,
                           resume_map, save)

BUILDERS = {
    "lstm_30k": build_lstm,
    "cnn_30k":  build_cnn_30k,
    "lstm_11k": build_lstm_11k,
}
OUT_PATH = os.path.join(EXPORTS_DIR, "matched_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(BUILDERS),
                        choices=list(BUILDERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    all_results = resume_map(OUT_PATH)

    for key in args.models:
        params = BUILDERS[key](timesteps=100).count_params()
        print(f"\n{'='*55}\n  {key.upper()}  ({params:,} params)\n{'='*55}",
              flush=True)
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
            runs.append(run_training(BUILDERS[key], key, seed,
                                     processed_dir="data/processed"))
            all_results[key] = summarize(params, runs)
            save(OUT_PATH, all_results)
        all_results[key] = summarize(params, runs)
    save(OUT_PATH, all_results)

    # Report incl. proposed-CNN reference from the capacity experiment
    cap_path = os.path.join(EXPORTS_DIR, "capacity_results.json")
    rows = []
    if os.path.exists(cap_path):
        with open(cap_path) as f:
            cap = json.load(f)
        if "proposed" in cap:
            rows.append(("cnn_10k (proposed)", cap["proposed"]))
    rows += [(k, all_results[k]) for k in all_results]

    print(f"\n{'='*76}")
    print(f"  {'Model':<20} {'Params':>8} {'Test acc (mean+/-std)':>24} {'Range':>20}")
    print(f"{'='*76}")
    for name, s in rows:
        print(f"  {name:<20} {s['params']:>8,} "
              f"{s['acc_mean']*100:>10.2f}% +/- {s['acc_std']*100:.2f}pp "
              f"  [{s['acc_min']*100:.2f}, {s['acc_max']*100:.2f}]%")
    print(f"{'='*76}")
    print(f"\nResults saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
