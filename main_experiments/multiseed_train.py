"""
multiseed_train.py
------------------
Runs CNN and LSTM training across multiple random seeds to report
mean ± std test accuracy — demonstrating result robustness beyond
a single seed=42 run.

Usage:
    python multiseed_train.py
    python multiseed_train.py --seeds 42 123 7 2024 99
    python multiseed_train.py --model cnn --seeds 42 123 7
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf

from preprocess import preprocess
from train import train as train_model

PROCESSED_DIR = "data/processed"
EXPORTS_DIR   = "exports"
SEEDS         = [42, 123, 7, 2024, 99]


def run_seed(seed: int, model_name: str, epochs: int, batch_size: int) -> dict:
    """Run one full preprocess + train cycle for a given seed."""
    print(f"\n{'─'*55}")
    print(f"  seed={seed}  model={model_name.upper()}")
    print(f"{'─'*55}")

    # Patch global seed in preprocess module
    import preprocess as pre_mod
    orig_seed = pre_mod.RANDOM_SEED
    pre_mod.RANDOM_SEED = seed
    preprocess(raw_dir="data/raw", processed_dir=PROCESSED_DIR)
    pre_mod.RANDOM_SEED = orig_seed

    # Patch WEIGHTS_DIR to avoid overwriting the seed=42 best weights
    import train as train_mod
    orig_weights = train_mod.WEIGHTS_DIR
    seed_weights = f"weights/seed_{seed}"
    os.makedirs(seed_weights, exist_ok=True)
    train_mod.WEIGHTS_DIR = seed_weights

    orig_exports = train_mod.EXPORTS_DIR
    seed_exports = os.path.join(EXPORTS_DIR, f"multiseed_{model_name}_seed{seed}")
    os.makedirs(seed_exports, exist_ok=True)
    train_mod.EXPORTS_DIR = seed_exports

    model, history = train_model(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        processed_dir=PROCESSED_DIR,
    )
    train_mod.WEIGHTS_DIR = orig_weights
    train_mod.EXPORTS_DIR = orig_exports

    # Evaluate on test set
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    from sklearn.metrics import accuracy_score
    y_pred    = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_acc  = float(accuracy_score(y_test, y_pred))
    best_val  = float(max(history.history["val_accuracy"]))
    stopped   = int(len(history.history["loss"]))

    print(f"  → test_acc={test_acc*100:.2f}%  best_val={best_val*100:.2f}%  "
          f"stopped_epoch={stopped}")
    return {
        "seed":          seed,
        "test_accuracy": round(test_acc, 6),
        "best_val_acc":  round(best_val, 6),
        "stopped_epoch": stopped,
    }


def report(model_name: str, seed_results: list):
    accs = [r["test_accuracy"] for r in seed_results]
    mean = float(np.mean(accs))
    std  = float(np.std(accs))
    mn   = float(np.min(accs))
    mx   = float(np.max(accs))

    print(f"\n{'='*55}")
    print(f"  MULTI-SEED RESULTS — {model_name.upper()}")
    print(f"{'='*55}")
    for r in seed_results:
        print(f"  seed={r['seed']:>4}  test={r['test_accuracy']*100:.2f}%  "
              f"val={r['best_val_acc']*100:.2f}%  epoch={r['stopped_epoch']}")
    print(f"  {'─'*45}")
    print(f"  Mean  : {mean*100:.2f}%")
    print(f"  Std   : {std*100:.2f} pp")
    print(f"  Range : [{mn*100:.2f}%, {mx*100:.2f}%]")
    print(f"{'='*55}")
    return {"mean": round(mean,4), "std": round(std,4),
            "min": round(mn,4), "max": round(mx,4), "per_seed": seed_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--model", choices=["cnn","lstm","both"], default="both")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    models = ["cnn","lstm"] if args.model == "both" else [args.model]
    all_results = {}

    for model_name in models:
        lr = 3e-4 if model_name == "lstm" else 1e-3

        # Patch LR in train module per model type
        import train as train_mod
        orig_lr = None

        seed_results = []
        for seed in args.seeds:
            r = run_seed(seed, model_name, args.epochs, args.batch_size)
            seed_results.append(r)

        all_results[model_name] = report(model_name, seed_results)

    # Save
    out_path = os.path.join(EXPORTS_DIR, "multiseed_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
