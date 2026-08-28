"""
evaluate.py
-----------
Load saved weights for CNN and/or LSTM, run on the test set,
and print accuracy, classification report, and confusion matrix.

Usage:
    python evaluate.py --model cnn
    python evaluate.py --model lstm
    python evaluate.py --model both
"""

import argparse
import os
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

from models.cnn_model import build_cnn
from models.lstm_model import build_lstm

PROCESSED_DIR = "data/processed"
WEIGHTS_DIR   = "weights"
EXPORTS_DIR   = "exports"
CLASS_NAMES   = ["Dark", "UVC Lamp", "Flame"]


def load_test(processed_dir: str):
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    return X_test, y_test


def evaluate_model(model_type: str, X_test: np.ndarray, y_test: np.ndarray):
    timesteps   = X_test.shape[1]
    weight_path = os.path.join(WEIGHTS_DIR, f"{model_type}_best.h5")

    model = build_cnn(timesteps) if model_type == "cnn" else build_lstm(timesteps)
    model.load_weights(weight_path)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Model : {model_type.upper()}  |  Test Accuracy : {acc * 100:.2f}%")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    _, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_type.upper()}")
    plt.tight_layout()
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    out_path = os.path.join(EXPORTS_DIR, f"confusion_matrix_{model_type}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    return acc, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "lstm", "both"], default="both")
    args = parser.parse_args()

    X_test, y_test = load_test(PROCESSED_DIR)
    targets = ["cnn", "lstm"] if args.model == "both" else [args.model]
    for m in targets:
        evaluate_model(m, X_test, y_test)
