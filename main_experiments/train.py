"""
train.py
--------
Train either the 1D CNN or LSTM model on preprocessed data.
Saves best weights and training history for downstream visualization.

Usage:
    python train.py --model cnn
    python train.py --model lstm
    python train.py --model cnn --epochs 80 --batch_size 64
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from models.cnn_model import build_cnn
from models.lstm_model import build_lstm

PROCESSED_DIR = "data/processed"
WEIGHTS_DIR   = "weights"
EXPORTS_DIR   = "exports"
CLASS_NAMES   = ["dark", "uvc_lamp", "methanol_flame"]
RANDOM_SEED   = 42


def load_splits(processed_dir: str):
    def load(name):
        return np.load(os.path.join(processed_dir, name))
    return (
        load("X_train.npy"), load("X_val.npy"),  load("X_test.npy"),
        load("y_train.npy"), load("y_val.npy"),  load("y_test.npy"),
    )


def train(model_name: str,
          epochs: int    = 80,
          batch_size: int = 64,
          lr: float      = 1e-3,
          processed_dir: str = PROCESSED_DIR):

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(processed_dir)
    timesteps = X_train.shape[1]

    print(f"\nData loaded — timesteps: {timesteps}, "
          f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # ── Build model ──────────────────────────────────────────────────────────
    if model_name == "cnn":
        model = build_cnn(timesteps=timesteps)
    elif model_name == "lstm":
        model = build_lstm(timesteps=timesteps)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'cnn' or 'lstm'.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Callbacks ────────────────────────────────────────────────────────────
    os.makedirs(WEIGHTS_DIR,  exist_ok=True)
    os.makedirs(EXPORTS_DIR,  exist_ok=True)
    weights_path = os.path.join(WEIGHTS_DIR, f"{model_name}_best.h5")

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=20,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(weights_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=8, min_lr=1e-6, verbose=1),
    ]

    # ── Training ─────────────────────────────────────────────────────────────
    print(f"\nTraining {model_name.upper()} for up to {epochs} epochs …\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save training history ─────────────────────────────────────────────────
    history_path = os.path.join(EXPORTS_DIR, f"history_{model_name}.json")
    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    history_dict["stopped_epoch"] = int(
        len(history.history["loss"])
    )
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{model_name.upper()} — Test loss: {loss:.4f}  |  Test accuracy: {acc:.4f}")
    print(f"Best weights saved to: {weights_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str,   default="cnn",
                        choices=["cnn", "lstm"])
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    train(model_name=args.model,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr)
