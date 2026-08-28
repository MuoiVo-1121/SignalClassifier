"""
models/capacity_variants.py
---------------------------
Capacity ablation variants of the proposed 1D CNN (models/cnn_model.py).

Design principle: keep the topology family fixed (Conv1D blocks -> GAP ->
Dense -> Softmax) and vary only capacity, so the comparison isolates model
size rather than architectural style.

  Tiny-CNN  : same 2-conv depth as proposed, ~1/4 width      (~0.8k params)
  Proposed  : Conv(32,5)-Conv(64,3), Dense(64)               (10,755 params)
  Deep-CNN  : 4 conv blocks (32-64-128-128), Dense(128)      (~97k params)
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_tiny_cnn(timesteps: int, num_classes: int = 3) -> tf.keras.Model:
    """Width-reduced version of the proposed CNN (same depth, 1/4 filters)."""
    inp = layers.Input(shape=(timesteps, 1), name="input")

    x = layers.Conv1D(8, kernel_size=5, padding="same",
                      activation="relu", name="conv1")(inp)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = layers.Conv1D(16, kernel_size=3, padding="same",
                      activation="relu", name="conv2")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    x = layers.Dense(16, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return models.Model(inputs=inp, outputs=out, name="Tiny_CNN")


def build_deep_cnn(timesteps: int, num_classes: int = 3) -> tf.keras.Model:
    """Depth/width-increased version of the proposed CNN (4 conv blocks)."""
    inp = layers.Input(shape=(timesteps, 1), name="input")

    x = layers.Conv1D(32, kernel_size=5, padding="same",
                      activation="relu", name="conv1")(inp)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = layers.Conv1D(64, kernel_size=3, padding="same",
                      activation="relu", name="conv2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same",
                      activation="relu", name="conv3")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool3")(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same",
                      activation="relu", name="conv4")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return models.Model(inputs=inp, outputs=out, name="Deep_CNN")


if __name__ == "__main__":
    for builder in (build_tiny_cnn, build_deep_cnn):
        m = builder(timesteps=100)
        print(f"{m.name}: {m.count_params():,} params")
