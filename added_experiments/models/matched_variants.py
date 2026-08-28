"""
models/matched_variants.py
--------------------------
Parameter-matched variants for the capacity-vs-architecture experiment.

The claim under test: the LSTM's failure is NOT explained by parameter count.
Evidence sought in both directions:

  cnn_30k   : proposed CNN topology widened to ~= LSTM's 30,723 params
              (32,259 params, within +5%)  -> expected to still succeed
  lstm_11k  : original diff-channel LSTM narrowed to ~= CNN's 10,755 params
              (11,080 params, within +3%)  -> expected to remain unstable

The original LSTM (30,723 params) lives in models/lstm_model.py.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_30k(timesteps: int, num_classes: int = 3) -> tf.keras.Model:
    """Proposed CNN widened (56/112 filters, Dense 112) -> 32,259 params."""
    inp = layers.Input(shape=(timesteps, 1), name="input")

    x = layers.Conv1D(56, kernel_size=5, padding="same",
                      activation="relu", name="conv1")(inp)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = layers.Conv1D(112, kernel_size=3, padding="same",
                      activation="relu", name="conv2")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    x = layers.Dense(112, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return models.Model(inputs=inp, outputs=out, name="CNN_30k")


def build_lstm_11k(timesteps: int, num_classes: int = 3) -> tf.keras.Model:
    """Original diff-channel LSTM narrowed (38/19 units, Dense 19) -> ~11k params."""
    inp = layers.Input(shape=(timesteps, 1), name="input")

    diff   = layers.Lambda(lambda z: z[:, 1:, :] - z[:, :-1, :], name="diff")(inp)
    x_trim = layers.Lambda(lambda z: z[:, 1:, :], name="trim")(inp)
    x      = layers.Concatenate(axis=-1, name="aug_input")([x_trim, diff])

    x = layers.LSTM(38, return_sequences=True, name="lstm1")(x)
    x = layers.Dropout(0.1, name="dropout1")(x)

    x = layers.LSTM(19, return_sequences=True, name="lstm2")(x)
    x = layers.Dropout(0.1, name="dropout2")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(19, activation="relu", name="dense1")(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return models.Model(inputs=inp, outputs=out, name="LSTM_11k")


if __name__ == "__main__":
    from lstm_model import build_lstm
    for builder in (build_cnn_30k, build_lstm_11k):
        m = builder(timesteps=100)
        print(f"{m.name}: {m.count_params():,} params")
