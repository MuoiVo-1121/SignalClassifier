"""
models/lstm_model.py
--------------------
Stacked LSTM for photocurrent signal classification.

Input shape : (batch, timesteps, 1)
Output      : softmax over 3 classes {dark, UVC lamp, methanol flame}
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm(timesteps: int, num_classes: int = 3) -> tf.keras.Model:
    """
    Architecture
    ------------
    Input  (timesteps, 1)
       │
    [x[t], x[t]-x[t-1]]  ← explicit diff channel makes autocorrelation visible:
                            small diff  → flame (ρ=0.90, smooth)
                            large diff  → lamp  (ρ=0.06, rough)
       │  (timesteps-1, 2)
    LSTM(64, return_sequences=True)
       │
    Dropout(0.1)
       │
    LSTM(32, return_sequences=True)
       │
    GlobalAveragePooling1D
       │
    Dense(32, ReLU)
       │
    Dense(num_classes, Softmax)
    """
    inp = layers.Input(shape=(timesteps, 1), name="input")

    # Compute step-difference: x[t] - x[t-1]  (shape: batch, T-1, 1)
    # Lamp (ρ≈0.06): large random diffs   → high diff variance
    # Flame (ρ≈0.90): small smooth diffs  → low diff variance
    # This makes the lamp/flame discriminating signal explicit for the LSTM.
    diff    = layers.Lambda(lambda z: z[:, 1:, :] - z[:, :-1, :],
                            name="diff")(inp)
    x_trim  = layers.Lambda(lambda z: z[:, 1:, :],
                            name="trim")(inp)
    x       = layers.Concatenate(axis=-1, name="aug_input")([x_trim, diff])

    x = layers.LSTM(64, return_sequences=True, name="lstm1")(x)
    x = layers.Dropout(0.1, name="dropout1")(x)

    x = layers.LSTM(32, return_sequences=True, name="lstm2")(x)
    x = layers.Dropout(0.1, name="dropout2")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)

    x = layers.Dense(32, activation="relu", name="dense1")(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="LSTM")
    return model


if __name__ == "__main__":
    m = build_lstm(timesteps=100)
    m.summary()
