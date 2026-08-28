"""
models/cnn_model.py
-------------------
1D CNN for photocurrent signal classification.

Input shape : (batch, timesteps, 1)
Output      : softmax over 3 classes {dark, UVC lamp, methanol flame}
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(timesteps: int, num_classes: int = 3) -> tf.keras.Model:
    """
    Architecture
    ------------
    Input  (timesteps, 1)
       │
    Conv1D(32, kernel=5, ReLU)   detect coarse local patterns
       │
    MaxPooling1D(2)
       │
    Conv1D(64, kernel=3, ReLU)   detect finer patterns
       │
    GlobalAveragePooling1D        collapse time axis
       │
    Dense(64, ReLU)
       │
    Dropout(0.3)
       │
    Dense(num_classes, Softmax)
    """
    inp = layers.Input(shape=(timesteps, 1), name="input")

    x = layers.Conv1D(32, kernel_size=5, padding="same",
                      activation="relu", name="conv1")(inp)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = layers.Conv1D(64, kernel_size=3, padding="same",
                      activation="relu", name="conv2")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="1D_CNN")
    return model


if __name__ == "__main__":
    m = build_cnn(timesteps=100)
    m.summary()
