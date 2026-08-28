"""
visualize_features.py
---------------------
Extract penultimate-layer (dense1) feature vectors from trained CNN and LSTM,
reduce to 2D with PCA and t-SNE, and save scatter plots to exports/.

Usage:
    python visualize_features.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from models.cnn_model import build_cnn
from models.lstm_model import build_lstm

PROCESSED_DIR = "data/processed"
WEIGHTS_DIR   = "weights"
EXPORTS_DIR   = "exports"

CLASS_NAMES  = ["Dark", "UVC Lamp", "Flame"]
CLASS_COLORS = ["#2196F3", "#4CAF50", "#F44336"]   # blue, green, red


def load_all_data(processed_dir: str):
    """Concatenate train, val, and test splits for richest visualization."""
    X = np.concatenate([
        np.load(os.path.join(processed_dir, "X_train.npy")),
        np.load(os.path.join(processed_dir, "X_val.npy")),
        np.load(os.path.join(processed_dir, "X_test.npy")),
    ], axis=0)
    y = np.concatenate([
        np.load(os.path.join(processed_dir, "y_train.npy")),
        np.load(os.path.join(processed_dir, "y_val.npy")),
        np.load(os.path.join(processed_dir, "y_test.npy")),
    ], axis=0)
    return X, y


def build_feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """Return a sub-model that outputs the penultimate Dense layer ('dense1')."""
    penultimate = model.get_layer("dense1")
    return tf.keras.Model(
        inputs=model.input,
        outputs=penultimate.output,
        name=f"{model.name}_features"
    )


def reduce_pca(features: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=42).fit_transform(features)


def reduce_tsne(features: np.ndarray, n_samples: int) -> np.ndarray:
    perplexity = min(30, max(5, n_samples // 4))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000,
        init="pca",
    ).fit_transform(features)


def scatter_2d(coords: np.ndarray, labels: np.ndarray, title: str, ax):
    """Draw a colored scatter plot on the given axes."""
    legend_patches = []
    for cls_idx, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        mask = labels == cls_idx
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=name,
            alpha=0.8, edgecolors="k", linewidths=0.3, s=55,
        )
        legend_patches.append(mpatches.Patch(color=color, label=name))
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Component 1", fontsize=8)
    ax.set_ylabel("Component 2", fontsize=8)
    ax.legend(handles=legend_patches, fontsize=8, loc="best")
    ax.tick_params(labelsize=7)


def save_single(coords, labels, title, fname):
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter_2d(coords, labels, title, ax)
    plt.tight_layout()
    out = os.path.join(EXPORTS_DIR, fname)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    print("Loading data...")
    X_all, y_all = load_all_data(PROCESSED_DIR)
    n_samples  = len(X_all)
    timesteps  = X_all.shape[1]
    print(f"  Total windows: {n_samples}, shape: {X_all.shape}")

    # ── CNN feature extraction ──────────────────────────────────────────
    print("\nExtracting CNN features...")
    cnn_model = build_cnn(timesteps)
    cnn_model.load_weights(os.path.join(WEIGHTS_DIR, "cnn_best.h5"))
    cnn_extractor = build_feature_extractor(cnn_model)
    cnn_features = cnn_extractor.predict(X_all, verbose=0)   # (N, 64)
    print(f"  CNN feature shape: {cnn_features.shape}")

    # ── LSTM feature extraction ─────────────────────────────────────────
    print("\nExtracting LSTM features...")
    lstm_model = build_lstm(timesteps)
    lstm_model.load_weights(os.path.join(WEIGHTS_DIR, "lstm_best.h5"))
    lstm_extractor = build_feature_extractor(lstm_model)
    lstm_features = lstm_extractor.predict(X_all, verbose=0)  # (N, 32)
    print(f"  LSTM feature shape: {lstm_features.shape}")

    # ── Dimensionality reduction ────────────────────────────────────────
    print("\nReducing dimensions...")
    print("  CNN  — PCA ...")
    cnn_pca  = reduce_pca(cnn_features)
    print("  CNN  — t-SNE ...")
    cnn_tsne = reduce_tsne(cnn_features, n_samples)
    print("  LSTM — PCA ...")
    lstm_pca  = reduce_pca(lstm_features)
    print("  LSTM — t-SNE ...")
    lstm_tsne = reduce_tsne(lstm_features, n_samples)

    # ── Individual scatter plots ────────────────────────────────────────
    print("\nSaving individual plots...")
    save_single(cnn_pca,   y_all, "CNN Feature Space — PCA",   "features_cnn_pca.png")
    save_single(cnn_tsne,  y_all, "CNN Feature Space — t-SNE", "features_cnn_tsne.png")
    save_single(lstm_pca,  y_all, "LSTM Feature Space — PCA",  "features_lstm_pca.png")
    save_single(lstm_tsne, y_all, "LSTM Feature Space — t-SNE","features_lstm_tsne.png")

    # ── Combined 2×2 figure ─────────────────────────────────────────────
    print("\nSaving combined 2×2 figure...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    scatter_2d(cnn_pca,   y_all, "CNN — PCA",    axes[0, 0])
    scatter_2d(cnn_tsne,  y_all, "CNN — t-SNE",  axes[0, 1])
    scatter_2d(lstm_pca,  y_all, "LSTM — PCA",   axes[1, 0])
    scatter_2d(lstm_tsne, y_all, "LSTM — t-SNE", axes[1, 1])
    fig.suptitle(
        "Feature Vector Visualization — Penultimate Layer (Dense)\n"
        "Dark (blue)  |  UVC Lamp (green)  |  Methanol Flame (red)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(EXPORTS_DIR, "features_combined.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    print("\nDone. All feature visualizations saved to exports/")


if __name__ == "__main__":
    main()
