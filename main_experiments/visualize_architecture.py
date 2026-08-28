"""
visualize_architecture.py
-------------------------
Draw matplotlib diagrams of the 1D CNN and LSTM model architectures.
Saves arch_cnn.png, arch_lstm.png, and arch_combined.png to exports/.

Usage:
    python visualize_architecture.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EXPORTS_DIR = "exports"

# ── Color palette ──────────────────────────────────────────────────────────
COLORS = {
    "input":   "#AED6F1",   # light blue
    "conv":    "#A9DFBF",   # light green
    "pool":    "#82E0AA",   # medium green
    "lstm":    "#F9E79F",   # light yellow
    "dense":   "#D7BDE2",   # light purple
    "dropout": "#FAD7A0",   # light orange
    "output":  "#F1948A",   # salmon
}

# ── Layer definitions ──────────────────────────────────────────────────────
CNN_LAYERS = [
    ("Input",                    "(batch, 100, 1)",  COLORS["input"]),
    ("Conv1D(32, k=5)\nReLU",    "(batch, 100, 32)", COLORS["conv"]),
    ("MaxPooling1D(2)",           "(batch,  50, 32)", COLORS["pool"]),
    ("Conv1D(64, k=3)\nReLU",    "(batch,  50, 64)", COLORS["conv"]),
    ("GlobalAvgPooling1D",        "(batch, 64)",      COLORS["pool"]),
    ("Dense(64)\nReLU",           "(batch, 64)",      COLORS["dense"]),
    ("Dropout(0.3)",              "(batch, 64)",      COLORS["dropout"]),
    ("Dense(3)\nSoftmax",         "(batch,  3)",      COLORS["output"]),
]

LSTM_LAYERS = [
    ("Input",                        "(batch, 100,  1)", COLORS["input"]),
    ("LSTM(64)\nreturn_seq=True",     "(batch, 100, 64)", COLORS["lstm"]),
    ("Dropout(0.3)",                  "(batch, 100, 64)", COLORS["dropout"]),
    ("LSTM(32)\nreturn_seq=False",    "(batch,  32)",     COLORS["lstm"]),
    ("Dropout(0.3)",                  "(batch,  32)",     COLORS["dropout"]),
    ("Dense(32)\nReLU",               "(batch,  32)",     COLORS["dense"]),
    ("Dense(3)\nSoftmax",             "(batch,   3)",     COLORS["output"]),
]

# ── Param counts per layer (for annotation) ───────────────────────────────
CNN_PARAMS  = [0, 192, 0, 6208, 0, 4160, 0, 195]
LSTM_PARAMS = [0, 16896, 0, 12416, 0, 1056, 99]


def draw_architecture(ax, layers, params, title,
                      box_w=0.70, box_h=0.52, gap=0.28):
    """
    Draw a vertical stack of labeled layer boxes with arrows on ax.

    Parameters
    ----------
    ax      : matplotlib Axes (axis('off') should already be called)
    layers  : list of (label, shape_str, color)
    params  : list of int param counts, same length as layers
    title   : diagram title
    """
    n = len(layers)
    total_height = n * (box_h + gap) + gap
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_height)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    x_center = 0.5

    for i, ((label, shape, color), n_params) in enumerate(zip(layers, params)):
        # y increases downward by reversing: top layer at highest y
        y_center = total_height - gap - box_h / 2 - i * (box_h + gap)

        # Box
        box = mpatches.FancyBboxPatch(
            (x_center - box_w / 2, y_center - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="#555555", linewidth=1.2,
            zorder=2,
        )
        ax.add_patch(box)

        # Layer name (bold)
        ax.text(x_center, y_center + 0.09, label,
                ha="center", va="center",
                fontsize=7.5, fontweight="bold", zorder=3)

        # Output shape (italic, smaller)
        ax.text(x_center, y_center - 0.09, shape,
                ha="center", va="center",
                fontsize=6.5, style="italic", color="#333333", zorder=3)

        # Param count badge (right side)
        if n_params > 0:
            ax.text(x_center + box_w / 2 + 0.03, y_center,
                    f"{n_params:,} params",
                    ha="left", va="center",
                    fontsize=6, color="#666666", zorder=3)

        # Arrow to next layer
        if i < n - 1:
            arrow_y_start = y_center - box_h / 2
            arrow_y_end   = y_center - box_h / 2 - gap
            ax.annotate(
                "", xy=(x_center, arrow_y_end),
                xytext=(x_center, arrow_y_start),
                arrowprops=dict(arrowstyle="-|>", color="#333333",
                                lw=1.4, mutation_scale=12),
                zorder=1,
            )

    # Total param label at bottom
    total = sum(params)
    ax.text(x_center, 0.06,
            f"Total parameters: {total:,}",
            ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#222222")


def main():
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    # ── Individual CNN diagram ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 10))
    draw_architecture(ax, CNN_LAYERS, CNN_PARAMS, "1D CNN Architecture")
    plt.tight_layout()
    out = os.path.join(EXPORTS_DIR, "arch_cnn.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── Individual LSTM diagram ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 9))
    draw_architecture(ax, LSTM_LAYERS, LSTM_PARAMS, "LSTM Architecture")
    plt.tight_layout()
    out = os.path.join(EXPORTS_DIR, "arch_lstm.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── Combined side-by-side diagram ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    draw_architecture(ax1, CNN_LAYERS,  CNN_PARAMS,  "1D CNN Architecture")
    draw_architecture(ax2, LSTM_LAYERS, LSTM_PARAMS, "LSTM Architecture")
    fig.suptitle("Model Architectures — 1D CNN vs LSTM",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    out = os.path.join(EXPORTS_DIR, "arch_combined.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    print("\nDone. All architecture diagrams saved to exports/")


if __name__ == "__main__":
    main()
