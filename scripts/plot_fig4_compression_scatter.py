import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FIG_DIR = ROOT_DIR / "figures" / "plots"
DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 1.5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull
from plot_style import apply_clean_style, save_figure, tab10

ARCHS = ["mlp", "cnn", "ds_cnn", "gru", "crnn", "cnn1d", "svdf"]
MARKERS = {"cnn": "o", "cnn1d": "s", "crnn": "^", "ds_cnn": "D", "gru": "v", "mlp": "P", "svdf": "X"}
METHODS = ["LoRA rank 2", "LoRA rank 4", "LoRA rank 8", "quant", "bpatch", "full"]
COLORS = {
    "LoRA rank 2": tab10(0),
    "LoRA rank 4": tab10(1),
    "LoRA rank 8": tab10(2),
    "quant":       tab10(5),
    "bpatch":      tab10(3),
    "full":        tab10(4),
}

SIZES_KIB = {
    "svdf":   {"LoRA rank 2": 1.15, "LoRA rank 4": 1.95, "LoRA rank 8": 3.55, "quant": 10.2,  "bpatch": 38.5,  "full": 40.8},
    "gru":    {"LoRA rank 2": 1.55, "LoRA rank 4": 2.65, "LoRA rank 8": 4.85, "quant": 18.2,  "bpatch": 71.0,  "full": 72.8},
    "crnn":   {"LoRA rank 2": 2.45, "LoRA rank 4": 4.20, "LoRA rank 8": 7.60, "quant": 33.5,  "bpatch": 131.0, "full": 134.0},
    "cnn1d":  {"LoRA rank 2": 1.95, "LoRA rank 4": 3.40, "LoRA rank 8": 6.25, "quant": 26.8,  "bpatch": 104.5, "full": 107.2},
    "cnn":    {"LoRA rank 2": 2.60, "LoRA rank 4": 4.75, "LoRA rank 8": 8.85, "quant": 12.6,  "bpatch": 48.2,  "full": 50.4},
    "ds_cnn": {"LoRA rank 2": 2.95, "LoRA rank 4": 4.65, "LoRA rank 8": 7.90, "quant": 10.75, "bpatch": 41.5,  "full": 43.0},
    "mlp":    {"LoRA rank 2": 8.60, "LoRA rank 4": 16.5, "LoRA rank 8": 32.2, "quant": 62.5,  "bpatch": 245.0, "full": 250.0},
}

ACC_WARM = {
    "svdf":   {"LoRA rank 2": 45.5, "LoRA rank 4": 56.0, "LoRA rank 8": 61.5, "quant": 64.0, "bpatch": 64.2, "full": 64.2},
    "gru":    {"LoRA rank 2": 73.0, "LoRA rank 4": 82.0, "LoRA rank 8": 85.0, "quant": 88.5, "bpatch": 88.8, "full": 88.8},
    "crnn":   {"LoRA rank 2": 83.5, "LoRA rank 4": 88.0, "LoRA rank 8": 90.5, "quant": 91.2, "bpatch": 91.4, "full": 91.4},
    "cnn1d":  {"LoRA rank 2": 86.8, "LoRA rank 4": 87.5, "LoRA rank 8": 91.0, "quant": 92.2, "bpatch": 92.3, "full": 92.3},
    "cnn":    {"LoRA rank 2": 75.5, "LoRA rank 4": 74.8, "LoRA rank 8": 77.8, "quant": 73.2, "bpatch": 73.5, "full": 73.5},
    "ds_cnn": {"LoRA rank 2": 89.2, "LoRA rank 4": 89.8, "LoRA rank 8": 90.2, "quant": 90.0, "bpatch": 90.1, "full": 90.1},
    "mlp":    {"LoRA rank 2": 52.5, "LoRA rank 4": 53.0, "LoRA rank 8": 53.5, "quant": 56.2, "bpatch": 56.4, "full": 56.4},
}

ACC_EARLY = {
    "svdf":   {"LoRA rank 2": 36.5, "LoRA rank 4": 44.5, "LoRA rank 8": 49.5, "quant": 64.0, "bpatch": 64.2, "full": 64.2},
    "gru":    {"LoRA rank 2": 50.0, "LoRA rank 4": 64.5, "LoRA rank 8": 71.0, "quant": 88.5, "bpatch": 88.8, "full": 88.8},
    "crnn":   {"LoRA rank 2": 70.5, "LoRA rank 4": 76.5, "LoRA rank 8": 83.0, "quant": 91.2, "bpatch": 91.4, "full": 91.4},
    "cnn1d":  {"LoRA rank 2": 75.5, "LoRA rank 4": 80.0, "LoRA rank 8": 84.8, "quant": 92.2, "bpatch": 92.3, "full": 92.3},
    "cnn":    {"LoRA rank 2": 48.0, "LoRA rank 4": 56.5, "LoRA rank 8": 59.2, "quant": 73.2, "bpatch": 73.5, "full": 73.5},
    "ds_cnn": {"LoRA rank 2": 78.5, "LoRA rank 4": 82.0, "LoRA rank 8": 82.2, "quant": 90.0, "bpatch": 90.1, "full": 90.1},
    "mlp":    {"LoRA rank 2": 45.2, "LoRA rank 4": 46.5, "LoRA rank 8": 48.5, "quant": 56.2, "bpatch": 56.4, "full": 56.4},
}


def draw_panel(ax, acc_dict, title):
    for method in METHODS:
        xs = np.array([SIZES_KIB[a][method] for a in ARCHS])
        ys = np.array([acc_dict[a][method] for a in ARCHS])
        pts = np.column_stack([np.log10(xs), ys])
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            ax.fill(10 ** hull_pts[:, 0], hull_pts[:, 1], color=COLORS[method], alpha=0.14)
        for a in ARCHS:
            ax.scatter(
                SIZES_KIB[a][method], acc_dict[a][method],
                color=COLORS[method], marker=MARKERS[a], s=55,
                edgecolors="black", linewidths=0.6, zorder=4,
            )
    ax.set_xscale("log")
    ax.set_xlim(0.7, 450)
    ax.set_ylim(28, 102)
    ax.set_xlabel("Update size (KiB)", fontsize=11)
    ax.set_title(title, fontsize=11, pad=4)
    apply_clean_style(ax, grid_axis="y")


def plot_scatter(out_path=None):
    if out_path is None:
        out_path = DEFAULT_FIG_DIR / "scatter_plot.pdf"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True)
    draw_panel(ax1, ACC_WARM, "Warm-started Base ([50%, 70%, 80%, 90%, 100%])")
    draw_panel(ax2, ACC_EARLY, "Early-drift Base ([10%, 30%, 50%, 70%, 100%])")
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=11)

    method_handles = [Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS]
    arch_handles = [
        plt.Line2D([0], [0], marker=MARKERS[a], color="w", markerfacecolor="#cccccc",
                   markeredgecolor="black", markersize=7, label=a) for a in ARCHS
    ]

    fig.legend(
        handles=method_handles, title="Update Method", loc="upper center",
        ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.30, -0.18),
    )
    fig.legend(
        handles=arch_handles, title="Architecture", loc="upper center",
        ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.76, -0.18),
    )

    save_figure(fig, out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    plot_scatter(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
