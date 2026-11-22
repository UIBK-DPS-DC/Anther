"""
Plot how often the dynamic algorithm selects class A vs class C,
one small bar-chart subplot per CSV file, all in a single row.

Usage:
    python plot_class_selection.py

Edit CSV_FILES / LABELS below to point at your 5 files.
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys

# ---- Configuration -----------------------------------------------------
SIZES = [1, 2, 5, 10, 50]

CSV_FILES = [f"dynamic_{size}KiB_data.csv" for size in SIZES]

# Labels shown under each subplot (e.g. scenario/run names). Must match
# CSV_FILES in length and order.
LABELS = [f"{size} KiB" for size in SIZES]

COLUMN = "best_class"     # column holding the selected class per row
CLASSES = ["A", "C"]      # bars to show, in this order
COLORS = {"A": "#4C72B0", "C": "#DD8452"}

FONT_SIZE = 13

# -------------------------------------------------------------------------

def plot_class_selection(output_file = None):

    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
    })

    n = len(CSV_FILES)
    fig, axes = plt.subplots(
        1, n,
        figsize=(1.6 * n, 1.8),  # tiny, wide row suitable for a paper column/page
        sharey=True,
    )

    if n == 1:
        axes = [axes]

    for ax, path, label in zip(axes, CSV_FILES, LABELS):
        df = pd.read_csv(path)
        counts = df[COLUMN].value_counts()
        values = [counts.get(c, 0) for c in CLASSES]

        ax.bar(CLASSES, values, color=[COLORS[c] for c in CLASSES], width=0.6)
        ax.set_title(label, fontsize=FONT_SIZE)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, fontsize=FONT_SIZE)
        ax.tick_params(axis="y", labelsize=FONT_SIZE)
        ax.set_yscale("log")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Count", fontsize=FONT_SIZE)

    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {output_file}")


def main():
    if len(sys.argv) == 1:
        output_file = None
    elif len(sys.argv) == 2:
        output_file = sys.argv[1]
    else:
        print("Usage: python plot_class_count.py [output.pdf]")
        sys.exit(1)

    plot_class_selection(output_file)


if __name__ == "__main__":
    main()