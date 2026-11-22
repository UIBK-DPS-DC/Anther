"""
Plot how often the dynamic algorithm selects class A vs class C,
one small bar-chart subplot per CSV file, all in a single row.

Usage:
    python plot_class_selection.py

Edit CSV_FILES / LABELS below to point at your 5 files.
"""

import matplotlib.pyplot as plt
import pandas as pd

# ---- Configuration -----------------------------------------------------

CSV_FILES = [
    "dynamic_1KiB_data.csv",
    "dynamic_2KiB_data.csv",
    "dynamic_5KiB_data.csv",
    "dynamic_10KiB_data.csv",
    "dynamic_50KiB_data.csv",
]

# Labels shown under each subplot (e.g. scenario/run names). Must match
# CSV_FILES in length and order.
LABELS = ["1 KiB", "2 KiB", "5 KiB", "10 KiB", "50 KiB"]

COLUMN = "best_class"     # column holding the selected class per row
CLASSES = ["A", "C"]      # bars to show, in this order
COLORS = {"A": "#4C72B0", "C": "#DD8452"}

FONT_SIZE = 13
OUTFILE = "class_selection_counts.pdf"  # vector format recommended for papers

# -------------------------------------------------------------------------

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
fig.savefig(OUTFILE, dpi=300, bbox_inches="tight")
print(f"Saved figure to {OUTFILE}")