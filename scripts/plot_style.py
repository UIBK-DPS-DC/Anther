import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 1.5

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.labelpad": 6,
    "axes.titlepad": 8,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
    "legend.borderaxespad": 0,
    "legend.labelspacing": 0.5,
    "legend.columnspacing": 1.5,
    "savefig.pad_inches": 0.12,
})

tab10 = plt.get_cmap("tab10")
PALETTE = [
    tab10(0),
    tab10(1),
    tab10(2),
    tab10(3),
    tab10(4),
    "#fed8b1",
]


def apply_clean_style(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.25)
        ax.set_axisbelow(True)


def save_figure(fig, out_path):
    """Apply shared typography, padding, and legend spacing before export."""
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(matplotlib.rcParams["axes.labelsize"])
        ax.yaxis.label.set_fontsize(matplotlib.rcParams["axes.labelsize"])
        ax.set_title(ax.get_title(), fontsize=matplotlib.rcParams["axes.titlesize"],
                     pad=matplotlib.rcParams["axes.titlepad"])
        ax.tick_params(axis="both", which="both", labelsize=12)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(matplotlib.rcParams["legend.fontsize"])
        legend.get_title().set_fontsize(matplotlib.rcParams["legend.title_fontsize"])
    fig.tight_layout(pad=1.0, w_pad=1.8, h_pad=1.8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = Bbox.union([
        ax.get_tightbbox(renderer) for ax in fig.axes if ax.get_visible()
    ])
    # Anchor the top of every legend 10 points below the lowest axis label,
    # independent of figure height, tick labels, and the number of legend rows.
    legend_top = (axes_bbox.y0 - 10 * fig.dpi / 72) / fig.bbox.height
    for legend in fig.legends:
        center = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).x0
        legend.set_bbox_to_anchor((center, legend_top), transform=fig.transFigure)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
