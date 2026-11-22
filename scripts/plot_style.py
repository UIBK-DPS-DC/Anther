import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 1.5

import matplotlib.pyplot as plt
import os
import re
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

# Physical PDF widths, matching the usual IEEEtran column/text widths. Export at
# these widths in LaTeX: font sizes then remain identical rather than being scaled
# differently for each figure by \includegraphics.
IEEE_COLUMN_WIDTH = float(os.environ.get("ANTHER_COLUMN_WIDTH_IN", "3.5"))
IEEE_TEXT_WIDTH = float(os.environ.get("ANTHER_TEXT_WIDTH_IN", "7.16"))
FONT_SIZE = 5.5
LEGEND_GAP_PT = 3

matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelpad": 2,
    "axes.titlepad": 3,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "legend.title_fontsize": FONT_SIZE,
    "legend.borderaxespad": 0,
    "legend.labelspacing": 0.2,
    "legend.columnspacing": 0.8,
    "legend.handlelength": 1.4,
    "legend.handleheight": 0.55,
    "legend.borderpad": 0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
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
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.25)
        ax.set_axisbelow(True)


def _wrap_text(fig, text, width_points, renderer):
    """Wrap by rendered width; keep math expressions intact."""
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in re.findall(r"(?:\$[^$]*\$|[^\s$])+", paragraph):
            candidate = f"{line} {word}" if line else word
            probe = Text(text=candidate, fontsize=FONT_SIZE, figure=fig)
            width = probe.get_window_extent(renderer).width * 72 / fig.dpi
            if line and width > width_points:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return "\n".join(lines)


def _style_legend_markers(legend):
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = legend.legendHandles
    for handle in handles:
        if isinstance(handle, Line2D):
            handle.set_markersize(min(handle.get_markersize(), 3.5))
            handle.set_markeredgewidth(min(handle.get_markeredgewidth(), 0.5))
            handle.set_linewidth(min(handle.get_linewidth(), 1.0))
        elif isinstance(handle, PathCollection):
            handle.set_sizes(handle.get_sizes().clip(max=12.25))
            handle.set_linewidths(handle.get_linewidths().clip(max=0.5))


def save_figure(fig, out_path, *, column_fraction=1.0):
    """Export exact IEEE widths with 5.5-pt type and compact, wide plot areas.

    One/two-panel figures occupy one column; four-panel figures occupy two.
    column_fraction=0.475 exports each of Figure 9's side-by-side subfigures
    at its actual final size, retaining the same 5.5-pt print typography.
    A fixed-width export box is essential: bbox_inches='tight' would change the
    PDF width, and LaTeX would subsequently scale its fonts by a different ratio.
    """
    columns = max(ax.get_subplotspec().get_gridspec().ncols for ax in fig.axes)
    if not 0 < column_fraction <= 1:
        raise ValueError("Column fraction must be in (0, 1]")
    width = (IEEE_TEXT_WIDTH if columns == 4 else IEEE_COLUMN_WIDTH) * column_fraction
    height = 1.25 if column_fraction < 1 else {1: 1.6, 2: 1.55, 4: 1.65}.get(columns, 1.6)
    fig.set_size_inches(width, height, forward=True)
    for text in fig.findobj(Text):
        text.set_fontsize(FONT_SIZE)
    notes = []
    for ax in fig.axes:
        ax.set_title(ax.get_title(), fontsize=FONT_SIZE, pad=3)
        ax.tick_params(axis="both", which="both", labelsize=FONT_SIZE)
        ax.xaxis.labelpad = ax.yaxis.labelpad = 2
        for line in ax.lines:
            line.set_markersize(3.5)
            line.set_linewidth(min(line.get_linewidth(), 1.3))
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_sizes(collection.get_sizes().clip(max=20))
        if ax.get_legend() is not None:
            _style_legend_markers(ax.get_legend())
        # At column width, boxed recording-scope notes would hide substantial
        # portions of the curves. Keep them visible below the chart instead.
        for text in list(ax.texts):
            if text.get_transform() == ax.transAxes and text.get_bbox_patch() is not None:
                notes.append(text.get_text())
                text.remove()
    # Reserve a sensible title width before tight_layout sees long original
    # headings. Fine-tune annotations against the actual axes widths below.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        title_width = (width * 72 - 44) / columns
        ax.set_title(_wrap_text(fig, ax.get_title(), title_width, renderer), pad=3)
    fig.tight_layout(pad=0.4, w_pad=0.7, h_pad=0.5)
    fig.canvas.draw()
    for ax in fig.axes:
        label_height = ax.get_window_extent(renderer).height * 72 / fig.dpi
        ax.yaxis.label.set_text(_wrap_text(fig, ax.get_ylabel(), label_height, renderer))
        annotation_width = ax.get_window_extent(renderer).width * 72 / fig.dpi * 0.94
        for text in ax.texts:
            text.set_text(_wrap_text(fig, text.get_text(), annotation_width, renderer))
        ticks = [t for t in ax.get_xticklabels() if t.get_visible()]
        boxes = [t.get_window_extent(renderer) for t in ticks]
        if any(left.x1 + 2 > right.x0 for left, right in zip(boxes, boxes[1:])):
            plt.setp(ticks, rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout(pad=0.4, w_pad=0.7, h_pad=0.5)
    fig.canvas.draw()

    # Rebuild legends at the final type size so handle lengths and row/column
    # spacing use the same units as their text. Fit each legend to the column,
    # and stack multiple groups (e.g. methods/architectures) without overlap.
    legends = []
    for old in list(fig.legends):
        handles = getattr(old, "legend_handles", None)
        if handles is None:
            handles = old.legendHandles
        labels = [t.get_text() for t in old.get_texts()]
        title = old.get_title().get_text()
        ncol = old._ncols
        old.remove()
        label_width = min(90 if columns < 4 else 125, width * 72 - 30)
        labels = [_wrap_text(fig, label, label_width, renderer) for label in labels]
        while True:
            legend = fig.legend(
                handles, labels, title=title or None, loc="upper center",
                ncol=ncol, frameon=False, fontsize=FONT_SIZE,
                title_fontsize=FONT_SIZE, bbox_to_anchor=(0.5, 0),
                borderaxespad=0,
            )
            _style_legend_markers(legend)
            fits = legend.get_window_extent(renderer).width <= (width - 0.08) * fig.dpi
            balanced = ncol == len(labels) or ncol <= (len(labels) + 1) // 2
            if (fits and balanced) or ncol == 1:
                break
            legend.remove()
            ncol -= 1
        legends.append(legend)

    axes_bbox = Bbox.union([
        ax.get_tightbbox(renderer) for ax in fig.axes if ax.get_visible()
    ])
    bottom = axes_bbox.y0
    for note in notes:
        top = bottom - LEGEND_GAP_PT * fig.dpi / 72
        text = fig.text(
            0.5, top / fig.bbox.height,
            _wrap_text(fig, note, width * 72 - 12, renderer),
            ha="center", va="top", fontsize=FONT_SIZE,
        )
        bottom = text.get_window_extent(renderer).y0
    for legend in legends:
        top = bottom - LEGEND_GAP_PT * fig.dpi / 72
        legend.set_bbox_to_anchor((0.5, top / fig.bbox.height), transform=fig.transFigure)
        bottom = legend.get_window_extent(renderer).y0
    padding = 3 / 72
    export_box = Bbox.from_extents(
        0, bottom / fig.dpi - padding,
        width, axes_bbox.y1 / fig.dpi + padding,
    )
    fig.savefig(out_path, dpi=300, bbox_inches=export_box, pad_inches=0)
