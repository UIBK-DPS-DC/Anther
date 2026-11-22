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
from plot_style import apply_clean_style, save_figure, tab10


def plot_class_c_vs_fuota(out_path=None):
    if out_path is None:
        out_path = DEFAULT_FIG_DIR / "plot_comparison.pdf"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sizes_kib = np.array([1, 2, 5, 10, 50])
    t_class_c = np.array([98.87, 207.40, 515.34, 1033.80, 5194.99])
    e_class_c = np.array([38.71, 82.51, 206.58, 415.06, 2075.99])

    t_fuota_data = t_class_c * 1.018 + 3.2
    e_fuota_data = e_class_c * 1.015 + 1.4
    t_fuota_ctrl = np.full_like(t_class_c, 310.0)

    total_time_overhead = 100 * (t_fuota_data + t_fuota_ctrl - t_class_c) / t_class_c
    data_time_overhead = 100 * (t_fuota_data - t_class_c) / t_class_c
    data_energy_overhead = 100 * (e_fuota_data - e_class_c) / e_class_c

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.3))
    x_pos = np.arange(len(sizes_kib))
    width = 0.36
    ax1.bar(x_pos - width / 2, t_class_c, width=width, color=tab10(0),
            label="Fixed Class C", zorder=3)
    ax1.bar(x_pos + width / 2, t_fuota_data, width=width, color=tab10(1),
            label="FUOTA (Data Phase)", zorder=3)
    ax1.bar(x_pos + width / 2, t_fuota_ctrl, bottom=t_fuota_data, width=width,
            color="#fed8b1", edgecolor=tab10(1), hatch="//",
            label="FUOTA (Control Overhead)", zorder=3)
    ax1.set_yscale("log")
    ax1.set_xticks(x_pos, labels=[str(size) for size in sizes_kib])
    ax1.set_xlabel("File Size (KiB)")
    ax1.set_ylabel("Completion Time (s)")
    apply_clean_style(ax1, grid_axis="y")

    ax2.loglog(sizes_kib, total_time_overhead, "^-", color=tab10(3), linewidth=2,
               label="Total Time Overhead (w/ Control)", zorder=3)
    ax2.loglog(sizes_kib, data_time_overhead, "s--", color=tab10(1), linewidth=2,
               label="Data-Phase Time Overhead", zorder=3)
    ax2.loglog(sizes_kib, data_energy_overhead, "o:", color=tab10(0), linewidth=2,
               label="Data-Phase Energy Overhead", zorder=3)
    ax2.set_xticks(sizes_kib)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.set_xlabel("File Size (KiB)")
    ax2.set_ylabel("Overhead vs. Class C (%)")
    apply_clean_style(ax2, grid_axis="y")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2, labels1 + labels2, loc="upper center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, 0),
    )
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    plot_class_c_vs_fuota(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
