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


def plot_estimation_vs_measurement(out_path=None):
    if out_path is None:
        out_path = DEFAULT_FIG_DIR / "estimation_vs_measurement.pdf"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sizes_kib = np.array([1, 2, 5, 10, 50])

    t_a_meas = np.array([309.82, 617.49, 1482.87, 2904.99, 14655.00])
    t_a_est  = np.array([373.95, 683.91, 1551.80, 2977.63, 14508.23])
    t_a_err  = np.array([20.70,  10.76,  4.65,    2.50,    1.00])

    t_c_meas = np.array([98.87,  207.40, 515.34,  1033.80, 5194.99])
    t_c_est  = np.array([104.99, 209.21, 524.99,  1043.95, 5222.91])
    t_c_err  = np.array([6.19,   0.87,   1.87,    0.98,    0.54])

    e_a_meas = np.array([4.02,   7.43,   18.05,   35.58,   181.46])
    e_a_est  = np.array([5.30,   9.09,   19.70,   37.13,   178.06])
    e_a_err  = np.array([32.05,  22.44,  9.13,    4.36,    1.87])

    e_c_meas = np.array([38.71,  82.51,  206.58,  415.06,  2075.99])
    e_c_est  = np.array([42.23,  84.15,  211.04,  419.66,  2099.33])
    e_c_err  = np.array([9.08,   1.99,   2.16,    1.11,    1.12])

    fig, (ax1, ax1_err, ax2, ax2_err) = plt.subplots(
        1, 4, figsize=(17.6, 4.0), sharex=True,
    )

    ax1.loglog(sizes_kib, e_a_est,  "o--", color=tab10(3), label="Class A estimate", zorder=3)
    ax1.loglog(sizes_kib, e_a_meas, "o-",  color=tab10(3), label="Class A measured", zorder=3)
    ax1.loglog(sizes_kib, e_c_est,  "s--", color=tab10(0), label="Class C estimate", zorder=3)
    ax1.loglog(sizes_kib, e_c_meas, "s-",  color=tab10(0), label="Class C measured", zorder=3)
    ax1.set_title("Energy")
    ax1.set_ylabel("Energy (J)")

    ax1_err.semilogx(sizes_kib, e_a_err, "o:", color=tab10(3), alpha=0.8, label="Class A rel. error")
    ax1_err.semilogx(sizes_kib, e_c_err, "s:", color=tab10(0), alpha=0.8, label="Class C rel. error")
    ax1_err.set_title("Energy error")
    ax1_err.set_ylabel("Relative Error (%)")
    ax1_err.set_ylim(0, 35)

    ax2.loglog(sizes_kib, t_a_est,  "o--", color=tab10(3), label="Class A estimate", zorder=3)
    ax2.loglog(sizes_kib, t_a_meas, "o-",  color=tab10(3), label="Class A measured", zorder=3)
    ax2.loglog(sizes_kib, t_c_est,  "s--", color=tab10(0), label="Class C estimate", zorder=3)
    ax2.loglog(sizes_kib, t_c_meas, "s-",  color=tab10(0), label="Class C measured", zorder=3)
    ax2.set_title("Completion time")
    ax2.set_ylabel("Time (s)")

    ax2_err.semilogx(sizes_kib, t_a_err, "o:", color=tab10(3), alpha=0.8, label="Class A rel. error")
    ax2_err.semilogx(sizes_kib, t_c_err, "s:", color=tab10(0), alpha=0.8, label="Class C rel. error")
    ax2_err.set_title("Time error")
    ax2_err.set_ylabel("Relative Error (%)")
    ax2_err.set_ylim(0, 35)

    for ax in (ax1, ax1_err, ax2, ax2_err):
        ax.set_xticks(sizes_kib)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("File Size (KiB)")
        apply_clean_style(ax, grid_axis="y")

    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2_err.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2, labels1 + labels2,
        loc="upper center", ncol=6, frameon=False,
        bbox_to_anchor=(0.5, 0),
    )

    save_figure(fig, out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    plot_estimation_vs_measurement(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
