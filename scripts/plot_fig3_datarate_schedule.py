import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FIG_DIR = ROOT_DIR / "figures" / "plots"
DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 1.5
import matplotlib.pyplot as plt
from plot_style import apply_clean_style, save_figure, tab10


def plot_datarate_schedule(out_path=None):
    if out_path is None:
        out_path = DEFAULT_FIG_DIR / "datarate_step_plot.pdf"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_pts = [0.0, 5.0, 13.3, 21.7, 30.0, 35.0, 43.3, 51.7, 60.0]
    dr_pts = [5, 4, 3, 2, 2, 2, 3, 4, 5]

    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.step(t_pts, dr_pts, where="post", color=tab10(1), linewidth=2.4, zorder=3)
    ax.set_xlim(0, 60)
    ax.set_ylim(1.5, 5.5)
    ax.set_xticks(t_pts)
    ax.set_yticks([2, 3, 4, 5])
    ax.set_yticklabels(["DR2 (SF10)", "DR3 (SF9)", "DR4 (SF8)", "DR5 (SF7)"], fontsize=11)
    ax.set_xlabel("Time (minutes)", fontsize=11)
    ax.set_ylabel("LoRaWAN Data Rate", fontsize=11)
    apply_clean_style(ax, grid_axis="y")
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    plot_datarate_schedule(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
