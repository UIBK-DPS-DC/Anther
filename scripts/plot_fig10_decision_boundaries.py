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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from anther_model import DR_PARAMS, PING_PERIODICITY_K, evaluate_classes, geometric_utility
from plot_style import save_figure, tab10

CLASS_CMAP = ListedColormap(["#fed8b1", tab10(1), tab10(0)])


def plot_decision_boundaries(out_path=None, ping_periodicity_k=PING_PERIODICITY_K):
    if out_path is None:
        out_path = DEFAULT_FIG_DIR / "plot_decision_boundary_gradient_all_dr.pdf"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bytes_grid = np.logspace(1.3, 4.7, 300)
    lambdas = np.linspace(0.0, 1.0, 250)
    B, L = np.meshgrid(bytes_grid, lambdas)

    fig, axes = plt.subplots(1, 4, figsize=(14.9, 3.5), sharey=True)
    for idx, dr in enumerate([2, 3, 4, 5]):
        winner_grid = np.zeros_like(B, dtype=int)
        for j, n_b in enumerate(bytes_grid):
            evals = evaluate_classes(
                n_b, dr, ping_periodicity_k=ping_periodicity_k, candidate_classes=("A", "B", "C")
            )
            u_a = geometric_utility(*evals["A"], lambdas)
            u_b = geometric_utility(*evals["B"], lambdas)
            u_c = geometric_utility(*evals["C"], lambdas)
            stacked = np.vstack([u_a, u_b, u_c])
            winner_grid[:, j] = np.argmax(stacked, axis=0)

        ax = axes[idx]
        ax.pcolormesh(B, L, winner_grid, cmap=CLASS_CMAP, shading="auto", vmin=-0.5, vmax=2.5)
        ax.contour(B, L, winner_grid, levels=[0.5, 1.5], colors="black", linewidths=1.4)
        ax.set_xscale("log")
        ax.set_title(f"$DR = {dr}$ ($SF{DR_PARAMS[dr]['SF']}$)", fontsize=11, pad=4)
        ax.set_xlabel("Remaining bytes", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=11)
        if idx == 0:
            ax.set_ylabel(r"Sensitivity $\lambda$", fontsize=11)
        ax.grid(False)

    t_ping = (2 ** ping_periodicity_k) * 0.96
    legend_handles = [
        Patch(facecolor="#fed8b1", edgecolor="none", label="Class A (Uplink-Driven)"),
        Patch(facecolor=tab10(1), edgecolor="none", label=rf"Class B (Ping Slots, $T_{{\mathrm{{ping}}}}={t_ping:.2f}\,\mathrm{{s}}$)"),
        Patch(facecolor=tab10(0), edgecolor="none", label="Class C (Continuous RX2)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.18),
    )
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    plot_decision_boundaries(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
