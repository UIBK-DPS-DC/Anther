import json
import matplotlib.pyplot as plt

# This script was generated using ChatGPT on the 03.09.2026 and adjusted to fit the needs for the thesis. The prompt was this:
# Make two python plots using the data above (Time (s) and Energy (J)). They should both have two y axes. The the left y axis should contain the actual values of Energy/Latency. The right one should show the relative error between the measurements and the estimates.

# ============================================================
# Data
# ============================================================

with open("data.json", "r") as f:
    data = json.load(f)


# ============================================================
# Helper function
# ============================================================

def plot_metric(metric, ylabel, title):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot each class
    for cls, c in [("A", "blue"), ("C", "orange")]:
        class_data = [x for x in data if x["class"] == cls]

        file_sizes = [
            x["file_size"] for x in class_data
        ]

        measured = [
            x[metric]["measured"] for x in class_data
        ]

        estimates = [
            x[metric]["estimate"] for x in class_data
        ]

        relative_errors = [
            (abs(x[metric]["measured"] - x[metric]["estimate"]) / x[metric]["measured"]) for x in class_data
        ]

        # Left axis: measured and estimated values
        ax1.plot(
            file_sizes,
            measured,
            marker="o",
            linewidth=2,
            label=f"Class {cls} measured",
            color=c
        )

        ax1.plot(
            file_sizes,
            estimates,
            marker="o",
            linestyle="--",
            linewidth=1.5,
            label=f"Class {cls} estimate",
            color=c
        )

        # Right axis: relative error
        ax2.plot(
            file_sizes,
            relative_errors,
            marker="o",
            linestyle=":",
            linewidth=2,
            label=f"Class {cls} relative error",
            color=c
        )

    # Logarithmic x-axis
    ax1.set_xscale("log")

    # Axis labels
    ax1.set_xlabel("File Size (KiB)", fontsize=25)
    ax1.set_xticks([1024, 2048, 5120, 10240, 51200], labels=["1", "2", "5", "10", "50"], fontsize=18)

    ax1.set_ylabel(ylabel, fontsize=25)
    ax1.set_yscale('log')

    if metric == "energy":
        ticks = [1, 5, 10, 50, 100, 500, 1000, 2000]
    else:
        ticks = [100, 500, 1000, 5000, 10000]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([f"{x}" for x in ticks])
    ax1.tick_params(axis='y', labelsize=15)

    ax2.set_ylabel("Relative Error (%)", fontsize=25)
    if metric == "energy":
        ax2.set_yticks([x/1000 for x in range(0, 350, 50)])
        ax2.set_yticklabels([f"{x/10:.1f}" for x in range(0, 350, 50)])
    else:
        ax2.set_yticks([x/1000 for x in range(0, 225, 25)])
        ax2.set_yticklabels([f"{x/10:.1f}" for x in range(0, 225, 25)])
    ax2.tick_params(axis='y', labelsize=15)

    # Title
    # ax1.set_title(title)

    # Grid
    ax1.grid(
        True,
        which="both",
        alpha=0.25
    )

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    combined_handles = handles1 + handles2
    combined_labels = labels1 + labels2
    zipped_data = zip(combined_handles, combined_labels)
    zipped_data = sorted(zipped_data, key=lambda x: x[1])
    handles, labels = zip(*zipped_data)
        

    ax1.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=18
    )

    fig.tight_layout()

    return fig


# ============================================================
# Time plot
# ============================================================

fig_time = plot_metric(
    metric="time",
    ylabel="Time (s)",
    title="Time: Measured vs. Estimated with Relative Error"
)

fig_time.savefig(
    "time_plot.png",
    dpi=200,
    bbox_inches="tight"
)


# ============================================================
# Energy plot
# ============================================================

fig_energy = plot_metric(
    metric="energy",
    ylabel="Energy (J)",
    title="Energy: Measured vs. Estimated with Relative Error"
)

fig_energy.savefig(
    "energy_plot.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()