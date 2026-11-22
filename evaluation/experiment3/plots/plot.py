import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import sys


def plot(input_file, output_file = None):
    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    df = pd.read_csv(input_file)

    # Convert timestamp to timedelta
    df["time"] = df["timestamp"] - df["timestamp"][0]


    # ------------------------------------------------------------
    # Utility axis transformation
    # ------------------------------------------------------------

    def utility_forward(x):
        """
        Nonlinear utility scale centered around 0.5.

        Values around 0.5 are expanded.
        Values near 0 and 1 are compressed.
        """
        return (
            0.5
            + 0.5
            * np.sign(x - 0.5)
            * np.abs(2 * x - 1) ** 0.5
        )


    def utility_inverse(y):
        """
        Inverse of utility_forward().
        """
        return (
            0.5
            + 0.5
            * np.sign(y - 0.5)
            * np.abs(2 * y - 1) ** 2
        )


    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(12, 6))


    # ------------------------------------------------------------
    # Utility (left axis)
    # ------------------------------------------------------------

    ax1.plot(
        df["time"],
        df["utility_A"],
        marker="o",
        label="Utility A"
    )

    ax1.plot(
        df["time"],
        df["utility_C"],
        marker="o",
        label="Utility C"
    )

    ax1.set_yscale(
        "function",
        functions=(utility_forward, utility_inverse)
    )

    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Utility")

    ax1.grid(True, alpha=0.3)


    # Datarate (right axis)
    ax2 = ax1.twinx()

    def datarate_to_plot_y(dr):
        return 0.25 + (dr - 2) / 3 * 0.5


    def datarate_to_utility_y(dr):
        return datarate_to_plot_y(dr)


    ax2.plot(
        df["time"],
        datarate_to_utility_y(df["datarate"]),
        marker="s",
        linestyle="--",
        label="Planned datarate",
        color="pink"
    )

    ax2.plot(
        df["time"],
        datarate_to_utility_y(df["used datarate"]),
        marker="s",
        linestyle="--",
        label="Used datarate",
        color="purple"
    )

    ax2.set_ylim(0, 1)

    ax2.set_yticks([
        0.25,
        datarate_to_plot_y(3),
        datarate_to_plot_y(4),
        0.75
    ])

    ax2.set_yticklabels([
        "2",
        "3",
        "4",
        "5"
    ])

    ax2.set_ylabel("Datarate")


    # Combined legend

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best"
    )


    # ------------------------------------------------------------
    # Final formatting
    # ------------------------------------------------------------

    ax1.set_xlabel("Time")

    fig.autofmt_xdate()

    plt.title("Utility and Datarate over Time")

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


def main():
    if len(sys.argv) == 2:
        output_file = None
    elif len(sys.argv) == 3:
        output_file = sys.argv[2]
    else:
        print("Usage: python plot.py input.csv [output.pdf]")
        sys.exit(1)

    input_file = sys.argv[1]

    plot(input_file, output_file)


if __name__ == "__main__":
    main()