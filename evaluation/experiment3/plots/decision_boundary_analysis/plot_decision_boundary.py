import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


df = pd.read_csv("utility_grid.csv")

# ---------------------------------------------------------
# Downsample ONLY for plotting
# ---------------------------------------------------------

max_bytes_points = 1000
max_lambda_points = 500

bytes_values = np.sort(df["remaining_bytes"].unique())
lambda_values = np.sort(df["lambda"].unique())

bytes_plot = bytes_values[
    np.linspace(
        0,
        len(bytes_values) - 1,
        min(max_bytes_points, len(bytes_values)),
    ).astype(int)
]

lambda_plot = lambda_values[
    np.linspace(
        0,
        len(lambda_values) - 1,
        min(max_lambda_points, len(lambda_values)),
    ).astype(int)
]

plot_df = df[
    df["remaining_bytes"].isin(bytes_plot)
    & df["lambda"].isin(lambda_plot)
].copy()


# ---------------------------------------------------------
# Create decision grid
# ---------------------------------------------------------

pivot = plot_df.pivot(
    index="remaining_bytes",
    columns="lambda",
    values="utility_a",
)

utility_a = pivot.loc[
    bytes_plot,
    lambda_plot,
].values

pivot = plot_df.pivot(
    index="remaining_bytes",
    columns="lambda",
    values="utility_c",
)

utility_c = pivot.loc[
    bytes_plot,
    lambda_plot,
].values


difference = utility_a - utility_c
winner = (difference > 0).astype(int)

lambda_grid, bytes_grid = np.meshgrid(
    lambda_plot,
    bytes_plot,
)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 7))

mesh = ax.pcolormesh(
    bytes_grid,
    lambda_grid,
    winner,
    shading="nearest",
    alpha=0.5,
)

# Decision boundary
ax.contour(
    bytes_grid,
    lambda_grid,
    difference,
    levels=[0],
    colors="black",
    linewidths=2,
)


ax.set_xscale("log")

ax.set_xlabel("Remaining bytes")
ax.set_ylabel("Lambda")

ax.set_title("Class A / Class C Decision Boundary")

ax.grid(
    True,
    which="both",
    alpha=0.2,
)


legend_elements = [
    Patch(
        label="Class A: utility_a > utility_c",
        color=mesh.cmap(1.0),
    ),
    Patch(
        label="Class C: utility_c > utility_a",
        color=mesh.cmap(0.0),
    ),
    Line2D(
        [0],
        [0],
        linewidth=2,
        color="black",
        label="Decision boundary",
    ),
]

ax.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()

plt.savefig(
    "plot_decision_boundary.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()