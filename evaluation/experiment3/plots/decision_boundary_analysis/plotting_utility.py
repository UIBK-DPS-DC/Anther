import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("utility_grid.csv")

lambdas = np.sort(df["lambda"].unique())
remaining_bytes = np.sort(df["remaining_bytes"].unique())

lambda_grid, bytes_grid = np.meshgrid(
    lambdas,
    remaining_bytes,
)

utility_a = (
    df.pivot(
        index="remaining_bytes",
        columns="lambda",
        values="utility_a",
    )
    .loc[remaining_bytes, lambdas]
    .values
)

utility_c = (
    df.pivot(
        index="remaining_bytes",
        columns="lambda",
        values="utility_c",
    )
    .loc[remaining_bytes, lambdas]
    .values
)


fig = plt.figure(figsize=(16, 7))

# Class A
ax1 = fig.add_subplot(121, projection="3d")

surf_a = ax1.plot_surface(
    bytes_grid,
    lambda_grid,
    utility_a,
    cmap="viridis",
    edgecolor="none",
)

ax1.set_xlabel("Remaining bytes")
ax1.set_ylabel("Lambda")
ax1.set_zlabel("Utility A")
ax1.set_title("Class A Utility")
ax1.set_xscale("log")

fig.colorbar(
    surf_a,
    ax=ax1,
    shrink=0.6,
    pad=0.1,
)


# Class C
ax2 = fig.add_subplot(122, projection="3d")

surf_c = ax2.plot_surface(
    bytes_grid,
    lambda_grid,
    utility_c,
    cmap="plasma",
    edgecolor="none",
)

ax2.set_xlabel("Remaining bytes")
ax2.set_ylabel("Lambda")
ax2.set_zlabel("Utility C")
ax2.set_title("Class C Utility")
ax2.set_xscale("log")

fig.colorbar(
    surf_c,
    ax=ax2,
    shrink=0.6,
    pad=0.1,
)


plt.tight_layout()
plt.show()

plt.savefig("utility_grid.png", dpi=300)