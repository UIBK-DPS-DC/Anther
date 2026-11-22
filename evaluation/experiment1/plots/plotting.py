import os
import json
import argparse
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODELS = [
    "cnn",
    "cnn1d",
    "crnn",
    "ds_cnn",
    "gru",
    "mlp",
    # "mlp_big",
    "svdf",
]

VERSIONS = [0, 1, 2, 3, 4]

SVD_INFORMATION = [
    "0.7",
    "0.8",
    "0.9",
    "0.95",
    "0.97",
    "0.98",
    "0.99",
    "1",
]


# ============================================================
# File helpers
# ============================================================

def get_file_size_in_kib(file_path):
    """Return file size in KiB."""
    return os.path.getsize(file_path) / 1024

def find_matching_file(directory, prefix):
    """
    Find the first file in 'directory' starting with 'prefix'.

    Returns None if no matching file exists.
    """
    if not os.path.isdir(directory):
        return None

    matching_files = [
        f
        for f in os.listdir(directory)
        if f.startswith(prefix)
    ]

    if not matching_files:
        return None

    matching_files.sort()

    return os.path.join(directory, matching_files[0])


# ============================================================
# Full updates
# ============================================================

def get_full_update_sizes(model):
    """
    Get the full model update size for every version transition.

    Version 0 is skipped because the original script treats it
    as the initial model rather than an update.
    """
    model_dir = os.path.join(BASE_DIR, "runs", model)

    sizes = []

    for version in VERSIONS:
        if version == 0:
            continue

        file_path = find_matching_file(
            model_dir,
            f"version_{version}_"
        )

        if file_path is None:
            sizes.append(None)
        else:
            sizes.append(get_file_size_in_kib(file_path))

    return sizes


def get_full_update_average(model):
    """Return the average full-update size for a model."""
    sizes = [
        size
        for size in get_full_update_sizes(model)
        if size is not None
    ]

    if not sizes:
        return None

    return sum(sizes) / len(sizes)


# ============================================================
# SVD updates
# ============================================================

def get_svd_update_sizes(svd_information, model):
    """
    Return every individual SVD update size.

    The returned list corresponds to:

        version 0 -> 1
        version 1 -> 2
        version 2 -> 3
        version 3 -> 4

    Missing files are represented by None.
    """
    svd_dir = os.path.join(
        BASE_DIR,
        f"svd_out_{svd_information}",
        model
    )

    sizes = []

    for version in range(len(VERSIONS) - 1):
        file_path = find_matching_file(
            svd_dir,
            f"{model}_version_{version}_"
        )

        if file_path is None:
            sizes.append(None)
        else:
            sizes.append(get_file_size_in_kib(file_path))

    return sizes

def get_svd_update_average(model, svd_information):
    """Return the average SVD-update size for a model."""
    sizes = [
        size
        for size in get_svd_update_sizes(svd_information, model)
        if size is not None
    ]

    if not sizes:
        return None

    return sum(sizes) / len(sizes)


# ============================================================
# Delta updates
# ============================================================

def get_delta_update_sizes(model):
    """
    Return every individual delta update size.

    The returned list corresponds to:

        version 0 -> 1
        version 1 -> 2
        version 2 -> 3
        version 3 -> 4
    """
    delta_dir = os.path.join(
        BASE_DIR,
        "delta_out",
        model
    )

    sizes = []

    for version in range(len(VERSIONS) - 1):
        file_path = find_matching_file(
            delta_dir,
            f"{model}_version_{version}_"
        )

        if file_path is None:
            sizes.append(None)
        else:
            sizes.append(get_file_size_in_kib(file_path))

    return sizes

def get_delta_update_average(model):
    """Return the average delta-update size for a model."""
    sizes = [
        size
        for size in get_delta_update_sizes(model)
        if size is not None
    ]

    if not sizes:
        return None

    return sum(sizes) / len(sizes)

# ============================================================
# Plot
# ============================================================

def plot():
    """
    Plot all update types.

    Layout for each model:

        Full
        Delta v0->v1
        Delta v1->v2
        Delta v2->v3
        Delta v3->v4

        SVD 0.7
            v0->v1
            v1->v2
            v2->v3
            v3->v4

        SVD 0.8
            ...

    Each individual SVD update is displayed as its own bar.

    For every model/SVD-information combination, a horizontal min/max
    range is displayed across the corresponding SVD bars.
    """

    # --------------------------------------------------------
    # Load all data
    # --------------------------------------------------------

    full_update_sizes = {
        model: get_full_update_average(model)
        for model in MODELS
    }

    delta_update_sizes = {
        model: get_delta_update_average(model)
        for model in MODELS
    }

    svd_update_sizes = {
        (model, information): get_svd_update_sizes(information, model)
        for model in MODELS
        for information in SVD_INFORMATION
    }

    print("Full update sizes:", full_update_sizes)
    print("Delta update sizes:", delta_update_sizes)
    print("SVD update sizes:", svd_update_sizes)

    # --------------------------------------------------------
    # Plot setup
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(24, 12))

    full_positions = []
    full_values = []
    full_labels = []

    svd_70_positions = []
    svd_70_values = []
    svd_70_labels = []

    svd_80_positions = []
    svd_80_values = []
    svd_80_labels = []

    svd_90_positions = []
    svd_90_values = []
    svd_90_labels = []

    svd_95_positions = []
    svd_95_values = []
    svd_95_labels = []

    svd_97_positions = []
    svd_97_values = []
    svd_97_labels = []

    svd_98_positions = []
    svd_98_values = []
    svd_98_labels = []

    svd_99_positions = []
    svd_99_values = []
    svd_99_labels = []

    svd_100_positions = []
    svd_100_values = []
    svd_100_labels = []

    delta_positions = []
    delta_values = []
    delta_labels = []

    # Positions of SVD bars belonging to each group.
    # Used later for min/max markers.
    svd_groups = []

    # Keep track of model boundaries for visual grouping.
    model_boundaries = []

    current_pos = 0

    # --------------------------------------------------------
    # Build plot data
    # --------------------------------------------------------

    for model_index, model in enumerate(MODELS):

        model_start = current_pos

        # ====================================================
        # DELTA UPDATES
        # ====================================================

        delta_val = delta_update_sizes[model]

        if delta_val is not None:
            delta_positions.append(current_pos)
            delta_values.append(delta_val)
            delta_labels.append(f"delta")

        current_pos += 1

        # Small separation before SVD data.
        current_pos += 1

        # ====================================================
        # SVD UPDATES
        # ====================================================

        for information in SVD_INFORMATION:

            values = svd_update_sizes[(model, information)]

            valid_values = [
                value
                for value in values
                if value is not None
            ]

            if not valid_values:
                continue

            # ------------------------------------------------
            # Calculate average, minimum and maximum
            # ------------------------------------------------

            average = sum(valid_values) / len(valid_values)
            minimum = min(valid_values)
            maximum = max(valid_values)

            # ------------------------------------------------
            # Store the average as the bar height
            # ------------------------------------------------

            if information == SVD_INFORMATION[0]:  # 0.7
                svd_70_positions.append(current_pos)
                svd_70_values.append(average)
                svd_70_labels.append(f"svd_{information}")
            elif information == SVD_INFORMATION[1]:  # 0.8
                svd_80_positions.append(current_pos)
                svd_80_values.append(average)
                svd_80_labels.append(f"svd_{information}")
            elif information == SVD_INFORMATION[2]:  # 0.9
                svd_90_positions.append(current_pos)
                svd_90_values.append(average)
                svd_90_labels.append(f"svd_{information}")
            elif information == SVD_INFORMATION[3]:  # 0.95
                svd_95_positions.append(current_pos)
                svd_95_values.append(average)
                svd_95_labels.append(f"svd_{information}")
            elif information == SVD_INFORMATION[4]:  # 0.97
                svd_97_positions.append(current_pos)
                svd_97_values.append(average)
                svd_97_labels.append(f"svd_{information}")
            elif information == SVD_INFORMATION[5]:  # 0.98
                svd_98_positions.append(current_pos)
                svd_98_values.append(average)
                svd_98_labels.append(f"svd_{information}")
            elif information == SVD_INFORMATION[6]:  # 0.99
                svd_99_positions.append(current_pos)
                svd_99_values.append(average)
                svd_99_labels.append(f"svd_{information}")
            elif information == SVD_INFORMATION[7]:  # 1.0
                svd_100_positions.append(current_pos)
                svd_100_values.append(average)
                svd_100_labels.append(f"svd_{information}")

            # ------------------------------------------------
            # Store min/max for error bar
            # ------------------------------------------------

            svd_groups.append({
                "model": model,
                "information": information,
                "position": current_pos,
                "average": average,
                "minimum": minimum,
                "maximum": maximum,
            })

            current_pos += 1

            # Separation between SVD information levels
            current_pos += 1

        # ====================================================
        # FULL UPDATE
        # ====================================================

        full_value = full_update_sizes[model]

        if full_value is not None:
            full_positions.append(current_pos)
            if model == "mlp_big":
                full_values.append(0)
            else:
                full_values.append(full_value)
            full_labels.append(f"full")

            current_pos += 1

        # ----------------------------------------------------
        # Model boundary
        # ----------------------------------------------------

        model_end = current_pos - 1

        model_boundaries.append({
            "model": model,
            "start": model_start,
            "end": model_end,
        })

        # Larger separation between models.
        current_pos += 2

    # ========================================================
    # Draw bars
    # ========================================================

    # Delta
    delta_bars = ax.bar(
        delta_positions,
        delta_values,
        color="tab:green",
        width=0.8,
        alpha=0.75,
        label="Update size",
    )

    svd_70_bars = ax.bar(
        svd_70_positions,
        svd_70_values,
        color="#ffe498",
        width=0.8,
        alpha=0.75,
        label="SVD 70% Information",
    )
    svd_80_bars = ax.bar(
        svd_80_positions,
        svd_80_values,
        color="#ffdc7d",
        width=0.8,
        alpha=0.75,
        label="SVD 80% Information",
    )
    svd_90_bars = ax.bar(
        svd_90_positions,
        svd_90_values,
        color="#ffcf4b",
        width=0.8,
        alpha=0.75,
        label="SVD 90% Information",
    )
    svd_95_bars = ax.bar(
        svd_95_positions,
        svd_95_values,
        color="#ffc628",
        width=0.8,
        alpha=0.75,
        label="SVD 95% Information",
    )
    svd_97_bars = ax.bar(
        svd_97_positions,
        svd_97_values,
        color="#ffc423",
        width=0.8,
        alpha=0.75,
        label="SVD 97% Information",
    )
    svd_98_bars = ax.bar(
        svd_98_positions,
        svd_98_values,
        color="#ffc219",
        width=0.8,
        alpha=0.75,
        label="SVD 98% Information",
    )
    svd_99_bars = ax.bar(
        svd_99_positions,
        svd_99_values,
        color="#ffbe0d",
        width=0.8,
        alpha=0.75,
        label="SVD 99% Information",
    )
    svd_100_bars = ax.bar(
        svd_100_positions,
        svd_100_values,
        color="#ffbb00",
        width=0.8,
        alpha=0.75,
        label="SVD 100% Information",
    )

    full_bars = ax.bar(
        full_positions,
        full_values,
        color="tab:blue",
        width=0.8,
        alpha=0.75,
        label="Full Update",
    )

    # ========================================================
    # Draw min/max markers for SVD groups
    # ========================================================

    for group in svd_groups:

        position = group["position"]
        minimum = group["minimum"]
        maximum = group["maximum"]

        left = position - 0.4
        right = position + 0.4

        # --------------------------------------------
        # Minimum line
        # --------------------------------------------

        ax.hlines(
            minimum,
            left,
            right,
            colors="black",
            linewidth=1,
            linestyles="--",
            label="_nolegend_",
        )

        # --------------------------------------------
        # Maximum line
        # --------------------------------------------

        ax.hlines(
            maximum,
            left,
            right,
            colors="black",
            linewidth=1,
            linestyles="--",
            label="_nolegend_",
        )

        # --------------------------------------------
        # Vertical min/max range
        # --------------------------------------------

        center = (left + right) / 2

        ax.vlines(
            center,
            minimum,
            maximum,
            colors="black",
            linewidth=1,
            label="_nolegend_",
        )

        # --------------------------------------------
        # Labels
        # --------------------------------------------

        # ax.text(
        #     right + 0.15,
        #     minimum,
        #     f"min {minimum:.0f}",
        #     va="center",
        #     ha="left",
        #     fontsize=8,
        # )

        # ax.text(
        #     right + 0.15,
        #     maximum,
        #     f"max {maximum:.0f}",
        #     va="center",
        #     ha="left",
        #     fontsize=8,
        # )

    # ========================================================
    # Model separators
    # ========================================================

    for boundary in model_boundaries[:-1]:

        separator_position = boundary["end"] + 1.5

        ax.axvline(
            separator_position,
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )

    # ========================================================
    # Add model names at the top
    # ========================================================

    for boundary in model_boundaries:

        center = (
            boundary["start"] +
            boundary["end"]
        ) / 2

        ax.text(
            center,
            -0.1,
            boundary["model"].upper(),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # ========================================================
    # Axes and labels
    # ========================================================

    ax.set_xlabel("Model / Update", fontsize=18, labelpad=20)
    ax.set_ylabel("Update Size (KiB)", fontsize=18)

    # ax.set_title(
    #     "Model Update Size Comparison"
    # )

    ax.set_xticks(full_positions + 
                  svd_70_positions + 
                  svd_80_positions + 
                  svd_90_positions + 
                  svd_95_positions + 
                  svd_97_positions + 
                  svd_98_positions + 
                  svd_99_positions + 
                  svd_100_positions + 
                  delta_positions)
    ax.set_xticklabels(
        full_labels + 
        svd_70_labels + 
        svd_80_labels + 
        svd_90_labels + 
        svd_95_labels + 
        svd_97_labels + 
        svd_98_labels + 
        svd_99_labels + 
        svd_100_labels + 
        delta_labels,
        rotation=90,
        fontsize=8,
    )

    # ========================================================
    # Legend
    # ========================================================

    # We create proxy lines for the min/max legend entries.
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[full_bars, 
                 svd_70_bars, 
                 svd_80_bars, 
                 svd_90_bars, 
                 svd_95_bars, 
                 svd_97_bars, 
                 svd_98_bars, 
                 svd_99_bars, 
                 svd_100_bars, 
                 delta_bars],
        labels=[
            "Full Update",
            "SVD 70% Information",
            "SVD 80% Information",
            "SVD 90% Information",
            "SVD 95% Information",
            "SVD 97% Information",
            "SVD 98% Information",
            "SVD 99% Information",
            "SVD 100% Information",
            "Delta Update",
        ],
        loc="upper right",
    )

    # ========================================================
    # Grid
    # ========================================================

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    # Keep grid behind bars.
    ax.set_axisbelow(True)

    # ========================================================
    # Layout
    # ========================================================

    plt.tight_layout(pad=2.0)

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plot full, SVD, and delta model update sizes."
    )

    parser.add_argument(
        "--svd_information",
        type=float,
        default=1.0,
        help=(
            "SVD information used for the output directory. "
            "All configured SVD energies are plotted; "
            "this argument is retained for compatibility."
        ),
    )

    args = parser.parse_args()

    plot()