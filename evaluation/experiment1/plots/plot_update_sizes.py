import os
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
    "svdf",
]

VERSIONS = [0, 1, 2, 3, 4]

# Versions for which SVD/delta/full updates exist.
BASE_VERSIONS = [0, 1, 2, 3]

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
        if f.startswith(prefix) and f.endswith(".bin")
    ]

    if not matching_files:
        return None

    matching_files.sort()

    return os.path.join(directory, matching_files[0])


# ============================================================
# SVD updates
# ============================================================

def get_svd_update_sizes(svd_energy, model):
    """
    Return every individual SVD update size.

    The returned list corresponds to:

        version 0 -> 1
        version 1 -> 2
        version 2 -> 3
        version 3 -> 4
    """
    svd_dir = os.path.join(
        BASE_DIR,
        f"svd_out_{svd_energy}",
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


def get_svd_update_average(model, svd_energy):
    """Return the average SVD-update size for a model."""
    sizes = [
        size
        for size in get_svd_update_sizes(svd_energy, model)
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
        "bpatch_out",
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


def get_full_update_sizes(model):
    """
    Get the full model update size for every version transition.

    Version 0 is skipped because it is treated as the initial
    model rather than an update.
    """
    model_dir = os.path.join(BASE_DIR, "full_out", model)

    sizes = []

    for version in VERSIONS:
        if version == 0:
            continue

        file_path = find_matching_file(
            model_dir,
            f"{model}_version_{version}_"
        )

        if file_path is None:
            sizes.append(None)
        else:
            sizes.append(get_file_size_in_kib(file_path))

    return sizes


# ============================================================
# Plot
# ============================================================

def plot_update_sizes():
    """
    Plot SVD/delta/full update sizes as bars, grouped by model.
    """

    # ========================================================
    # Load update-size data
    # ========================================================

    delta_update_sizes = {
        model: get_delta_update_average(model)
        for model in MODELS
    }

    svd_update_sizes = {
        (model, energy): get_svd_update_sizes(energy, model)
        for model in MODELS
        for energy in SVD_INFORMATION
    }

    full_update_sizes = {
        model: get_full_update_sizes(model)
        for model in MODELS
    }

    # ========================================================
    # Plot setup
    # ========================================================

    fig, ax = plt.subplots(figsize=(24, 10))

    # ========================================================
    # Bar positions
    # ========================================================

    delta_positions = []
    delta_values = []
    delta_labels = []

    full_positions = []
    full_values = []
    full_labels = []

    svd_positions = {energy: [] for energy in SVD_INFORMATION}
    svd_values = {energy: [] for energy in SVD_INFORMATION}
    svd_labels = {energy: [] for energy in SVD_INFORMATION}

    svd_groups = []

    model_boundaries = []

    current_pos = 0

    # ========================================================
    # Build the x-axis
    # ========================================================

    for model in MODELS:

        model_start = current_pos

        # ----------------------------------------------------
        # SVD
        # ----------------------------------------------------

        for energy in SVD_INFORMATION:

            values = svd_update_sizes[(model, energy)]

            valid_values = [v for v in values if v is not None]

            if not valid_values:
                continue

            average = sum(valid_values) / len(valid_values)
            minimum = min(valid_values)
            maximum = max(valid_values)

            svd_positions[energy].append(current_pos)
            svd_values[energy].append(average)
            svd_labels[energy].append(f"svd_{energy}")

            svd_groups.append({
                "model": model,
                "energy": energy,
                "position": current_pos,
                "average": average,
                "minimum": minimum,
                "maximum": maximum,
            })

            current_pos += 1

            # Separation between SVD energy levels.
            current_pos += 1

        # Separation before delta.
        current_pos += 1

        # ----------------------------------------------------
        # Delta
        # ----------------------------------------------------

        delta_val = delta_update_sizes[model]

        if delta_val is not None:

            delta_positions.append(current_pos)
            delta_values.append(delta_val)
            delta_labels.append("delta")

            current_pos += 1

        # ----------------------------------------------------
        # Full
        # ----------------------------------------------------

        full_val = full_update_sizes[model]

        if full_val is not None:

            full_positions.append(current_pos)
            full_values.append(sum(full_val) / len(full_val))
            full_labels.append("full")

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

        # Larger gap between models.
        current_pos += 2

    # ========================================================
    # Draw bars
    # ========================================================

    delta_bars = ax.bar(
        delta_positions,
        delta_values,
        color="tab:green",
        width=0.8,
        alpha=0.75,
        label="Delta Update",
    )

    full_bars = ax.bar(
        full_positions,
        full_values,
        color="tab:blue",
        width=0.8,
        alpha=0.75,
        label="Full Update",
    )

    svd_colors = {
        "0.7": "#ffe498",
        "0.8": "#ffdc7d",
        "0.9": "#ffcf4b",
        "0.95": "#ffc628",
        "0.97": "#ffc423",
        "0.98": "#ffc219",
        "0.99": "#ffbe0d",
        "1": "#ffbb00",
    }

    svd_bars = {}

    for energy in SVD_INFORMATION:

        svd_bars[energy] = ax.bar(
            svd_positions[energy],
            svd_values[energy],
            color=svd_colors[energy],
            width=0.8,
            alpha=0.75,
            label=f"SVD {float(energy) * 100:.0f}% Information",
        )

    # ========================================================
    # SVD min/max markers
    # ========================================================

    for group in svd_groups:

        position = group["position"]
        minimum = group["minimum"]
        maximum = group["maximum"]

        left = position - 0.4
        right = position + 0.4

        ax.hlines(
            minimum, left, right,
            colors="black", linewidth=1, linestyles="--",
            label="_nolegend_",
        )

        ax.hlines(
            maximum, left, right,
            colors="black", linewidth=1, linestyles="--",
            label="_nolegend_",
        )

        ax.vlines(
            position, minimum, maximum,
            colors="black", linewidth=1,
            label="_nolegend_",
        )

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
    # Model names
    # ========================================================

    for boundary in model_boundaries:

        center = (boundary["start"] + boundary["end"]) / 2

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

    all_positions = delta_positions.copy() + full_positions.copy()
    all_labels = delta_labels.copy() + full_labels.copy()

    for energy in SVD_INFORMATION:
        all_positions.extend(svd_positions[energy])
        all_labels.extend(svd_labels[energy])

    x_axis_data = sorted(
        zip(all_positions, all_labels),
        key=lambda item: item[0]
    )

    ax.set_xticks([item[0] for item in x_axis_data])
    ax.set_xticklabels(
        [item[1] for item in x_axis_data],
        rotation=90,
        fontsize=8,
    )

    # ========================================================
    # Legend
    # ========================================================

    ax.legend(
        handles=[
            delta_bars,
            *[svd_bars[energy] for energy in SVD_INFORMATION],
            full_bars,
        ],
        labels=[
            "Delta Update",
            *[
                f"SVD {float(energy) * 100:.0f}% Information"
                for energy in SVD_INFORMATION
            ],
            "Full Update",
        ],
        loc="upper right",
    )

    # ========================================================
    # Grid / layout
    # ========================================================

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=2.0)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plot model update sizes (SVD / delta / full)."
    )

    parser.parse_args()

    plot_update_sizes()