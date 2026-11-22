import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    "cnn",
    "cnn1d",
    "crnn",
    "ds_cnn",
    "gru",
    "mlp",
    "svdf",
]

BASE_VERSIONS = [0, 1, 2, 3]

# Version 4 is only the normal/final model.
FINAL_VERSION = 4

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
# File discovery
# ============================================================

def find_benchmark_files():
    """
    Recursively scan CURRENT_DIR for all files ending in
    '_benchmark.json'.

    Returns a list of absolute file paths.
    """

    benchmark_files = []

    for root, dirs, files in os.walk(CURRENT_DIR):

        for filename in files:

            if filename.endswith("_benchmark.json"):

                filepath = os.path.join(root, filename)
                benchmark_files.append(filepath)

    return benchmark_files


# ============================================================
# Filename parsing
# ============================================================

def parse_benchmark_filename(filepath):
    """
    Extract model and version from a benchmark filename.

    Examples:

        version_0_9333_benchmark.json
            -> version = 0

        version_1_27999_benchmark.json
            -> version = 1

        version_3_12345_svd_updated_benchmark.json
            -> version = 3

    The checkpoint number is deliberately ignored.
    """

    filename = os.path.basename(filepath)

    match = re.match(
        r"version_(\d+)_.*_benchmark\.json$",
        filename
    )

    if match is None:
        return None

    version = int(match.group(1))

    return version


# ============================================================
# Determine update type from path
# ============================================================

def get_update_type(filepath):
    """
    Determine whether a benchmark is:

        base
        svd_0.7
        svd_0.8
        svd_0.9
        svd_0.95
        svd_0.97
        svd_0.98
        svd_0.99
        svd_1

    based on the directory structure.

    Expected examples:

        runs/mlp/version_0_9333_benchmark.json

        benchmark_svd_partial_update_0.7/mlp/
            version_0_9333_svd_updated_benchmark.json

        benchmark_svd_partial_update_0.8/mlp/
            version_0_9333_svd_updated_benchmark.json
    """

    normalized_path = os.path.normpath(filepath)

    path_parts = normalized_path.split(os.sep)

    # --------------------------------------------------------
    # SVD directories
    # --------------------------------------------------------

    for information in SVD_INFORMATION:

        expected_dir = f"benchmark_svd_partial_update_{information}"

        if expected_dir in path_parts:
            return f"svd_{information}"

    # --------------------------------------------------------
    # Otherwise this is a normal/base benchmark
    # --------------------------------------------------------

    return "base"


# ============================================================
# Determine model from path
# ============================================================

def get_model(filepath):
    """
    Determine the model name from the directory structure.

    Expected:

        runs/mlp/...
        benchmark_svd_partial_update_0.7/mlp/...
    """

    normalized_path = os.path.normpath(filepath)
    path_parts = normalized_path.split(os.sep)

    # Look for a directory matching one of our models.
    for model in MODELS:

        if model in path_parts:
            return model

    return None


# ============================================================
# Read validation accuracy
# ============================================================

def get_validation_accuracy(filepath):
    """
    Read validation_accuracy from a benchmark JSON file.
    """

    try:

        with open(filepath, "r") as f:
            data = json.load(f)

    except (OSError, json.JSONDecodeError) as exc:

        print(
            f"WARNING: Could not read {filepath}: {exc}"
        )

        return None

    if "validation_accuracy" not in data:

        print(
            f"WARNING: No validation_accuracy in {filepath}"
        )

        return None

    return data["validation_accuracy"]


# ============================================================
# Build benchmark database
# ============================================================

def load_benchmarks():
    """
    Scan all benchmark files and organize them as:

        data[version][update_type][model] = accuracy

    Example:

        data[0]["base"]["mlp"] = 0.923
        data[0]["svd_0.7"]["mlp"] = 0.918
        data[0]["svd_0.8"]["mlp"] = 0.921
        ...

    """

    data = {}

    benchmark_files = find_benchmark_files()

    print(
        f"Found {len(benchmark_files)} benchmark files."
    )

    for filepath in benchmark_files:

        model = get_model(filepath)
        version = parse_benchmark_filename(filepath)
        update_type = get_update_type(filepath)

        if model is None:
            print(
                f"WARNING: Could not determine model: "
                f"{filepath}"
            )
            continue

        if version is None:
            print(
                f"WARNING: Could not determine version: "
                f"{filepath}"
            )
            continue

        accuracy = get_validation_accuracy(filepath)

        if accuracy is None:
            continue

        # Create nested dictionaries.
        if version not in data:
            data[version] = {}

        if update_type not in data[version]:
            data[version][update_type] = {}

        # ----------------------------------------------------
        # Check for duplicates
        # ----------------------------------------------------

        if model in data[version][update_type]:

            print(
                f"WARNING: Duplicate benchmark for "
                f"{model}, version {version}, "
                f"{update_type}"
            )

            print(
                f"  Existing: "
                f"{data[version][update_type][model]}"
            )

            print(
                f"  Ignoring: {filepath}"
            )

            continue

        data[version][update_type][model] = accuracy

    return data


# ============================================================
# Print discovered data
# ============================================================

def print_data(data):
    """
    Print the discovered benchmark structure for debugging.
    """

    print("\n" + "=" * 70)
    print("DISCOVERED BENCHMARK DATA")
    print("=" * 70)

    update_types = [
        "base",
        "svd_0.7",
        "svd_0.8",
        "svd_0.9",
        "svd_0.95",
        "svd_0.97",
        "svd_0.98",
        "svd_0.99",
        "svd_1",
    ]

    for version in sorted(data.keys()):

        print(f"\nVersion {version}")

        for update_type in update_types:

            if update_type not in data[version]:
                continue

            print(f"  {update_type}")

            for model in MODELS:

                if model in data[version][update_type]:

                    accuracy = (
                        data[version][update_type][model]
                    )

                    print(
                        f"    {model:8s}: "
                        f"{accuracy:.6f}"
                    )


# ============================================================
# Plot
# ============================================================

def plot_accuracy(data):
    """
    Create the accuracy comparison plot.

    Each x-axis group represents one model-generation method:

        Version 0
            Base
            SVD 70%
            SVD 80%
            SVD 90%
            SVD 95%
            SVD 97%
            SVD 98%
            SVD 99%
            SVD 100%

        Version 1
            ...

        Version 4
            Base only

    Each method contains 7 bars, one for each model.
    """

    update_types = [
        "base",
        "svd_0.7",
        "svd_0.8",
        "svd_0.9",
        "svd_0.95",
        "svd_0.97",
        "svd_0.98",
        "svd_0.99",
        "svd_1",
    ]

    update_labels = {
        "base": "Base",
        "svd_0.7": "SVD 70%",
        "svd_0.8": "SVD 80%",
        "svd_0.9": "SVD 90%",
        "svd_0.95": "SVD 95%",
        "svd_0.97": "SVD 97%",
        "svd_0.98": "SVD 98%",
        "svd_0.99": "SVD 99%",
        "svd_1": "SVD 100%",
    }

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    # One color per model.
    # This makes it easy to follow a particular model across
    # all versions and SVD settings.

    model_colors = {
        "cnn": "tab:blue",
        "cnn1d": "tab:orange",
        "crnn": "tab:green",
        "ds_cnn": "tab:red",
        "gru": "tab:purple",
        "mlp": "tab:brown",
        "svdf": "tab:pink",
    }

    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(figsize=(22, 10))

    # --------------------------------------------------------
    # One x/y series per model
    # --------------------------------------------------------

    model_x = {
        model: []
        for model in MODELS
    }

    model_y = {
        model: []
        for model in MODELS
    }

    group_centers = []
    group_labels = []

    current_group = 0
    group_spacing = 1.0


    # ========================================================
    # Versions 0-3
    # ========================================================

    for version in BASE_VERSIONS:

        for update_type in update_types:

            # Skip if this data doesn't exist
            if version not in data:
                continue

            if update_type not in data[version]:
                continue

            # ------------------------------------------------
            # One x position for ALL models
            # ------------------------------------------------

            x = current_group * group_spacing

            group_centers.append(x)

            if update_type == "base":
                group_labels.append(
                    f"Base v{version}"
                )
            else:
                group_labels.append(
                    f"{update_labels[update_type]}"
                )

            # ------------------------------------------------
            # Add one point to each model's line
            # ------------------------------------------------

            for model in MODELS:

                accuracy = (
                    data[version]
                    .get(update_type, {})
                    .get(model)
                )

                if accuracy is None:
                    continue

                model_x[model].append(x)
                model_y[model].append(accuracy)

            current_group += 1

        # ----------------------------------------------------
        # Larger gap between versions
        # ----------------------------------------------------

        current_group += 1


    # ========================================================
    # Version 4
    # ========================================================

    version = FINAL_VERSION
    update_type = "base"

    if version in data and update_type in data[version]:

        x = current_group * group_spacing

        group_centers.append(x)

        group_labels.append(
            "v4\nBase"
        )

        for model in MODELS:

            accuracy = (
                data[version]
                .get(update_type, {})
                .get(model)
            )

            if accuracy is None:
                continue

            model_x[model].append(x)
            model_y[model].append(accuracy)


    # ========================================================
    # Plot the 7 model lines
    # ========================================================

    for model in MODELS:

        if not model_x[model]:
            continue

        ax.plot(
            model_x[model],
            model_y[model],
            marker="o",
            linewidth=2,
            color=model_colors[model],
            label=model,
        )


    # ========================================================
    # Formatting
    # ========================================================

    ax.set_xticks(group_centers)

    ax.set_xticklabels(
        group_labels,
        rotation=90,
    )

    ax.set_xlabel(
        "Model Version / Update Method",
        fontsize=18,
    )

    ax.set_ylabel(
        "Validation Accuracy",
        fontsize=18,
    )

    # ax.set_title(
    #     "Validation Accuracy Across Model Versions "
    #     "and SVD Updates"
    # )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.set_axisbelow(True)

    ax.legend(
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )

    plt.tight_layout(pad=2.0)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    data = load_benchmarks()

    print_data(data)

    plot_accuracy(data)