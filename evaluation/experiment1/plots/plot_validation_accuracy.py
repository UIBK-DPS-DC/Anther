import os
import json
import re
import argparse

import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODELS = [
    # "cnn",
    "cnn1d",
    "crnn",
    "ds_cnn",
    "gru",
    "mlp",
    "svdf",
]

VERSIONS = [0, 1, 2, 3, 4]

# Versions from which an update (SVD / delta / full) exists.
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
# Benchmark file discovery
# ============================================================

def find_benchmark_files():
    """
    Recursively scan BASE_DIR for all files ending in
    '_benchmark.json'.
    """
    benchmark_files = []

    for root, dirs, files in os.walk(BASE_DIR):
        for filename in files:
            if filename.endswith("_benchmark.json"):
                benchmark_files.append(os.path.join(root, filename))

    return benchmark_files


def parse_benchmark_filename(filepath):
    """
    Extract version from a benchmark filename.

    Examples:

        version_0_9333_benchmark.json
            -> 0

        version_3_12345_svd_updated_benchmark.json
            -> 3
    """
    filename = os.path.basename(filepath)

    match = re.match(
        r"version_(\d+)_.*_benchmark\.json$",
        filename
    )

    if match is None:
        return None

    return int(match.group(1))


def get_update_type(filepath):
    """
    Determine whether a benchmark is:

        base
        svd_0.7
        svd_0.8
        ...
        svd_1
    """
    normalized_path = os.path.normpath(filepath)
    path_parts = normalized_path.split(os.sep)

    for energy in SVD_INFORMATION:
        expected_dir = f"benchmark_svd_partial_update_{energy}"
        if expected_dir in path_parts:
            return f"svd_{energy}"

    return "base"


def get_model(filepath):
    """Determine the model name from the directory structure."""
    normalized_path = os.path.normpath(filepath)
    path_parts = normalized_path.split(os.sep)

    for model in MODELS:
        if model in path_parts:
            return model

    return None


def get_validation_accuracy(filepath):
    """Read validation_accuracy from a benchmark JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not read {filepath}: {exc}")
        return None

    if "validation_accuracy" not in data:
        print(f"WARNING: No validation_accuracy in {filepath}")
        return None

    return data["validation_accuracy"]


def load_benchmarks():
    """
    Organize benchmark data as:

        data[version][update_type][model] = accuracy
    """
    data = {}

    benchmark_files = find_benchmark_files()

    print(f"Found {len(benchmark_files)} benchmark files.")

    for filepath in benchmark_files:

        model = get_model(filepath)
        version = parse_benchmark_filename(filepath)
        update_type = get_update_type(filepath)

        if model is None:
            print(f"WARNING: Could not determine model: {filepath}")
            continue

        if version is None:
            print(f"WARNING: Could not determine version: {filepath}")
            continue

        accuracy = get_validation_accuracy(filepath)

        if accuracy is None:
            continue

        data.setdefault(version, {})
        data[version].setdefault(update_type, {})

        if model in data[version][update_type]:
            print(
                f"WARNING: Duplicate benchmark for {model}, "
                f"version {version}, {update_type}"
            )
            print(f"  Existing: {data[version][update_type][model]}")
            print(f"  Ignoring: {filepath}")
            continue

        data[version][update_type][model] = accuracy

    return data


def print_data(data):

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
                    accuracy = data[version][update_type][model]
                    print(f"    {model:8s}: {accuracy:.6f}")


# ============================================================
# X-axis construction
#
# One continuous sequence, shared by all models:
#
#   v0, v0+70%, v0+80%, ..., v0+1, v0+delta, v0+full,
#   v1, v1+70%, ..., v1+full,
#   v2, ...
#   v3, ..., v3+full,
#   v4
# ============================================================

def build_categories():
    """
    Build the ordered list of (version, update_type, label) tuples
    that make up the shared x-axis.
    """
    categories = []

    for version in VERSIONS:

        categories.append((version, "base", f"v{version}"))

        if version in BASE_VERSIONS:

            for energy in SVD_INFORMATION:
                categories.append((
                    version,
                    f"svd_{energy}",
                    f"v{version}+{float(energy) * 100:.0f}%",
                ))

            categories.append((version, "delta", f"v{version}+delta"))
            categories.append((version, "full", f"v{version}+full"))

    return categories


def get_accuracy_for_category(accuracy_data, model, version, update_type):
    """
    Resolve the accuracy value for a single (model, version,
    update_type) x-axis category.

    'delta' and 'full' updates don't have their own accuracy entry;
    both represent arriving at the next version, so they use the
    base accuracy of version + 1.
    """
    if update_type in ("delta", "full"):
        return (
            accuracy_data
            .get(version + 1, {})
            .get("base", {})
            .get(model)
        )

    return (
        accuracy_data
        .get(version, {})
        .get(update_type, {})
        .get(model)
    )


def build_model_series(accuracy_data, categories, model):
    """
    Build the (x_positions, y_values) series for a single model
    across the full category sequence. Missing points are simply
    skipped, the line just doesn't have a point there.
    """
    x_values = []
    y_values = []

    for x, (version, update_type, _label) in enumerate(categories):

        accuracy = get_accuracy_for_category(
            accuracy_data, model, version, update_type
        )

        if accuracy is None:
            continue

        x_values.append(x)
        y_values.append(accuracy)

    return x_values, y_values


# ============================================================
# Plot
# ============================================================

def plot_validation_accuracy(accuracy_data):
    """
    Plot validation accuracy across the update sequence in three
    subplots:

        1. Raw accuracy, one line per model.
        2. Average accuracy across all models.
        3. Accuracy per model, min-max normalized to each model's
           own range, to compare relative change regardless of
           absolute accuracy level.
    """

    categories = build_categories()

    model_colors = {
        "cnn": "tab:blue",
        "cnn1d": "tab:orange",
        "crnn": "tab:green",
        "ds_cnn": "tab:red",
        "gru": "tab:purple",
        "mlp": "tab:brown",
        "svdf": "tab:pink",
    }

    # --------------------------------------------------------
    # Build per-model series once, reused across subplots.
    # --------------------------------------------------------

    model_series = {
        model: build_model_series(accuracy_data, categories, model)
        for model in MODELS
    }

    fig, (ax_raw, ax_avg, ax_norm) = plt.subplots(
        3, 1, figsize=(24, 20), sharex=True
    )

    # --------------------------------------------------------
    # Subplot 1: raw accuracy, one line per model.
    # --------------------------------------------------------

    for model in MODELS:

        x_values, y_values = model_series[model]

        if not x_values:
            continue

        ax_raw.plot(
            x_values,
            y_values,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=model_colors[model],
            label=model,
        )

    ax_raw.set_title("Validation Accuracy (raw)", fontsize=16)
    ax_raw.set_ylabel("Validation Accuracy", fontsize=14)
    ax_raw.grid(alpha=0.3)
    ax_raw.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    # --------------------------------------------------------
    # Subplot 2: average accuracy across models.
    # --------------------------------------------------------

    average_x = []
    average_y = []

    for x, (version, update_type, _label) in enumerate(categories):

        values = [
            get_accuracy_for_category(
                accuracy_data, model, version, update_type
            )
            for model in MODELS
        ]

        values = [v for v in values if v is not None]

        if not values:
            continue

        average_x.append(x)
        average_y.append(sum(values) / len(values))

    ax_avg.plot(
        average_x,
        average_y,
        marker="o",
        markersize=4,
        linewidth=2,
        color="tab:gray",
        label="Average across models",
    )

    ax_avg.set_title(
        "Validation Accuracy (average across models)", fontsize=16
    )
    ax_avg.set_ylabel("Validation Accuracy", fontsize=14)
    ax_avg.grid(alpha=0.3)
    ax_avg.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    # --------------------------------------------------------
    # Subplot 3: per-model, min-max normalized to each model's
    # own range.
    # --------------------------------------------------------

    for model in MODELS:

        x_values, y_values = model_series[model]

        if not y_values:
            continue

        minimum = min(y_values)
        maximum = max(y_values)

        if maximum == minimum:
            normalized = [0.5 for _ in y_values]
        else:
            normalized = [
                (value - minimum) / (maximum - minimum)
                for value in y_values
            ]

        ax_norm.plot(
            x_values,
            normalized,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=model_colors[model],
            label=model,
        )

    ax_norm.set_title(
        "Validation Accuracy (normalized per model, min-max scaled)",
        fontsize=16,
    )
    ax_norm.set_ylabel(
        "Normalized Accuracy (0 = model min, 1 = model max)",
        fontsize=14,
    )
    ax_norm.grid(alpha=0.3)
    ax_norm.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    # --------------------------------------------------------
    # Shared x-axis
    # --------------------------------------------------------

    ax_norm.set_xlabel("Update Step", fontsize=14, labelpad=15)

    ax_norm.set_xticks(range(len(categories)))
    ax_norm.set_xticklabels(
        [label for _, _, label in categories],
        rotation=90,
        fontsize=8,
    )

    # Mark the start of each new version ("base") across all subplots.
    version_boundaries = [
        x
        for x, (_version, update_type, _label) in enumerate(categories)
        if update_type == "base"
    ]

    for boundary_x in version_boundaries:
        for ax in (ax_raw, ax_avg, ax_norm):
            ax.axvline(
                boundary_x,
                linewidth=1,
                linestyle="--",
                alpha=0.4,
                color="black",
            )

    plt.tight_layout(pad=2.0)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Plot validation accuracy across the update sequence "
            "(base / SVD / delta / full)."
        )
    )

    parser.parse_args()

    accuracy_data = load_benchmarks()

    print_data(accuracy_data)

    plot_validation_accuracy(accuracy_data)