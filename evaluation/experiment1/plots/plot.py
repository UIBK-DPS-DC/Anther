import os
import json
import re
import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

# Versions for which updates exist.
BASE_VERSIONS = [0, 1, 2, 3]

# Final normal model.
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
# File helpers
# ============================================================

def get_file_size_in_kib(file_path):
    """Return file size in KiB."""
    return os.path.getsize(file_path) / 1024


def find_matching_file(directory, prefix, extension=".bin"):
    """
    Find the first file in 'directory' starting with 'prefix'
    and ending with 'extension'.
    """
    if not os.path.isdir(directory):
        return None

    matching_files = [
        f
        for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(extension)
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

    Corresponds to:

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

    for version in BASE_VERSIONS:

        file_path = find_matching_file(
            svd_dir,
            f"{model}_version_{version}_"
        )

        if file_path is None:
            sizes.append(None)
        else:
            sizes.append(
                get_file_size_in_kib(file_path)
            )

    return sizes


# ============================================================
# SVD + Quantization updates
# ============================================================

def get_svd_quant_update_sizes(svd_energy, model):
    """
    Return every individual SVD+Quant update size.

    Same layout as get_svd_update_sizes(), but the update
    files live in a directory suffixed with "_quant":

        svd_out_{svd_energy}_quant/<model>/

    Corresponds to:

        version 0 -> 1
        version 1 -> 2
        version 2 -> 3
        version 3 -> 4
    """
    svd_quant_dir = os.path.join(
        BASE_DIR,
        f"svd_out_{svd_energy}_quant",
        model
    )

    sizes = []

    for version in BASE_VERSIONS:

        file_path = find_matching_file(
            svd_quant_dir,
            f"{model}_version_{version}_"
        )

        if file_path is None:
            sizes.append(None)
        else:
            sizes.append(
                get_file_size_in_kib(file_path)
            )

    return sizes


# ============================================================
# Quantization updates
# ============================================================

def get_quant_update_sizes(model):
    """
    Return every individual QuantDelta update size.

    Corresponds to:

        version 0 -> 1
        version 1 -> 2
        version 2 -> 3
        version 3 -> 4

    Expected directory:

        quant_out/<model>/
    """
    quant_dir = os.path.join(
        BASE_DIR,
        "quant_out",
        model
    )

    sizes = []

    for version in BASE_VERSIONS:

        file_path = find_matching_file(
            quant_dir,
            f"{model}_version_{version}_"
        )

        if file_path is None:
            sizes.append(None)
        else:
            sizes.append(
                get_file_size_in_kib(file_path)
            )

    return sizes


def get_quant_update_average(model):
    """Return average QuantDelta update size."""
    sizes = [
        size
        for size in get_quant_update_sizes(model)
        if size is not None
    ]

    if not sizes:
        return None

    return sum(sizes) / len(sizes)


# ============================================================
# BPatch updates
# ============================================================

def get_bpatch_update_sizes(model):
    """
    Return every individual bsdiff/bpatch update size.

    Expected directory:

        bpatch_out/<model>/
    """
    bpatch_dir = os.path.join(
        BASE_DIR,
        "bpatch_out",
        model
    )

    sizes = []

    for version in BASE_VERSIONS:

        file_path = find_matching_file(
            bpatch_dir,
            f"{model}_version_{version}_"
        )

        if file_path is None:
            sizes.append(None)
        else:
            sizes.append(
                get_file_size_in_kib(file_path)
            )

    return sizes


def get_bpatch_update_average(model):
    """Return average BPatch update size."""
    sizes = [
        size
        for size in get_bpatch_update_sizes(model)
        if size is not None
    ]

    if not sizes:
        return None

    return sum(sizes) / len(sizes)


# ============================================================
# Full model updates
# ============================================================

def get_full_update_sizes(model):
    """
    Get the full model size for versions 1..4.

    Version 0 is the initial model and therefore skipped.
    """
    model_dir = os.path.join(
        BASE_DIR,
        "full_out",
        model
    )

    sizes = []

    for version in VERSIONS:

        if version == 0:
            continue

        file_path = find_matching_file(
            model_dir,
            f"{model}_version_{version}_",
            extension=".bin"
        )

        if file_path is None:
            sizes.append(None)
        else:
            sizes.append(
                get_file_size_in_kib(file_path)
            )

    return sizes


def get_full_update_average(model):
    """Return average full-model size."""
    sizes = [
        size
        for size in get_full_update_sizes(model)
        if size is not None
    ]

    if not sizes:
        return None

    return sum(sizes) / len(sizes)


# ============================================================
# Accuracy: file discovery
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

                benchmark_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    return benchmark_files


# ============================================================
# Accuracy: filename parsing
# ============================================================

def parse_benchmark_filename(filepath):
    """
    Extract version from a benchmark filename.

    Examples:

        version_0_9333_benchmark.json
            -> 0

        version_1_27999_benchmark.json
            -> 1

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


# ============================================================
# Accuracy: determine update type
# ============================================================

def get_update_type(filepath):
    """
    Determine whether a benchmark is:

        base
        svd_0.7
        ...
        svd_1
        svd_quant_0.7
        ...
        svd_quant_1
        quant
    """

    normalized_path = os.path.normpath(filepath)

    path_parts = normalized_path.split(os.sep)

    # --------------------------------------------------------
    # SVD + Quantization
    #
    # Checked first since these directories are suffixed
    # versions of the plain SVD directories below.
    # --------------------------------------------------------

    for energy in SVD_INFORMATION:

        expected_dir_quant = (
            f"benchmark_svd_partial_update_{energy}_quant"
        )

        if expected_dir_quant in path_parts:
            return f"svd_quant_{energy}"

    # --------------------------------------------------------
    # SVD
    # --------------------------------------------------------

    for energy in SVD_INFORMATION:

        expected_dir = (
            f"benchmark_svd_partial_update_{energy}"
        )

        if expected_dir in path_parts:
            return f"svd_{energy}"

    # --------------------------------------------------------
    # Quantization
    # --------------------------------------------------------

    if "benchmark_quant_update" in path_parts:
        return "quant"

    # --------------------------------------------------------
    # Normal/base benchmark
    # --------------------------------------------------------

    return "base"


# ============================================================
# Accuracy: determine model
# ============================================================

def get_model(filepath):
    """
    Determine the model name from the directory structure.
    """

    normalized_path = os.path.normpath(filepath)

    path_parts = normalized_path.split(os.sep)

    for model in MODELS:

        if model in path_parts:
            return model

    return None


# ============================================================
# Accuracy: read validation accuracy
# ============================================================

def get_validation_accuracy(filepath):
    """Read validation_accuracy from a benchmark JSON file."""

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
# Accuracy: build database
# ============================================================

def load_benchmarks():
    """
    Organize benchmark data as:

        data[version][update_type][model] = accuracy
    """

    data = {}

    benchmark_files = find_benchmark_files()

    print(
        f"Found {len(benchmark_files)} benchmark files."
    )

    for filepath in benchmark_files:

        model = get_model(filepath)

        version = parse_benchmark_filename(
            filepath
        )

        update_type = get_update_type(
            filepath
        )

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

        accuracy = get_validation_accuracy(
            filepath
        )

        if accuracy is None:
            continue

        if version not in data:
            data[version] = {}

        if update_type not in data[version]:
            data[version][update_type] = {}

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

        data[
            version
        ][
            update_type
        ][
            model
        ] = accuracy

    return data


# ============================================================
# Accuracy: print discovered data
# ============================================================

def print_data(data):

    print("\n" + "=" * 70)
    print("DISCOVERED BENCHMARK DATA")
    print("=" * 70)

    update_types = [
        "base",
        "quant",
        "bpatch",
        "svd_quant_0.7",
        "svd_quant_0.8",
        "svd_quant_0.9",
        "svd_quant_0.95",
        "svd_quant_0.97",
        "svd_quant_0.98",
        "svd_quant_0.99",
        "svd_quant_1",
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

                if model in data[
                    version
                ][
                    update_type
                ]:

                    accuracy = (
                        data[
                            version
                        ][
                            update_type
                        ][
                            model
                        ]
                    )

                    print(
                        f"    {model:8s}: "
                        f"{accuracy:.6f}"
                    )


# ============================================================
# Fused plot
# ============================================================

def plot(accuracy_data):

    # ========================================================
    # Load update-size data
    # ========================================================

    quant_update_sizes = {
        model:
            get_quant_update_sizes(model)
        for model in MODELS
    }

    bpatch_update_sizes = {
        model:
            get_bpatch_update_sizes(model)
        for model in MODELS
    }

    full_update_sizes = {
        model:
            get_full_update_sizes(model)
        for model in MODELS
    }

    svd_update_sizes = {
        (model, energy):
            get_svd_update_sizes(
                energy,
                model
            )
        for model in MODELS
        for energy in SVD_INFORMATION
    }

    svd_quant_update_sizes = {
        (model, energy):
            get_svd_quant_update_sizes(
                energy,
                model
            )
        for model in MODELS
        for energy in SVD_INFORMATION
    }

    # ========================================================
    # Plot setup
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(28, 12)
    )

    ax_accuracy = ax.twinx()

    # ========================================================
    # Position containers
    # ========================================================

    quant_positions = []
    quant_values = []

    bpatch_positions = []
    bpatch_values = []

    full_positions = []
    full_values = []

    svd_positions = {
        energy: []
        for energy in SVD_INFORMATION
    }

    svd_values = {
        energy: []
        for energy in SVD_INFORMATION
    }

    svd_groups = []

    svd_quant_positions = {
        energy: []
        for energy in SVD_INFORMATION
    }

    svd_quant_values = {
        energy: []
        for energy in SVD_INFORMATION
    }

    svd_quant_groups = []

    model_boundaries = []

    position_map = {}

    current_pos = 0

    # ========================================================
    # Build x-axis
    # ========================================================

    for model in MODELS:

        model_start = current_pos

        # ----------------------------------------------------
        # Quantization
        # ----------------------------------------------------

        current_pos += 1

        quant_values_model = [
            value
            for value in quant_update_sizes[model]
            if value is not None
        ]

        if quant_values_model:

            quant_average = (
                sum(quant_values_model)
                / len(quant_values_model)
            )

            quant_positions.append(
                current_pos
            )

            quant_values.append(
                quant_average
            )

            position_map[
                (model, "quant")
            ] = current_pos

            current_pos += 2

        # ----------------------------------------------------
        # SVD + Quantization
        #
        # Placed entirely to the left of the default SVD
        # block below, for this model.
        # ----------------------------------------------------

        for energy in SVD_INFORMATION:

            values = svd_quant_update_sizes[
                (model, energy)
            ]

            valid_values = [
                value
                for value in values
                if value is not None
            ]

            if not valid_values:
                continue

            average = (
                sum(valid_values)
                / len(valid_values)
            )

            minimum = min(valid_values)
            maximum = max(valid_values)

            svd_quant_positions[
                energy
            ].append(current_pos)

            svd_quant_values[
                energy
            ].append(average)

            position_map[
                (model, "svd_quant", energy)
            ] = current_pos

            svd_quant_groups.append({
                "model": model,
                "energy": energy,
                "position": current_pos,
                "average": average,
                "minimum": minimum,
                "maximum": maximum,
            })

            current_pos += 1

            # Gap between SVD+Quant energy levels.
            current_pos += 1

        # ----------------------------------------------------
        # SVD
        # ----------------------------------------------------

        for energy in SVD_INFORMATION:

            values = svd_update_sizes[
                (model, energy)
            ]

            valid_values = [
                value
                for value in values
                if value is not None
            ]

            if not valid_values:
                continue

            average = (
                sum(valid_values)
                / len(valid_values)
            )

            minimum = min(valid_values)
            maximum = max(valid_values)

            svd_positions[
                energy
            ].append(current_pos)

            svd_values[
                energy
            ].append(average)

            position_map[
                (model, "svd", energy)
            ] = current_pos

            svd_groups.append({
                "model": model,
                "energy": energy,
                "position": current_pos,
                "average": average,
                "minimum": minimum,
                "maximum": maximum,
            })

            current_pos += 1

            # Gap between SVD energy levels.
            current_pos += 1

        # ----------------------------------------------------
        # BPatch
        # ----------------------------------------------------

        bpatch_values_model = [
            value
            for value in bpatch_update_sizes[model]
            if value is not None
        ]

        if bpatch_values_model:

            bpatch_average = (
                sum(bpatch_values_model)
                / len(bpatch_values_model)
            )

            bpatch_positions.append(
                current_pos
            )

            bpatch_values.append(
                bpatch_average
            )

            position_map[
                (model, "bpatch")
            ] = current_pos

            current_pos += 1

        # ----------------------------------------------------
        # Full
        # ----------------------------------------------------

        full_values_model = [
            value
            for value in full_update_sizes[model]
            if value is not None
        ]

        if full_values_model:

            full_average = (
                sum(full_values_model)
                / len(full_values_model)
            )

            full_positions.append(
                current_pos
            )

            full_values.append(
                full_average
            )

            position_map[
                (model, "full")
            ] = current_pos

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

        current_pos += 2

    # ========================================================
    # Draw bars
    # ========================================================

    quant_bars = ax.bar(
        quant_positions,
        quant_values,
        color="tab:cyan",
        width=0.8,
        alpha=0.75,
        label="Quantized Update",
    )

    bpatch_bars = ax.bar(
        bpatch_positions,
        bpatch_values,
        color="tab:purple",
        width=0.8,
        alpha=0.75,
        label="BPatch Update",
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

    svd_quant_colors = {
        "0.7": "#a8e6a1",
        "0.8": "#8fdd85",
        "0.9": "#71d268",
        "0.95": "#54c74d",
        "0.97": "#48c241",
        "0.98": "#3bbd34",
        "0.99": "#2cb624",
        "1": "#1fae14",
    }

    svd_bars = {}

    for energy in SVD_INFORMATION:

        svd_bars[energy] = ax.bar(
            svd_positions[energy],
            svd_values[energy],
            color=svd_colors[energy],
            width=0.8,
            alpha=0.75,
            label=(
                f"SVD "
                f"{float(energy) * 100:.0f}% "
                f"Information"
            ),
        )

    svd_quant_bars = {}

    for energy in SVD_INFORMATION:

        svd_quant_bars[energy] = ax.bar(
            svd_quant_positions[energy],
            svd_quant_values[energy],
            color=svd_quant_colors[energy],
            width=0.8,
            alpha=0.75,
            label=(
                f"SVD+Quant "
                f"{float(energy) * 100:.0f}% "
                f"Information"
            ),
        )

    # ========================================================
    # SVD / SVD+Quant min/max markers
    # ========================================================

    for group in svd_groups + svd_quant_groups:

        position = group["position"]

        minimum = group["minimum"]
        maximum = group["maximum"]

        left = position - 0.4
        right = position + 0.4

        ax.hlines(
            minimum,
            left,
            right,
            colors="black",
            linewidth=1,
            linestyles="--",
            label="_nolegend_",
        )

        ax.hlines(
            maximum,
            left,
            right,
            colors="black",
            linewidth=1,
            linestyles="--",
            label="_nolegend_",
        )

        ax.vlines(
            position,
            minimum,
            maximum,
            colors="black",
            linewidth=1,
            label="_nolegend_",
        )

    # ========================================================
    # Model separators
    # ========================================================

    for boundary in model_boundaries[:-1]:

        separator_position = (
            boundary["end"] + 1.5
        )

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

        center = (
            boundary["start"]
            + boundary["end"]
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
    # Accuracy lines
    # ========================================================

    model_colors = {
        "cnn": "tab:blue",
        "cnn1d": "tab:orange",
        "crnn": "tab:green",
        "ds_cnn": "tab:red",
        "gru": "tab:purple",
        "mlp": "tab:brown",
        "svdf": "tab:pink",
    }

    version_linestyles = {
        0: "-",
        1: "--",
        2: "-.",
        3: ":",
    }

    accuracy_x = {
        (model, version): []
        for model in MODELS
        for version in BASE_VERSIONS
    }

    accuracy_y = {
        (model, version): []
        for model in MODELS
        for version in BASE_VERSIONS
    }

    # ========================================================
    # Build accuracy lines
    # ========================================================

    for model in MODELS:

        for version in BASE_VERSIONS:

            # ------------------------------------------------
            # Quantization
            #
            # Quant update generated from version N is
            # evaluated using its own benchmark for version N.
            # ------------------------------------------------

            key = (
                model,
                "quant"
            )

            if key in position_map:

                accuracy = (
                    accuracy_data
                    .get(version+1, {})
                    .get("quant", {})
                    .get(model)
                )

                if accuracy is not None:

                    accuracy_x[
                        (model, version)
                    ].append(
                        position_map[key]
                    )

                    accuracy_y[
                        (model, version)
                    ].append(
                        accuracy
                    )

            # ------------------------------------------------
            # SVD + Quantization points
            # ------------------------------------------------

            for energy in SVD_INFORMATION:

                key = (
                    model,
                    "svd_quant",
                    energy
                )

                if key not in position_map:
                    continue

                update_type = f"svd_quant_{energy}"

                accuracy = (
                    accuracy_data
                    .get(version, {})
                    .get(update_type, {})
                    .get(model)
                )

                if accuracy is None:
                    continue

                accuracy_x[
                    (model, version)
                ].append(
                    position_map[key]
                )

                accuracy_y[
                    (model, version)
                ].append(
                    accuracy
                )

            # ------------------------------------------------
            # SVD points
            # ------------------------------------------------

            for energy in SVD_INFORMATION:

                key = (
                    model,
                    "svd",
                    energy
                )

                if key not in position_map:
                    continue

                update_type = f"svd_{energy}"

                accuracy = (
                    accuracy_data
                    .get(version, {})
                    .get(update_type, {})
                    .get(model)
                )

                if accuracy is None:
                    continue

                accuracy_x[
                    (model, version)
                ].append(
                    position_map[key]
                )

                accuracy_y[
                    (model, version)
                ].append(
                    accuracy
                )

            # ------------------------------------------------
            # BPatch / Full
            #
            # These updates reconstruct the NEXT version.
            # Therefore use the BASE accuracy of version N+1.
            #
            # v0 -> v1 base accuracy
            # v1 -> v2 base accuracy
            # v2 -> v3 base accuracy
            # v3 -> v4 base accuracy
            # ------------------------------------------------

            next_version = version + 1

            for update_type in [
                "bpatch",
                "full",
            ]:

                key = (
                    model,
                    update_type
                )

                if key not in position_map:
                    continue

                accuracy = (
                    accuracy_data
                    .get(next_version, {})
                    .get("base", {})
                    .get(model)
                )

                if accuracy is None:
                    continue

                accuracy_x[
                    (model, version)
                ].append(
                    position_map[key]
                )

                accuracy_y[
                    (model, version)
                ].append(
                    accuracy
                )

    # ========================================================
    # Plot accuracy lines
    # ========================================================

    for model in MODELS:

        for version in BASE_VERSIONS:

            points = sorted(
                zip(
                    accuracy_x[
                        (model, version)
                    ],
                    accuracy_y[
                        (model, version)
                    ]
                ),
                key=lambda item: item[0]
            )

            if not points:
                continue

            x_values = [
                point[0]
                for point in points
            ]

            y_values = [
                point[1]
                for point in points
            ]

            ax_accuracy.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2,
                markersize=6,
                color=model_colors[model],
                linestyle=version_linestyles[version],
                label=f"{model} v{version}",
                zorder=10,
            )

    # ========================================================
    # Axes
    # ========================================================

    ax.set_xlabel(
        "Model / Update",
        fontsize=18,
        labelpad=20,
    )

    ax.set_ylabel(
        "Update Size (KiB)",
        fontsize=18,
    )

    ax_accuracy.set_ylabel(
        "Validation Accuracy",
        fontsize=18,
    )

    # ========================================================
    # X-axis labels
    # ========================================================

    all_positions = []
    all_labels = []

    for energy in SVD_INFORMATION:

        all_positions.extend(
            svd_quant_positions[energy]
        )

        all_labels.extend(
            [
                f"SVD+Q {float(energy) * 100:.0f}%"
            ]
            * len(svd_quant_positions[energy])
        )

    for energy in SVD_INFORMATION:

        all_positions.extend(
            svd_positions[energy]
        )

        all_labels.extend(
            [
                f"SVD {float(energy) * 100:.0f}%"
            ]
            * len(svd_positions[energy])
        )

    all_positions.extend(
        quant_positions
    )

    all_labels.extend(
        ["Quant"] * len(quant_positions)
    )

    all_positions.extend(
        bpatch_positions
    )

    all_labels.extend(
        ["BPatch"] * len(bpatch_positions)
    )

    all_positions.extend(
        full_positions
    )

    all_labels.extend(
        ["Full"] * len(full_positions)
    )

    x_axis_data = sorted(
        zip(
            all_positions,
            all_labels
        ),
        key=lambda item: item[0]
    )

    ax.set_xticks(
        [
            item[0]
            for item in x_axis_data
        ]
    )

    ax.set_xticklabels(
        [
            item[1]
            for item in x_axis_data
        ],
        rotation=90,
        fontsize=8,
    )

    # ========================================================
    # Accuracy scale
    # ========================================================

    all_accuracy_values = []

    for model in MODELS:

        for version in BASE_VERSIONS:

            all_accuracy_values.extend(
                accuracy_y[
                    (model, version)
                ]
            )

    if all_accuracy_values:

        min_accuracy = min(
            all_accuracy_values
        )

        max_accuracy = max(
            all_accuracy_values
        )

        if max_accuracy <= 1.0:

            padding = max(
                0.01,
                (max_accuracy - min_accuracy)
                * 0.15
            )

            ax_accuracy.set_ylim(
                max(
                    0,
                    min_accuracy - padding
                ),
                min(
                    1,
                    max_accuracy + padding
                )
            )

        else:

            padding = max(
                1.0,
                (max_accuracy - min_accuracy)
                * 0.15
            )

            ax_accuracy.set_ylim(
                max(
                    0,
                    min_accuracy - padding
                ),
                min(
                    100,
                    max_accuracy + padding
                )
            )

    # ========================================================
    # Update-size legend
    # ========================================================

    update_legend = ax.legend(
        handles=[
            *[
                svd_quant_bars[energy]
                for energy in SVD_INFORMATION
            ],
            *[
                svd_bars[energy]
                for energy in SVD_INFORMATION
            ],
            quant_bars,
            bpatch_bars,
            full_bars,
        ],
        labels=[
            *[
                f"SVD+Quant "
                f"{float(energy) * 100:.0f}% "
                f"Information"
                for energy in SVD_INFORMATION
            ],
            *[
                f"SVD "
                f"{float(energy) * 100:.0f}% "
                f"Information"
                for energy in SVD_INFORMATION
            ],
            "Quantized Update",
            "BPatch Update",
            "Full Update",
        ],
        loc="upper right",
    )

    ax.add_artist(update_legend)

    # ========================================================
    # Accuracy legend
    # ========================================================

    accuracy_handles = []

    for model in MODELS:

        for version in BASE_VERSIONS:

            if not accuracy_x[
                (model, version)
            ]:
                continue

            accuracy_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=model_colors[model],
                    marker="o",
                    linestyle=version_linestyles[version],
                    linewidth=2,
                    label=f"{model} v{version}",
                )
            )

    ax_accuracy.legend(
        handles=accuracy_handles,
        title="Accuracy Model / Version",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )

    # ========================================================
    # Grid
    # ========================================================

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.set_axisbelow(True)

    # ========================================================
    # Layout
    # ========================================================

    plt.tight_layout(
        pad=2.0
    )

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Plot model update sizes together with "
            "validation accuracy for SVD, SVD+Quant, "
            "Quantization, BPatch, and Full model updates."
        )
    )

    parser.add_argument(
        "--svd_energy",
        type=float,
        default=1.0,
        help=(
            "Retained for compatibility. "
            "All configured SVD energies are plotted."
        ),
    )

    args = parser.parse_args()

    accuracy_data = load_benchmarks()

    print_data(
        accuracy_data
    )

    plot(
        accuracy_data
    )