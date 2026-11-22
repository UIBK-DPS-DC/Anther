#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Accuracy loading
# ---------------------------------------------------------------------------

def load_summary(model_dir):
    """Load summary.json from a trained-model/accuracy directory."""
    summary_path = Path(model_dir) / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find {summary_path}")

    with open(summary_path, "r") as f:
        return json.load(f)


def find_summary_files(root):
    """
    Find summary.json files recursively below root.

    This is useful for runs/... where the exact run subdirectory can vary.
    """
    root = Path(root)
    if root.is_file() and root.name == "summary.json":
        return [root]
    if not root.exists():
        return []
    return sorted(root.rglob("summary.json"))


def get_version_number(version_key, version_info):
    """
    Determine the numerical model version.

    Prefer the checkpoint name, e.g. version_1_27999.pt, and fall back
    to the summary key when it is numeric.
    """
    checkpoint = str(version_info.get("checkpoint", ""))

    match = re.search(r"version[_-]?(\d+)[_-]", checkpoint, re.IGNORECASE)
    if match:
        return int(match.group(1))

    try:
        return int(version_key)
    except (TypeError, ValueError):
        return None


def parse_summary(summary):
    """
    Convert a summary into:

        {
            architecture: {
                version_number: {
                    "val_accuracy": ...,
                    "test_accuracy": ...
                }
            }
        }
    """
    result = {}

    for architecture, architecture_data in summary.items():
        if not isinstance(architecture_data, dict):
            continue

        versions = architecture_data.get("versions", {})
        if not isinstance(versions, dict):
            continue

        result.setdefault(architecture, {})

        for version_key, version_info in versions.items():
            if not isinstance(version_info, dict):
                continue

            version_number = get_version_number(version_key, version_info)

            if version_number is None:
                print(
                    f"Warning: Could not determine version number for "
                    f"{architecture}/{version_key}"
                )
                continue

            result[architecture][version_number] = {
                key: version_info[key]
                for key in ("val_accuracy", "test_accuracy")
                if key in version_info
            }

    return result


def merge_parsed_summaries(parsed_summaries):
    """
    Merge multiple parsed summaries.

    Later summaries overwrite an identical architecture/version pair.
    """
    merged = {}

    for parsed in parsed_summaries:
        for architecture, versions in parsed.items():
            merged.setdefault(architecture, {})
            merged[architecture].update(versions)

    return merged


def load_accuracy_data(root, label="accuracy root"):
    """
    Load one summary.json or all summary.json files below root.

    This supports both:
      evaluation/trained_models_LowRA_rank_X/summary.json
    and:
      evaluation/runs/<some_run>/.../summary.json
    """
    root = Path(root)

    if root.is_file():
        summary_files = [root]
    else:
        summary_files = find_summary_files(root)

    if not summary_files:
        raise FileNotFoundError(
            f"No summary.json found below {root} ({label})."
        )

    parsed = []
    for summary_path in summary_files:
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            parsed.append(parse_summary(summary))
        except Exception as exc:
            print(f"Warning: Could not read {summary_path}: {exc}")

    merged = merge_parsed_summaries(parsed)

    if not merged:
        raise RuntimeError(f"No usable accuracy data found below {root}.")

    return merged

def load_benchmark_accuracy_data(root, label="benchmark accuracy root"):
    """
    Load accuracy data from *_benchmark.json files.

    Expected layout:

        root/
            cnn/
                version_0_9333_benchmark.json
                version_1_27999_benchmark.json
                ...
            cnn1d/
                version_0_9333_benchmark.json
                ...

    Each benchmark JSON contains:

        {
            "validation_accuracy": 0.3668,
            "test_accuracy": 0.3510
        }

    Returns:

        {
            "cnn": {
                0: {
                    "val_accuracy": ...,
                    "test_accuracy": ...
                },
                1: {
                    ...
                }
            },
            ...
        }
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(
            f"Could not find {root} ({label})."
        )

    benchmark_files = sorted(root.rglob("*_benchmark.json"))

    if not benchmark_files:
        raise FileNotFoundError(
            f"No *_benchmark.json files found below {root} ({label})."
        )

    result = {}

    for benchmark_path in benchmark_files:
        try:
            with open(benchmark_path, "r") as f:
                benchmark = json.load(f)

            # Architecture is the directory immediately containing
            # the benchmark file, e.g. runs/cnn/...
            architecture = benchmark_path.parent.name

            # Extract the model version from filenames such as:
            #
            # version_0_9333_benchmark.json
            # version_1_27999_benchmark.json
            # version_4_93333_benchmark.json
            match = re.search(
                r"version_(\d+)(?:_|$)",
                benchmark_path.stem
            )

            if not match:
                print(
                    f"Warning: Could not determine version from "
                    f"{benchmark_path}"
                )
                continue

            version = int(match.group(1))

            if "validation_accuracy" not in benchmark:
                print(
                    f"Warning: {benchmark_path} has no "
                    f"'validation_accuracy'"
                )
                continue

            if "test_accuracy" not in benchmark:
                print(
                    f"Warning: {benchmark_path} has no "
                    f"'test_accuracy'"
                )
                continue

            result.setdefault(architecture, {})
            result[architecture][version] = {
                "val_accuracy": benchmark["validation_accuracy"],
                "test_accuracy": benchmark["test_accuracy"],
            }

        except Exception as exc:
            print(
                f"Warning: Could not read {benchmark_path}: {exc}"
            )

    if not result:
        raise RuntimeError(
            f"No usable benchmark accuracy data found below "
            f"{root} ({label})."
        )

    return result

# ---------------------------------------------------------------------------
# Update-file parsing
# ---------------------------------------------------------------------------

def get_file_size_kib(path):
    """Return raw file size in KiB."""
    return path.stat().st_size / 1024.0


def infer_update_pair(path):
    """
    Infer (source_version, target_version) from an update filename.

    Supported examples include:
        delta_0_to_1.lrq
        version_0_to_1.lrq
        delta_0_to_1.bin
        cnn_version_1_27999.bin
        cnn_version_0_9333_to_cnn_version_1_27999.bin
        update_0.bin
        delta_0.lrq

    Returns (source, target), or (None, None).
    """
    name = Path(path).stem.lower()

    # Explicit source -> target:
    #   delta_0_to_1
    #   version_0_to_1
    #   cnn_version_0_9333_to_cnn_version_1_27999
    pairs = re.findall(
        r"version[_-]?(\d+)(?:[_-]\d+)?[_-]?(?:to|->)[_-]?"
        r"(?:[a-z0-9]+[_-])?version[_-]?(\d+)",
        name,
        flags=re.IGNORECASE,
    )
    if pairs:
        return int(pairs[-1][0]), int(pairs[-1][1])

    # Generic "0_to_1".
    match = re.search(r"(?:^|[_-])(\d+)[_-]to[_-](\d+)(?:$|[_-])", name)
    if match:
        return int(match.group(1)), int(match.group(2))

    # update_0 / delta_0 / version_0
    # A single encoded version means "this update starts at version N".
    match = re.search(r"(?:update|delta)[_-]?(\d+)", name)
    if match:
        source = int(match.group(1))
        return source, source + 1

    # A single version in names such as:
    #   cnn_version_1_27999.bin
    # Here the first number is the model version and the second number
    # is the dataset/training-step identifier, not another version.
    match = re.search(r"version[_-]?(\d+)(?:[_-]\d+)?", name)
    if match:
        target = int(match.group(1))
        return target - 1, target

    # Just a number, e.g. 0.lrq
    match = re.fullmatch(r"(\d+)", name)
    if match:
        source = int(match.group(1))
        return source, source + 1

    return None, None


def infer_target_version(path):
    """Return the target version encoded by a full/bpatch/quant filename."""
    source, target = infer_update_pair(path)
    if target is not None:
        return target

    # Extra fallback for names like cnn_version_1_27999.bin.
    name = Path(path).stem.lower()
    matches = re.findall(r"version[_-]?(\d+)", name)
    if matches:
        return int(matches[-1])

    return None


def find_update_files(update_dir, architecture, extensions=None):
    """
    Find update files for one architecture.

    Preferred layout:
        update_dir/<architecture>/*

    A recursive filename search is used as a fallback.
    """
    update_dir = Path(update_dir)
    extensions = extensions or {".lrq", ".bin"}

    architecture_dir = update_dir / architecture
    if architecture_dir.exists():
        files = [
            p for p in architecture_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]
        if files:
            return sorted(files)

    # Some folders may not have architecture subdirectories.
    files = [
        p
        for p in update_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
        and architecture.lower() in p.name.lower()
    ]
    return sorted(files)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def make_point(
    architecture,
    source_version,
    target_version,
    size_kib,
    accuracy,
    series,
    file_path,
    rank=None,
):
    return {
        "architecture": architecture,
        "source_version": source_version,
        "target_version": target_version,
        "version": target_version,
        "rank": rank,
        "size_kib": size_kib,
        "accuracy": accuracy * 100.0,
        "series": series,
        "file": str(file_path),
    }


def collect_lora_data(model_dir, update_dir, accuracy_type, rank):
    """
    Collect LoRA update sizes and pair them with the accuracy of the
    resulting model version.

    Example:
        delta_0_to_1.lrq -> accuracy of version 1
    """
    accuracies = load_accuracy_data(model_dir, f"LoRA rank {rank}")
    data = []

    for architecture, versions in accuracies.items():
        files = find_update_files(update_dir, architecture, {".lrq"})

        if not files:
            print(
                f"Warning: No .lrq files found for architecture "
                f"'{architecture}' in {update_dir}"
            )
            continue

        # Prefer explicit filename version numbers. Unknown names fall back
        # to sorted order 0->1, 1->2, ...
        parsed_files = []
        for path in files:
            source, target = infer_update_pair(path)
            parsed_files.append((source, target, path))

        parsed_files.sort(
            key=lambda item: (
                item[0] is None,
                item[0] if item[0] is not None else 999999,
                str(item[2]),
            )
        )

        for fallback_index, (source, target, path) in enumerate(parsed_files):
            if source is None:
                source = fallback_index
                target = source + 1

            if target not in versions:
                print(
                    f"Warning: Could not match {path} to "
                    f"{architecture} version {source} -> {target} "
                    f"for rank {rank}"
                )
                continue

            if accuracy_type not in versions[target]:
                print(
                    f"Warning: {path} matched version {target}, but "
                    f"{accuracy_type} is missing in the summary."
                )
                continue

            data.append(
                make_point(
                    architecture=architecture,
                    source_version=source,
                    target_version=target,
                    size_kib=get_file_size_kib(path),
                    accuracy=versions[target][accuracy_type],
                    series=f"LoRA rank {rank}",
                    file_path=path,
                    rank=rank,
                )
            )

    return data


def collect_binary_data(
    update_dir,
    accuracy_root,
    accuracy_type,
    series,
    extensions=(".bin",),
    skip_target_versions=None,
):
    """
    Collect ordinary binary update/full-model sizes and pair them with
    model accuracies.

    The accuracy root may contain one summary.json or many summaries under
    runs/. The target version encoded in each filename is used to select
    the corresponding accuracy.
    """
    if series in ["bpatch_out", "full_out", "quant_out"]:
        accuracies = load_benchmark_accuracy_data(
            accuracy_root,
            label="runs benchmark accuracy",
        )
    else:
        accuracies = load_accuracy_data(accuracy_root, series)
    data = []
    extensions = {ext.lower() for ext in extensions}

    architectures = sorted(accuracies)

    for architecture in architectures:
        files = find_update_files(update_dir, architecture, extensions)

        if not files:
            print(
                f"Warning: No matching update files found for architecture "
                f"'{architecture}' in {update_dir}"
            )
            continue

        for path in files:
            target = infer_target_version(path)

            if target is None:
                print(
                    f"Warning: Could not infer a target version from {path}"
                )
                continue

            if skip_target_versions and target in skip_target_versions:
                continue

            if target not in accuracies[architecture]:
                print(
                    f"Warning: Could not match {path} to "
                    f"{architecture} version {target}"
                )
                continue

            if accuracy_type not in accuracies[architecture][target]:
                print(
                    f"Warning: {accuracy_type} missing for "
                    f"{architecture} version {target}"
                )
                continue

            # For full_out files such as cnn_version_1_27999.bin there is
            # no source update in the filename. We still plot them at the
            # corresponding target version. For bpatch/quant files with an
            # explicit pair, retain the source version as well.
            source, explicit_target = infer_update_pair(path)
            if explicit_target != target:
                source = target - 1

            data.append(
                make_point(
                    architecture=architecture,
                    source_version=source,
                    target_version=target,
                    size_kib=get_file_size_kib(path),
                    accuracy=accuracies[architecture][target][accuracy_type],
                    series=series,
                    file_path=path,
                )
            )

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_data(all_data: list, accuracy_type, log_x=True, output=None):
    """Create a scatter plot with color = data source and marker = architecture."""
    import numpy as np
    from collections import defaultdict

    if not all_data:
        raise RuntimeError("No data points were found.")

    # all_data = list(filter(lambda x: not d["series"].startswith("LoRA") or x["rank"] == 8, all_data))
    all_data = list(filter(lambda x: x["target_version"] == 4, all_data))
    for d in all_data:
        d['series'] = d['series'].replace('_out', '')

    architectures = sorted({d["architecture"] for d in all_data})

    markers = [
        "o",   # circle
        "s",   # square
        "^",   # triangle
        "D",   # diamond
        "v",   # down triangle
        "P",   # plus-filled
        "X",   # x-filled
        "<",
        ">",
        "p",
        "h",
    ]

    architecture_markers = {
        architecture: markers[i % len(markers)]
        for i, architecture in enumerate(architectures)
    }

    # Stable, visually distinct colors. One color = one data source.
    series_order = [
        "LoRA rank 2",
        "LoRA rank 4",
        "LoRA rank 8",
        "BPatch",
        "Full",
        "Quant INT8",
    ]
    available_series = [s for s in series_order if any(d["series"] == s for d in all_data)]
    extra_series = sorted(
        {d["series"] for d in all_data if d["series"] not in series_order}
    )
    available_series.extend(extra_series)

    palette = plt.get_cmap("tab10")
    series_colors = {
        series: palette(i % 10)
        for i, series in enumerate(available_series)
    }

    fig, ax = plt.subplots(figsize=(13, 8))

    # Collect and deterministically jitter only exact overlaps.
    overlapping_groups = defaultdict(list)
    for i, d in enumerate(all_data):
        key = (
            round(d["size_kib"], 6),
            round(d["accuracy"], 6),
        )
        overlapping_groups[key].append(i)

    x_jitter_factor = 0.025
    y_jitter = 0.07

    x_plot = [d["size_kib"] for d in all_data]
    y_plot = [d["accuracy"] for d in all_data]

    for indices in overlapping_groups.values():
        if len(indices) == 1:
            continue

        n = len(indices)
        offsets = np.arange(n) - (n - 1) / 2

        for offset, idx in zip(offsets, indices):
            x_plot[idx] *= 1 + offset * x_jitter_factor
            y_plot[idx] += offset * y_jitter

    # Plot one group per series. Architecture is encoded by marker.
    for series in available_series:
        for architecture in architectures:
            indices = [
                i
                for i, d in enumerate(all_data)
                if d["series"] == series
                and d["architecture"] == architecture
            ]

            if not indices:
                continue

            x = [x_plot[i] for i in indices]
            y = [y_plot[i] for i in indices]

            ax.scatter(
                x,
                y,
                marker=architecture_markers[architecture],
                s=150,
                color=series_colors[series],
                alpha=0.78,
                edgecolors="black",
            )

            # for idx in indices:
            #     d = all_data[idx]

            #     if d["series"].startswith("LoRA"):
            #         annotation = (
            #             f"{d['architecture']}"
            #             f" r{d['rank']}"
            #             # f" v{d['target_version']}"
            #         )
            #     else:
            #         annotation = (
            #             f"{d['architecture']}"
            #             f" {d['series']}"
            #             # f" v{d['target_version']}"
            #         )

                # if annotation == "mlp bpatch":
                #     ax.annotate(
                #         annotation,
                #         (x_plot[idx], y_plot[idx]),
                #         xytext=(-60, -25),
                #         textcoords="offset points",
                #         fontsize=18,
                #         alpha=0.85,
                #     )
                # elif annotation.__contains__("ull") or annotation == "crnn r8":
                #     ax.annotate(
                #         annotation,
                #         (x_plot[idx], y_plot[idx]),
                #         xytext=(-30, 10),
                #         textcoords="offset points",
                #         fontsize=18,
                #         alpha=0.85,
                #     )
                # else:
                #     ax.annotate(
                #         annotation,
                #         (x_plot[idx], y_plot[idx]),
                #         xytext=(-30, -20),
                #         textcoords="offset points",
                #         fontsize=18,
                #         alpha=0.85,
                #     )

    ax.set_xlabel("Size of TinyML update (KiB)", fontsize=25)

    accuracy_label = (
        "Validation accuracy (%)"
        if accuracy_type == "val_accuracy"
        else "Test accuracy (%)"
    )
    ax.set_ylabel(accuracy_label, fontsize=25)

    # ax.set_title(
    #     "TinyML Update Size vs. Model Accuracy\n"
    #     f"{'Validation' if accuracy_type == 'val_accuracy' else 'Test'} accuracy",
    #     fontsize=15,
    # )

    if log_x:
        ax.set_xscale("log")
        ticks = []
        i = 0
        base_numbers = [x * 10**i for x in range(1, 10)]
        tick = base_numbers.pop(0)
        while tick < max(x_plot):
            ticks.append(tick)
            if len(base_numbers) == 0:
                i += 1
                base_numbers = [x * 10**i for x in range(1, 10)]
            tick = base_numbers.pop(0)
        ticks.append(tick)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=-90)
        ax.tick_params(labelsize=18)

    ax.grid(
        True,
        which="both" if log_x else "major",
        alpha=0.25,
    )

    # Separate legends make the encoding immediately understandable:
    # color = source/rank, marker = architecture.
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=series_colors[s],
            markeredgecolor="black",
            markersize=8,
            label=s,
        )
        for s in available_series
    ]

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=architecture_markers[a],
            linestyle="",
            color="black",
            markerfacecolor="white",
            markersize=8,
            label=a,
        )
        for a in architectures
    ]

    legend1 = ax.legend(
        handles=color_handles,
        title="Update Method",
        bbox_to_anchor=(1.02, 1.00),
        loc="upper left",
        fontsize=22,
        title_fontsize=22,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=marker_handles,
        title="Model architecture",
        bbox_to_anchor=(1.02, 0.47),
        loc="upper left",
        fontsize=22,
        title_fontsize=22,
    )

    fig.tight_layout()

    if output:
        fig.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved plot to: {output}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot TinyML update size vs. model accuracy for LoRA ranks "
            "2/4/8 and bpatch/full/quant update sources."
        )
    )

    parser.add_argument(
        "--root",
        default="../",
        help="Evaluation directory containing all mentioned folders.",
    )

    parser.add_argument(
        "--models-rank2",
        default="lora/trained_models_LowRA_rank_2",
        help="Rank-2 trained-model directory.",
    )
    parser.add_argument(
        "--models-rank4",
        default="lora/trained_models_LowRA_rank_4",
        help="Rank-4 trained-model directory.",
    )
    parser.add_argument(
        "--models-rank8",
        default="lora/trained_models_LowRA_rank_8",
        help="Rank-8 trained-model directory.",
    )

    parser.add_argument(
        "--lora-update-rank2",
        default="lora/lora_update_rank_2_bits_8",
        help="Rank-2 LoRA update directory.",
    )
    parser.add_argument(
        "--lora-update-rank4",
        default="lora/lora_update_rank_4_bits_8",
        help="Rank-4 LoRA update directory.",
    )
    parser.add_argument(
        "--lora-update-rank8",
        default="lora/lora_update_rank_8_bits_8",
        help="Rank-8 LoRA update directory.",
    )

    parser.add_argument(
        "--bpatch-out",
        default="bpatch_method/bpatch_out",
        help="Binary bpatch update-size directory.",
    )
    parser.add_argument(
        "--full-out",
        default="full/full_out",
        help="Full-model binary size directory.",
    )
    parser.add_argument(
        "--quant-out",
        default="quant/quant_out",
        help="Quantized update-size directory.",
    )

    parser.add_argument(
        "--runs",
        default="runs",
        help=(
            "Directory containing the accuracies for bpatch_out/full_out. "
            "All summary.json files below it are searched."
        ),
    )
    parser.add_argument(
        "--benchmark-quant-update",
        default="quant/benchmark_quant_update",
        help=(
            "Directory containing the accuracies for quant_out. "
            "All summary.json files below it are searched."
        ),
    )

    parser.add_argument(
        "--accuracy",
        choices=["val", "test"],
        default="test",
        help="Which accuracy to plot. Default: test.",
    )

    parser.add_argument(
        "--linear-x",
        action="store_true",
        help="Use a linear x-axis instead of logarithmic.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output filename, e.g. tinyml_scatter.png",
    )

    args = parser.parse_args()

    root = Path(args.root)

    def under_root(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    accuracy_type = (
        "val_accuracy"
        if args.accuracy == "val"
        else "test_accuracy"
    )

    # ------------------------------------------------------------
    # LoRA: rank-specific accuracy + rank-specific .lrq update sizes
    # ------------------------------------------------------------
    datasets = []

    lora_configs = [
        (2, args.models_rank2, args.lora_update_rank2),
        (4, args.models_rank4, args.lora_update_rank4),
        (8, args.models_rank8, args.lora_update_rank8),
    ]

    for rank, model_dir, update_dir in lora_configs:
        print(f"Reading LoRA rank-{rank} data...")
        data = collect_lora_data(
            under_root(model_dir),
            under_root(update_dir),
            accuracy_type,
            rank,
        )
        print(f"  LoRA rank-{rank} points: {len(data)}")
        datasets.extend(data)

    # ------------------------------------------------------------
    # bpatch + full: both use the same accuracies from runs/...
    # ------------------------------------------------------------
    print("Reading bpatch_out data...")
    bpatch_data = collect_binary_data(
        under_root(args.bpatch_out),
        under_root(args.runs),
        accuracy_type,
        "bpatch_out",
        extensions=(".bin",),
    )
    print(f"  bpatch_out points: {len(bpatch_data)}")
    datasets.extend(bpatch_data)

    print("Reading full_out data...")
    full_data = collect_binary_data(
        under_root(args.full_out),
        under_root(args.runs),
        accuracy_type,
        "full_out",
        extensions=(".bin",),
        # full_out contains the original version 0 model as well; it is
        # not an update and should not be included in this scatter.
        skip_target_versions={0},
    )
    print(f"  full_out points: {len(full_data)}")
    datasets.extend(full_data)

    # ------------------------------------------------------------
    # quant: quant_out sizes + benchmark_quant_update accuracies
    # ------------------------------------------------------------
    print("Reading quant_out data...")
    quant_data = collect_binary_data(
        under_root(args.quant_out),
        under_root(args.benchmark_quant_update),
        accuracy_type,
        "quant_out",
        extensions=(".bin",),
    )
    print(f"  quant_out points: {len(quant_data)}")
    datasets.extend(quant_data)

    print(f"\nTotal points: {len(datasets)}")

    if not datasets:
        raise RuntimeError(
            "No data points were found. Check the folder paths and "
            "the summary.json / filename version conventions."
        )

    plot_data(
        datasets,
        accuracy_type,
        log_x=not args.linear_x,
        output=args.output,
    )


if __name__ == "__main__":
    main()