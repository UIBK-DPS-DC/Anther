#!/usr/bin/env python3

"""
Run the complete SVD benchmark pipeline.

For every model architecture:

    1. Take the source .pt checkpoint.
    2. Find the corresponding SVD update file (.json or .bin).
    3. Apply the SVD update using apply_svd.py.
    4. Benchmark the resulting .pt using train.py.
    5. Store the updated model and benchmark JSON.

Usage examples:

    JSON:
        python benchmark_svd.py --svd-information 0.95

    Binary:
        python benchmark_svd.py --svd-information 0.95 --format bin
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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

VERSIONS = [
    ("version_0_9333", "version_1_27999"),
    ("version_1_27999", "version_2_46666"),
    ("version_2_46666", "version_3_65333"),
    ("version_3_65333", "version_4_93333"),
]


def run_command(command: list[str]) -> None:
    """Run a subprocess and fail immediately if it fails."""

    print()
    print("=" * 80)
    print("RUNNING:")
    print(" ".join(str(x) for x in command))
    print("=" * 80)

    subprocess.run(command, check=True)


def find_svd_file(
    svd_dir: Path,
    model: str,
    source_version: str,
    target_version: str,
    file_format: str,
) -> Path:
    """
    Locate the SVD update file for a model/version transition.

    file_format:
        "json" -> search for .json
        "bin"  -> search for .bin
    """

    extension = f".{file_format}"

    expected = (
        svd_dir
        / model
        / f"{model}_{source_version}_to_{target_version}{extension}"
    )

    if expected.exists():
        return expected

    # Fall back to recursive search in case the naming is slightly different.
    pattern = (
        f"*{source_version}*to*{target_version}*{extension}"
    )

    candidates = list(
        (svd_dir / model).glob(pattern)
    )

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find SVD file for:\n"
            f"  model={model}\n"
            f"  transition={source_version} -> {target_version}\n"
            f"  format={file_format}\n"
            f"  searched={svd_dir / model}\n"
            f"  expected={expected}"
        )

    raise RuntimeError(
        f"Found multiple possible SVD files for "
        f"{model} {source_version}->{target_version} "
        f"with format '{file_format}':\n"
        + "\n".join(f"  {x}" for x in candidates)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply and benchmark all SVD model updates."
    )

    parser.add_argument(
        "--svd-information",
        type=str,
        required=True,
        help="SVD information directory, e.g. 0.95",
    )

    parser.add_argument(
        "--format",
        choices=["json", "bin"],
        default="json",
        help=(
            "SVD update file format. "
            "Use 'json' for JSON triplets or 'bin' for binary triplets. "
            "Default: json."
        ),
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing original model checkpoints.",
    )

    parser.add_argument(
        "--svd-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing svd_out_X directories. "
            "Defaults to svd_out_<information>."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Benchmark output directory. "
            "Defaults to benchmark_svd_partial_update_<information>."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/speech_commands"),
        help="Speech Commands dataset directory.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda", "auto"],
    )

    parser.add_argument(
        "--apply-script",
        type=Path,
        default=Path("apply_svd_quant.py"),
    )

    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("train.py"),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate existing updated models and benchmarks.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help="Models to benchmark.",
    )

    args = parser.parse_args()

    information = args.svd_information

    svd_root = (
        args.svd_root
        if args.svd_root is not None
        else Path(f"svd_out_{information}_quant2")
    )

    output_root = (
        args.output_root
        if args.output_root is not None
        else Path(
            f"benchmark_svd_partial_update_{information}_quant2"
            # f"_{args.format}"
        )
    )

    print()
    print("=" * 80)
    print("SVD BENCHMARK PIPELINE")
    print("=" * 80)
    print(f"SVD information:       {information}")
    print(f"SVD format:            {args.format}")
    print(f"Runs directory:        {args.runs_dir}")
    print(f"SVD directory:         {svd_root}")
    print(f"Output directory:      {output_root}")
    print(f"Data directory:        {args.data_dir}")
    print(f"Device:                {args.device}")
    print(f"Models:                {args.models}")
    print("=" * 80)

    total = 0
    skipped = 0

    for model in args.models:
        print()
        print("#" * 80)
        print(f"# MODEL: {model}")
        print("#" * 80)

        model_output_dir = output_root / model
        model_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for source_version, target_version in VERSIONS:

            print()
            print("-" * 80)
            print(
                f"{model}: "
                f"{source_version} -> {target_version}"
            )
            print("-" * 80)

            source_pt = (
                args.runs_dir
                / model
                / f"{source_version}.pt"
            )

            if not source_pt.exists():
                raise FileNotFoundError(
                    f"Source model does not exist:\n  {source_pt}"
                )

            svd_file = find_svd_file(
                svd_root,
                model,
                source_version,
                target_version,
                args.format,
            )

            output_pt = (
                model_output_dir
                / f"{source_version}_svd_updated.pt"
            )

            benchmark_json = (
                model_output_dir
                / f"{source_version}_svd_updated_benchmark.json"
            )

            print(f"Source model: {source_pt}")
            print(f"SVD updates:  {svd_file}")
            print(f"Format:       {args.format}")
            print(f"Updated model: {output_pt}")
            print(f"Benchmark:     {benchmark_json}")

            # -----------------------------------------------------------------
            # Step 1: Apply SVD
            # -----------------------------------------------------------------

            if output_pt.exists() and not args.force:
                print()
                print(
                    "[1/2] Updated model already exists; "
                    "skipping SVD application."
                )
                skipped += 1

            else:
                output_pt.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                run_command(
                    [
                        sys.executable,
                        str(args.apply_script),
                        str(source_pt),
                        str(svd_file),
                        "--output",
                        str(output_pt),
                    ]
                )

            # -----------------------------------------------------------------
            # Step 2: Benchmark updated model
            # -----------------------------------------------------------------

            if benchmark_json.exists() and not args.force:
                print()
                print(
                    "[2/2] Benchmark already exists; "
                    "skipping evaluation."
                )
                skipped += 1

            else:
                run_command(
                    [
                        sys.executable,
                        str(args.train_script),
                        "--data-dir",
                        str(args.data_dir),
                        "--out-dir",
                        str(model_output_dir),
                        "--device",
                        args.device,
                        "--benchmark-model",
                        str(output_pt),
                        "--benchmark-output",
                        benchmark_json.name,
                    ]
                )

            total += 1

    print()
    print("=" * 80)
    print("SVD BENCHMARK PIPELINE FINISHED")
    print("=" * 80)
    print(f"Processed transitions: {total}")
    print(f"Skipped existing:      {skipped}")
    print(f"Format:                {args.format}")
    print(f"Output:                {output_root.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()