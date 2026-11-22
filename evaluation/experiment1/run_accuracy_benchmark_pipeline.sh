#!/usr/bin/env bash
#
# run_benchmark_pipeline.sh
#
# Runs train.py --benchmark-model for every model/version checkpoint
# found in the supplied directory.
#
# Usage:
#   ./run_benchmark_pipeline.sh <benchmark_dir> [path_to_train.py] [device]
#
# Example:
#   ./run_benchmark_pipeline.sh \
#       benchmark_svd_partial_update_1_bin \
#       ./train.py \
#       cuda
#

set -euo pipefail

BENCHMARK_DIR="${1:?Usage: $0 <benchmark_dir> [train_script] [device]}"
TRAIN_SCRIPT="${2:-train.py}"
DEVICE="${3:-cuda}"

DATA_DIR="data/speech_commands/"

if [[ ! -d "$BENCHMARK_DIR" ]]; then
  echo "Error: benchmark directory '$BENCHMARK_DIR' does not exist." >&2
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Error: train.py not found at '$TRAIN_SCRIPT'." >&2
  exit 1
fi

model_count=0
version_count=0

for model_dir in "$BENCHMARK_DIR"/*/; do
  [[ -d "$model_dir" ]] || continue

  model_name="$(basename "$model_dir")"

  mapfile -t pt_files < <(
    find "$model_dir" -maxdepth 1 -type f -name '*.pt' \
      | sort
  )

  if [[ "${#pt_files[@]}" -eq 0 ]]; then
    echo "Skipping '$model_name': no .pt files found."
    continue
  fi

  echo "=== $model_name (${#pt_files[@]} versions) ==="

  model_count=$((model_count + 1))

  for model_file in "${pt_files[@]}"; do

    model_base="$(basename "$model_file" .pt)"

    benchmark_output="${model_base}_benchmark.json"

    echo
    echo "  [$model_name] $model_base"
    echo "    Model:  $model_file"
    echo "    Output: $benchmark_output"

    if python3 "$TRAIN_SCRIPT" \
        --data-dir "$DATA_DIR" \
        --out-dir "$model_dir" \
        --device "$DEVICE" \
        --benchmark-model "$model_file" \
        --benchmark-output "$benchmark_output"; then

      version_count=$((version_count + 1))

    else

      echo "  ERROR: train.py failed for:"
      echo "         $model_file" >&2

    fi

  done
done

echo
echo "Done."
echo "Processed model dirs: $model_count"
echo "Benchmarked versions: $version_count"
echo "Benchmark directory:  $BENCHMARK_DIR"