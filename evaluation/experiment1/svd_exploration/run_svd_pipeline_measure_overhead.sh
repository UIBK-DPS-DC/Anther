#!/usr/bin/env bash
#
# run_svd_pipeline.sh
#
# Runs svd.py between every pair of consecutive versions within each model
# directory and measures the execution time of every SVD operation.
#
# Usage:
#   ./run_svd_pipeline.sh <input_dir> <output_dir> [path_to_svd.py] [svd_energy]
#
# Example:
#   ./run_svd_pipeline.sh ./runs ./svd_out ./svd.py 1

set -euo pipefail

INPUT_DIR="${1:?Usage: $0 <input_dir> <output_dir> [svd_script]}"
SVD_SCRIPT="${3:-$(dirname "$0")/svd.py}"
SVD_ENERGY="${4:-1}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir> [svd_script]}_${SVD_ENERGY}"

# Log file
LOG_FILE="$OUTPUT_DIR/svd_timing.log"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: input directory '$INPUT_DIR' does not exist." >&2
  exit 1
fi

if [[ ! -f "$SVD_SCRIPT" ]]; then
  echo "Error: svd.py not found at '$SVD_SCRIPT'." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Initialize log
{
  echo "============================================================"
  echo "SVD Pipeline Timing Log"
  echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Input directory: $INPUT_DIR"
  echo "Output directory: $OUTPUT_DIR"
  echo "SVD script: $SVD_SCRIPT"
  echo "SVD energy: $SVD_ENERGY"
  echo "============================================================"
  echo
} > "$LOG_FILE"

model_count=0
pair_count=0
failed_count=0

# Total pipeline timer
pipeline_start=$(date +%s.%N)

# Iterate over each immediate subdirectory (one per model type: cnn, cnn1d, ...)
for model_dir in "$INPUT_DIR"/*/; do
  [[ -d "$model_dir" ]] || continue
  model_name="$(basename "$model_dir")"

  # Collect .pt files in this directory, sorted numerically by the version
  # index embedded in the filename (version_<N>_<metric>.pt), not lexically.
  mapfile -t pt_files < <(
    find "$model_dir" -maxdepth 1 -type f -name '*.pt' \
      | while read -r f; do
          base="$(basename "$f")"
          if [[ "$base" =~ version_([0-9]+)_ ]]; then
            echo "${BASH_REMATCH[1]} $f"
          else
            echo "WARN: skipping unrecognized filename '$f'" >&2
          fi
        done \
      | sort -n -k1,1 \
      | cut -d' ' -f2-
  )

  if [[ "${#pt_files[@]}" -lt 2 ]]; then
    echo "Skipping '$model_name': fewer than 2 recognizable .pt versions found."
    continue
  fi

  model_out_dir="$OUTPUT_DIR/$model_name"
  mkdir -p "$model_out_dir"

  echo "=== $model_name (${#pt_files[@]} versions) ==="
  echo "=== $model_name (${#pt_files[@]} versions) ===" >> "$LOG_FILE"

  model_count=$((model_count + 1))

  for ((i = 0; i < ${#pt_files[@]} - 1; i++)); do
    old_file="${pt_files[$i]}"
    new_file="${pt_files[$((i + 1))]}"

    old_base="$(basename "$old_file" .pt)"
    new_base="$(basename "$new_file" .pt)"

    out_file="$model_out_dir/${model_name}_${old_base}_to_${new_base}.json"

    echo "  [$model_name] $old_base -> $new_base"

    # ------------------------------------------------------------
    # Measure execution time of this individual SVD operation
    # ------------------------------------------------------------
    start_time=$(date +%s.%N)

    if python3 "$SVD_SCRIPT" \
        "$old_file" \
        "$new_file" \
        --output "$out_file" \
        --svd_energy "$SVD_ENERGY"; then

      end_time=$(date +%s.%N)

      elapsed=$(awk "BEGIN {printf \"%.3f\", $end_time - $start_time}")

      pair_count=$((pair_count + 1))

      echo "    Time: ${elapsed} seconds"

      echo "$(date '+%Y-%m-%d %H:%M:%S') | $model_name | $old_base -> $new_base | ${elapsed} seconds | SUCCESS" \
        >> "$LOG_FILE"

    else

      end_time=$(date +%s.%N)

      elapsed=$(awk "BEGIN {printf \"%.3f\", $end_time - $start_time}")

      failed_count=$((failed_count + 1))

      echo "  ERROR: svd.py failed for $old_file -> $new_file" >&2

      echo "$(date '+%Y-%m-%d %H:%M:%S') | $model_name | $old_base -> $new_base | ${elapsed} seconds | FAILED" \
        >> "$LOG_FILE"
    fi
  done

  echo >> "$LOG_FILE"
done

# ------------------------------------------------------------
# Total pipeline time
# ------------------------------------------------------------
pipeline_end=$(date +%s.%N)
total_elapsed=$(awk "BEGIN {printf \"%.3f\", $pipeline_end - $pipeline_start}")

{
  echo "============================================================"
  echo "Pipeline Summary"
  echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Models processed: $model_count"
  echo "Successful SVD operations: $pair_count"
  echo "Failed SVD operations: $failed_count"
  echo "Total pipeline time: ${total_elapsed} seconds"
  echo "============================================================"
} >> "$LOG_FILE"

echo
echo "Done."
echo "Processed $model_count model dirs."
echo "$pair_count SVD diffs written under: $OUTPUT_DIR"
echo "Total pipeline time: ${total_elapsed} seconds"
echo "Timing log: $LOG_FILE"
