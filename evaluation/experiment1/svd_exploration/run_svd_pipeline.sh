#!/usr/bin/env bash
#
# run_svd_pipeline.sh
#
# Scans a directory of model subdirectories (each containing .pt version
# checkpoints named like "version_<N>_<metric>.pt") and runs svd_delta.py
# between every pair of consecutive versions within each model directory.
#
# For 5 versions (0..4) per model, this produces 4 diff files per model:
#   0->1, 1->2, 2->3, 3->4
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
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir> [svd_script]}_${SVD_ENERGY}_quant2"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: input directory '$INPUT_DIR' does not exist." >&2
  exit 1
fi

if [[ ! -f "$SVD_SCRIPT" ]]; then
  echo "Error: svd.py not found at '$SVD_SCRIPT'." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

model_count=0
pair_count=0

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
  model_count=$((model_count + 1))

  for ((i = 0; i < ${#pt_files[@]} - 1; i++)); do
    old_file="${pt_files[$i]}"
    new_file="${pt_files[$((i + 1))]}"

    old_base="$(basename "$old_file" .pt)"
    new_base="$(basename "$new_file" .pt)"

    out_file="$model_out_dir/${model_name}_${old_base}_to_${new_base}.bin"

    echo "  [$model_name] $old_base -> $new_base"
    if python3 "$SVD_SCRIPT" "$old_file" "$new_file" --output "$out_file" --svd_information "$SVD_ENERGY"; then
      pair_count=$((pair_count + 1))
    else
      echo "  ERROR: svd.py failed for $old_file -> $new_file" >&2
    fi
  done
done

echo
echo "Done. Processed $model_count model dirs, $pair_count diffs written under: $OUTPUT_DIR"