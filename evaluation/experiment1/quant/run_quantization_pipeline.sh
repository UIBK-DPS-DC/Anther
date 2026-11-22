#!/usr/bin/env bash
#
# run_quant_pipeline.sh
#
# QuantDelta baseline: scans a directory of model subdirectories (each
# containing .pt version checkpoints named "version_<N>_<metric>.pt") and
# runs quant_delta.py between every pair of consecutive versions within
# each model directory. Mirrors run_svd_pipeline.sh / run_delta_pipeline.sh's
# structure so outputs line up 1:1 for comparison.
#
# Usage:
#   ./run_quant_pipeline.sh <input_dir> <output_dir> [path_to_quant_delta.py]
#
# Example:
#   ./run_quant_pipeline.sh ./runs ./quant_out ./quant_delta.py

set -euo pipefail

INPUT_DIR="${1:?Usage: $0 <input_dir> <output_dir> [quant_script]}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir> [quant_script]}"
QUANT_SCRIPT="${3:-$(dirname "$0")/quant_delta.py}"
EXT="bin"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: input directory '$INPUT_DIR' does not exist." >&2
  exit 1
fi

if [[ ! -f "$QUANT_SCRIPT" ]]; then
  echo "Error: quant_delta.py not found at '$QUANT_SCRIPT'." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

model_count=0
pair_count=0

for model_dir in "$INPUT_DIR"/*/; do
  [[ -d "$model_dir" ]] || continue
  model_name="$(basename "$model_dir")"

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

    out_file="$model_out_dir/${model_name}_${old_base}_to_${new_base}.${EXT}"

    echo "  [$model_name] $old_base -> $new_base"
    if python3 "$QUANT_SCRIPT" "$old_file" "$new_file" --output "$out_file"; then
      pair_count=$((pair_count + 1))
    else
      echo "  ERROR: quant_delta.py failed for $old_file -> $new_file" >&2
    fi
  done
done

echo
echo "Done. Processed $model_count model dirs, $pair_count quantized deltas written under: $OUTPUT_DIR"