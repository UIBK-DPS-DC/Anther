#!/usr/bin/env bash
#
# run_quant_apply_pipeline.sh
#
# QuantDelta baseline: scans model subdirectories containing .pt version
# checkpoints and corresponding QuantDelta .bin update files. Applies each
# update to the correct base checkpoint and writes the reconstructed model
# version to the output directory.
#
# Usage:
#   ./run_quant_apply_pipeline.sh \
#       <base_dir> \
#       <quant_delta_dir> \
#       <output_dir> \
#       [path_to_apply_quant.py]
#
# Example:
#   ./run_quant_apply_pipeline.sh \
#       ./runs \
#       ./quant_out \
#       ./benchmark_quant_update \
#       ./apply_quant.py

set -euo pipefail

BASE_DIR="${1:?Usage: $0 <base_dir> <quant_delta_dir> <output_dir> [apply_script]}"
QUANT_DIR="${2:?Usage: $0 <base_dir> <quant_delta_dir> <output_dir> [apply_script]}"
OUTPUT_DIR="${3:?Usage: $0 <base_dir> <quant_delta_dir> <output_dir> [apply_script]}"
APPLY_SCRIPT="${4:-$(dirname "$0")/apply_quant.py}"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Error: base directory '$BASE_DIR' does not exist." >&2
  exit 1
fi

if [[ ! -d "$QUANT_DIR" ]]; then
  echo "Error: quantized update directory '$QUANT_DIR' does not exist." >&2
  exit 1
fi

if [[ ! -f "$APPLY_SCRIPT" ]]; then
  echo "Error: apply_quant.py not found at '$APPLY_SCRIPT'." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

model_count=0
update_count=0

for model_dir in "$BASE_DIR"/*/; do
  [[ -d "$model_dir" ]] || continue

  model_name="$(basename "$model_dir")"
  quant_model_dir="$QUANT_DIR/$model_name"

  if [[ ! -d "$quant_model_dir" ]]; then
    echo "Skipping '$model_name': no matching QuantDelta directory found."
    continue
  fi

  # --------------------------------------------------
  # Find base .pt versions
  # --------------------------------------------------

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

  # --------------------------------------------------
  # Apply every consecutive QuantDelta update
  # --------------------------------------------------

  for ((i = 0; i < ${#pt_files[@]} - 1; i++)); do

    old_file="${pt_files[$i]}"
    new_file="${pt_files[$((i + 1))]}"

    old_base="$(basename "$old_file" .pt)"
    new_base="$(basename "$new_file" .pt)"

    # QuantDelta output naming follows the delta pipeline:
    #
    #   model_version_0_metric_to_model_version_1_metric.bin
    #
    delta_file="$quant_model_dir/${model_name}_${old_base}_to_${new_base}.bin"

    if [[ ! -f "$delta_file" ]]; then
      echo "  ERROR: QuantDelta update not found:"
      echo "         $delta_file" >&2
      continue
    fi

    out_file="$model_out_dir/${new_base}.pt"

    echo "  [$model_name] $old_base -> $new_base"

    if python3 "$APPLY_SCRIPT" \
        "$old_file" \
        "$delta_file" \
        --output "$out_file"; then

      update_count=$((update_count + 1))

    else

      echo "  ERROR: apply_quant.py failed for:"
      echo "         base:  $old_file"
      echo "         delta: $delta_file" >&2

    fi

  done
done

echo
echo "Done. Processed $model_count model dirs, $update_count QuantDelta updates applied under: $OUTPUT_DIR"