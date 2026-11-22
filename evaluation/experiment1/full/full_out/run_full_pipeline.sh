#!/usr/bin/env bash
#
# run_full_model_pipeline.sh
#
# Full model baseline: scans a directory of model subdirectories (each
# containing .pt version checkpoints named "version_<N>_<metric>.pt") and
# converts every version into a complete serialized .bin model file.
#
# Usage:
#   ./run_full_model_pipeline.sh <input_dir> <output_dir> [path_to_full_model.py]
#
# Example:
#   ./run_full_model_pipeline.sh ./runs ./full_out ./full_model.py

set -euo pipefail

INPUT_DIR="${1:?Usage: $0 <input_dir> <output_dir> [full_model_script]}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir> [full_model_script]}"
FULL_MODEL_SCRIPT="${3:-$(dirname "$0")/full_model.py}"
EXT="bin"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: input directory '$INPUT_DIR' does not exist." >&2
  exit 1
fi

if [[ ! -f "$FULL_MODEL_SCRIPT" ]]; then
  echo "Error: full_model.py not found at '$FULL_MODEL_SCRIPT'." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

model_count=0
version_count=0

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

  if [[ "${#pt_files[@]}" -lt 1 ]]; then
    echo "Skipping '$model_name': no recognizable .pt versions found."
    continue
  fi

  model_out_dir="$OUTPUT_DIR/$model_name"
  mkdir -p "$model_out_dir"

  echo "=== $model_name (${#pt_files[@]} versions) ==="
  model_count=$((model_count + 1))

  for pt_file in "${pt_files[@]}"; do
    pt_base="$(basename "$pt_file" .pt)"

    out_file="$model_out_dir/${model_name}_${pt_base}.${EXT}"

    echo "  [$model_name] $pt_base -> $(basename "$out_file")"

    if python3 "$FULL_MODEL_SCRIPT" \
        "$pt_file" \
        --output "$out_file"; then

      version_count=$((version_count + 1))

    else
      echo "  ERROR: full_model.py failed for $pt_file" >&2
    fi
  done
done

echo
echo "Done. Processed $model_count model dirs, $version_count full model versions written under: $OUTPUT_DIR"