#!/usr/bin/env bash
#
# run_bpatch_pipeline.sh
#
# BPatch baseline: scans a directory of model subdirectories containing
# full serialized .bin model versions and runs bpatch.py encode between
# every pair of consecutive versions within each model directory.
#
# Usage:
#   ./run_bpatch_pipeline.sh <input_dir> <output_dir> [path_to_bpatch.py]
#
# Example:
#   ./run_bpatch_pipeline.sh ./full_out ./bpatch_out ./bpatch.py
#
# Input:
#   full_out/
#   ├── cnn/
#   │   ├── cnn_version_0_9333.bin
#   │   ├── cnn_version_1_27999.bin
#   │   └── cnn_version_2_40000.bin
#   └── ...
#
# Output:
#   bpatch_out/
#   ├── cnn/
#   │   ├── cnn_version_0_9333_to_version_1_27999.bin
#   │   └── cnn_version_1_27999_to_version_2_40000.bin
#   └── ...

set -euo pipefail

INPUT_DIR="${1:?Usage: $0 <input_dir> <output_dir> [bpatch_script]}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir> [bpatch_script]}"
BPATCH_SCRIPT="${3:-$(dirname "$0")/bpatch.py}"
EXT="bin"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: input directory '$INPUT_DIR' does not exist." >&2
  exit 1
fi

if [[ ! -f "$BPATCH_SCRIPT" ]]; then
  echo "Error: bpatch.py not found at '$BPATCH_SCRIPT'." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

model_count=0
pair_count=0

for model_dir in "$INPUT_DIR"/*/; do
  [[ -d "$model_dir" ]] || continue

  model_name="$(basename "$model_dir")"

  if [[ ! $model_name =~ ^(svdf)$ ]]; then
    echo "Skipping model directory name: '$model_name'" >&2
    continue
  fi

  # Find all .bin files containing version_<N>_ and sort by version number.
  mapfile -t bin_files < <(
    find "$model_dir" -maxdepth 1 -type f -name '*.bin' \
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

  if [[ "${#bin_files[@]}" -lt 2 ]]; then
    echo "Skipping '$model_name': fewer than 2 recognizable .bin versions found."
    continue
  fi

  model_out_dir="$OUTPUT_DIR/$model_name"
  mkdir -p "$model_out_dir"

  echo "=== $model_name (${#bin_files[@]} versions) ==="
  model_count=$((model_count + 1))

  for ((i = 0; i < ${#bin_files[@]} - 1; i++)); do
    old_file="${bin_files[$i]}"
    new_file="${bin_files[$((i + 1))]}"

    old_base="$(basename "$old_file" .bin)"
    new_base="$(basename "$new_file" .bin)"

    out_file="$model_out_dir/${old_base}_to_${new_base}.${EXT}"

    echo "  [$model_name] $old_base -> $new_base"

    if python3 "$BPATCH_SCRIPT" encode \
        "$old_file" \
        "$new_file" \
        "$out_file" \
        -d; then

      pair_count=$((pair_count + 1))

    else
      echo "  ERROR: bpatch.py failed for $old_file -> $new_file" >&2
    fi
  done
done

echo
echo "Done. Processed $model_count model dirs, $pair_count BPatch deltas written under: $OUTPUT_DIR"