#!/usr/bin/env bash
#
# run_extract_pipeline.sh
#
# Scans a directory of model subdirectories (each containing .pt version
# checkpoints named like "version_<N>_<metric>.pt") and runs lora_tinyml_export.py
# between every pair of consecutive versions within each model directory.
#
# For 5 versions (0..4) per model, this produces 4 diff files per model:
#   0->1, 1->2, 2->3, 3->4
#
# The svd-compress command now operates exclusively on LoRA parameters:
#   - dense/frozen model weights are ignored
#   - LoRA parameters are differenced between consecutive checkpoints
#   - unchanged LoRA parameters are skipped
#   - changed LoRA deltas are quantized directly
#
# Expected layout under --root:
#
#   <root>/
#     DSCNN/
#       version_0_acc0.81.pt
#       version_1_acc0.85.pt
#       version_2_acc0.88.pt
#     CRNN/
#       version_0_acc0.77.pt
#       version_1_acc0.83.pt
#
# Usage:
#   ./run_extract_pipeline.sh --root ./checkpoints --out-dir ./deltas
#
# Options:
#   --root DIR             Root directory, one subdirectory per model (required)
#   --out-dir DIR          Where to write .lrq output files (required)
#   --export-script PATH   Path to lora_tinyml_export.py
#                          (default: alongside this script)
#   --python PATH          Python interpreter to use (default: python3)
#   --bits {4,8}           Quantization bit-width (default: 8)
#   --with-compare         Also run `compare` against the newest checkpoint
#                          in each model directory. This needs a resolvable
#                          architecture name and a LoRA-shaped checkpoint.
#                          Failures are warnings and do not abort the scan.
#   --with-verify          Also run `verify` for every generated delta,
#                          checking that base LoRA + dequantized delta
#                          reconstructs the finetuned LoRA parameters.
#   --dry-run              Print the commands without running them
#   -h, --help             Show this help and exit
#
# Examples:
#   # Basic run, defaults everywhere
#   ./run_extract_pipeline.sh --root ./checkpoints --out-dir ./deltas
#
#   # 4-bit deltas
#   ./run_extract_pipeline.sh --root ./checkpoints --out-dir ./deltas --bits 4
#
#   # Also print the size comparison
#   ./run_extract_pipeline.sh --root ./checkpoints --out-dir ./deltas \
#       --with-compare
#
#   # Also verify every generated delta
#   ./run_extract_pipeline.sh --root ./checkpoints --out-dir ./deltas \
#       --with-verify
#
#   # See what would run without touching anything
#   ./run_extract_pipeline.sh --root ./checkpoints --out-dir ./deltas \
#       --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT=""
OUT_DIR=""
EXPORT_SCRIPT="$SCRIPT_DIR/extract_lora_update.py"
PYTHON_BIN="python3"
BITS=8
WITH_COMPARE=0
WITH_VERIFY=0
DRY_RUN=0

KNOWN_MODELS=(MLP MLP_BIG CNN CNN1D DSCNN GRU CRNN SVDF)

usage() {
    sed -n '2,66p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --root)
            ROOT="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --export-script)
            EXPORT_SCRIPT="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --bits)
            BITS="$2"
            shift 2
            ;;
        --with-compare)
            WITH_COMPARE=1
            shift
            ;;
        --with-verify)
            WITH_VERIFY=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage 1
            ;;
    esac
done

[ -n "$ROOT" ] || {
    echo "error: --root is required" >&2
    usage 1
}

[ -n "$OUT_DIR" ] || {
    echo "error: --out-dir is required" >&2
    usage 1
}

[ -d "$ROOT" ] || {
    echo "error: --root '$ROOT' is not a directory" >&2
    exit 1
}

[ -f "$EXPORT_SCRIPT" ] || {
    echo "error: export script not found at '$EXPORT_SCRIPT' (use --export-script)" >&2
    exit 1
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "error: python interpreter '$PYTHON_BIN' not found" >&2
    exit 1
}

if [ "$BITS" != "4" ] && [ "$BITS" != "8" ]; then
    echo "error: --bits must be 4 or 8" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

run_cmd() {
    echo "  + $*"

    if [ "$DRY_RUN" -ne 1 ]; then
        "$@"
    fi
}

# Optional commands such as compare/verify should warn and continue if they
# fail rather than aborting the entire scan.
run_cmd_soft() {
    echo "  + $*"

    if [ "$DRY_RUN" -eq 1 ]; then
        return 0
    fi

    if ! "$@"; then
        echo "  [warn] command failed, continuing: $*" >&2
        return 1
    fi

    return 0
}

resolve_model_name() {
    local upper
    upper=$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')

    for m in "${KNOWN_MODELS[@]}"; do
        if [ "$upper" = "$m" ]; then
            printf '%s' "$m"
            return 0
        fi
    done

    printf ''
    return 1
}

n_models=0
n_deltas=0
n_skipped=0

shopt -s nullglob

for model_dir in "$ROOT"/*/; do
    [ -d "$model_dir" ] || continue

    dir_name="$(basename "$model_dir")"

    echo "== $dir_name =="

    n_models=$((n_models + 1))

    # Gather version_<N>_*.pt files -> array of "N:path",
    # sorted numerically by version number.
    entries=()

    for f in "$model_dir"version_*.pt; do
        bn="$(basename "$f")"

        if [[ $bn =~ ^version_([0-9]+)_ ]]; then
            entries+=("${BASH_REMATCH[1]}:$f")
        else
            echo "  [warn] '$bn' doesn't match 'version_<N>_...pt', skipping file" >&2
        fi
    done

    if [ ${#entries[@]} -eq 0 ]; then
        echo "  [warn] no version_<N>_*.pt files found, skipping model" >&2
        n_skipped=$((n_skipped + 1))
        continue
    fi

    sorted=()

    while IFS= read -r line; do
        sorted+=("$line")
    done < <(
        printf '%s\n' "${entries[@]}" |
            sort -t: -k1,1n
    )

    out_subdir="$OUT_DIR/$dir_name"
    mkdir -p "$out_subdir"

    count=${#sorted[@]}
    latest_checkpoint="${sorted[$((count - 1))]#*:}"

    if [ "$count" -lt 2 ]; then
        echo "  [warn] only one version found; need at least two versions to compute a delta" >&2
        n_skipped=$((n_skipped + 1))
        continue
    fi

    #
    # Generate one LoRA delta for every consecutive pair.
    #
    for ((i = 0; i < count - 1; i++)); do
        ver_a="${sorted[i]%%:*}"
        path_a="${sorted[i]#*:}"

        ver_b="${sorted[i + 1]%%:*}"
        path_b="${sorted[i + 1]#*:}"

        out_file="$out_subdir/delta_${ver_a}_to_${ver_b}.lrq"

        run_cmd "$PYTHON_BIN" "$EXPORT_SCRIPT" extract \
            --base "$path_a" \
            --finetuned "$path_b" \
            --bits "$BITS" \
            --out "$out_file"

        n_deltas=$((n_deltas + 1))

        #
        # Verify this particular delta immediately if requested.
        #
        if [ "$WITH_VERIFY" -eq 1 ]; then
            run_cmd_soft "$PYTHON_BIN" "$EXPORT_SCRIPT" verify \
                --lora-binary "$out_file" \
                --base "$path_a" \
                --finetuned "$path_b" || true
        fi
    done

    #
    # Optional comparison against the newest checkpoint.
    #
    if [ "$WITH_COMPARE" -eq 1 ]; then
        model_name="$(resolve_model_name "$dir_name" || true)"

        if [ -n "$model_name" ]; then
            latest_delta="$out_subdir/delta_"

            # Find the delta ending at the latest version.
            latest_ver="${sorted[$((count - 1))]%%:*}"
            previous_ver="${sorted[$((count - 2))]%%:*}"
            latest_delta="$out_subdir/delta_${previous_ver}_to_${latest_ver}.lrq"

            if [ -f "$latest_delta" ] || [ "$DRY_RUN" -eq 1 ]; then
                run_cmd_soft "$PYTHON_BIN" "$EXPORT_SCRIPT" compare \
                    --checkpoint "$latest_checkpoint" \
                    --model-name "$model_name" \
                    --bits "$BITS" \
                    --lora-binary "$latest_delta" || true
            else
                echo "  [warn] --with-compare could not find latest delta '$latest_delta'" >&2
            fi
        else
            echo "  [warn] --with-compare skipped for '$dir_name': architecture unresolved" >&2
        fi
    fi
done

echo
echo "Summary: $n_models model dir(s) scanned, $n_deltas consecutive-version delta(s)," \
     "$n_skipped skipped."