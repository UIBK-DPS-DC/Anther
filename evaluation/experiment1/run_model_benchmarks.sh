#!/bin/bash

MODELS=(
    cnn
    cnn1d
    crnn
    ds_cnn
    gru
    mlp
    svdf
)

INPUT_DIR="${1:?Usage: $0 <input_dir> <output_dir>}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir>}"

VERSIONS=(0 1 2 3 4)

for model in "${MODELS[@]}"; do
    for version in "${VERSIONS[@]}"; do

        model_file=$(find "${INPUT_DIR}/$model" \
            -maxdepth 1 \
            -type f \
            -name "version_${version}_*.pt" \
            | head -n 1)

        if [ -z "$model_file" ]; then
            echo "WARNING: No model found for $model version $version"
            continue
        fi

        filename=$(basename "$model_file" .pt)

        echo "============================================================"
        echo "Benchmarking: $model version $version"
        echo "Model:        $model_file"
        echo "============================================================"

        python3 train_lora.py \
            --data-dir data/speech_commands/ \
            --out-dir "${OUTPUT_DIR}/$model" \
            --device cuda \
            --benchmark-model "$model_file" \
            --benchmark-output "${filename}_benchmark.json"

        if [ $? -ne 0 ]; then
            echo "ERROR: Benchmark failed for $model version $version"
        fi

    done
done