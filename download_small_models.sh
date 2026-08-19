#!/bin/bash

set -u

MODELS=(
    "Qwen/Qwen2-VL-2B-Instruct"
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen/Qwen3-VL-2B-Instruct"
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3.5-4B"

    "OpenGVLab/InternVL2-1B"
    "OpenGVLab/InternVL2-2B"
    "OpenGVLab/InternVL2_5-1B"
    "OpenGVLab/InternVL2_5-2B"
    "OpenGVLab/InternVL3-1B"
    "OpenGVLab/InternVL3-2B"
    "OpenGVLab/InternVL3_5-4B"
)

for MODEL in "${MODELS[@]}"; do

    echo
    echo "=============================================="
    echo "Downloading: $MODEL"
    echo "=============================================="

    hf download "$MODEL"

    if [ $? -eq 0 ]; then
        echo "SUCCESS: $MODEL"
    else
        echo "FAILED: $MODEL"
    fi

done

echo
echo "=============================================="
echo "Download batch complete"
echo "=============================================="
