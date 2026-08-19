#!/bin/bash

set -e

CACHE="/ssd_scratch/cvit/hf_cache/models"

mkdir -p "$CACHE"

echo "=============================================="
echo "Downloading LARGE VLM models"
echo "Cache: $CACHE"
echo "=============================================="

download_model () {
    MODEL="$1"
    NAME="$2"

    echo
    echo "=============================================="
    echo "Downloading: $NAME"
    echo "HF model: $MODEL"
    echo "=============================================="

    hf download "$MODEL" \
        --local-dir "$CACHE/$NAME"

    echo
    echo "DONE: $NAME"
}


# ============================================================
# QWEN 2
# ============================================================

download_model \
    "Qwen/Qwen2-VL-7B-Instruct" \
    "Qwen2-VL-7B-Instruct"


# ============================================================
# QWEN 2.5
# ============================================================

download_model \
    "Qwen/Qwen2.5-VL-7B-Instruct" \
    "Qwen2.5-VL-7B-Instruct"


# ============================================================
# QWEN 3
# ============================================================

download_model \
    "Qwen/Qwen3-VL-8B-Instruct" \
    "Qwen3-VL-8B-Instruct"


# ============================================================
# QWEN 3.5
# ============================================================

download_model \
    "Qwen/Qwen3.5-4B" \
    "Qwen3.5-4B"

download_model \
    "Qwen/Qwen3.5-9B" \
    "Qwen3.5-9B"


# ============================================================
# INTERNVL 2
# ============================================================

download_model \
    "OpenGVLab/InternVL2-4B" \
    "InternVL2-4B"

download_model \
    "OpenGVLab/InternVL2-8B" \
    "InternVL2-8B"


# ============================================================
# INTERNVL 2.5
# ============================================================

download_model \
    "OpenGVLab/InternVL2_5-4B" \
    "InternVL2_5-4B"

download_model \
    "OpenGVLab/InternVL2_5-8B" \
    "InternVL2_5-8B"


# ============================================================
# INTERNVL 3
# ============================================================

download_model \
    "OpenGVLab/InternVL3-8B" \
    "InternVL3-8B"


# ============================================================
# INTERNVL 3.5
# ============================================================

download_model \
    "OpenGVLab/InternVL3_5-4B" \
    "InternVL3_5-4B"

download_model \
    "OpenGVLab/InternVL3_5-8B" \
    "InternVL3_5-8B"


echo
echo "=============================================="
echo "ALL LARGE MODELS DOWNLOADED"
echo "=============================================="

du -sh "$CACHE"/*
