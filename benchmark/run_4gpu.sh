#!/bin/bash

source /home2/gagandeep/benchmark/setup.sh

CUDA_VISIBLE_DEVICES=0 \
python /home2/gagandeep/benchmark/benchmark.py \
    --model Qwen3-VL-4B \
    --start 901 --end 1000 \
    > qwen25_3b.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 \
python /home2/gagandeep/benchmark/benchmark.py \
    --model Qwen2-VL-2B \
    --start 901 --end 1000 \
    > qwen2_2b.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 \
python /home2/gagandeep/benchmark/benchmark.py \
    --model Qwen3-VL-2B \
    --start 901 --end 1000 \
    > qwen3_2b.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 \
python /home2/gagandeep/benchmark/benchmark.py \
    --model Qwen3.5-4B \
    --start 901 --end 1000 \
    > internvl25_2b.log 2>&1 &

echo "Started all 4 benchmarks."
echo "GPU 0: Qwen3-VL-4B"
echo "GPU 1: Qwen2-VL-2B"
echo "GPU 2: Qwen3-VL-2B"
echo "GPU 3: Qwen3.5-4B""
