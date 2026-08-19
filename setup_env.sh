#!/bin/bash

source /home2/gagandeep/internvl_env/bin/activate

export HF_HOME=/ssd_scratch/cvit/hf_cache
export HF_HUB_CACHE=/ssd_scratch/cvit/hf_cache/hub
export TRANSFORMERS_CACHE=/ssd_scratch/cvit/hf_cache/hub

export FORCE_QWENVL_VIDEO_READER=decord

export LD_LIBRARY_PATH=/home2/gagandeep/internvl_env/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1

