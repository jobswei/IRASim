#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5

exec torchrun \
  --standalone \
  --nproc_per_node=2 \
  main.py --config "configs/train/agibot/frame_ada.yaml"
