#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash inference.sh <checkpoint_path> [start_index] [max_samples] [output_dir]
#
# Example:
#   bash inference.sh results/03/21/libero_frame_ada/checkpoints/0000500.pt 0 1

CHECKPOINT_PATH="${1:-results/REPLACE_ME/checkpoints/0000500.pt}"
START_INDEX="${2:-0}"
MAX_SAMPLES="${3:-1}"
OUTPUT_DIR="${4:-work_dirs/inference/libero/$(basename "${CHECKPOINT_PATH%.pt}")}"
DEFAULT_PYTHON_BIN="python"
if [ -x "/data/firefly/.usr/miniconda3/envs/irasim/bin/python" ]; then
  DEFAULT_PYTHON_BIN="/data/firefly/.usr/miniconda3/envs/irasim/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

# Match training config `configs/train/libero/frame_ada.yaml`:
#   mode=val
#   num_frames=87
#   mask_frame_num=7
"${PYTHON_BIN}" examples/video2world_libero_inference.py \
  --config configs/train/libero/frame_ada.yaml \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --inference-mode episode \
  --mode val \
  --conditioning-frames 7 \
  --start-index "${START_INDEX}" \
  --max-samples "${MAX_SAMPLES}"
