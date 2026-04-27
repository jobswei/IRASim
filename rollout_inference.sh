#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash rollout_inference.sh <checkpoint_path> <total_frames> [output_dir] [num_inference_steps] [num_gpus] [start_index] [max_samples]
#
# Example:
#   bash rollout_inference.sh work_dirs/results/03/21/libero_frame_ada_smoke_local-debug/checkpoints/0000001.pt 256
#   bash rollout_inference.sh work_dirs/results/03/21/libero_frame_ada_smoke_local-debug/checkpoints/0000001.pt 256 work_dirs/inference/libero_rollout/run_0001 50 4 0 8

CHECKPOINT_PATH="${1:-results/REPLACE_ME/checkpoints/0000500.pt}"
TOTAL_FRAMES="${2:-256}"
OUTPUT_DIR="${3:-work_dirs/inference/libero_rollout/$(basename "${CHECKPOINT_PATH%.pt}")_${TOTAL_FRAMES}f}"
NUM_INFERENCE_STEPS="${4:-50}"
NUM_GPUS="${5:-1}"
START_INDEX="${6:-0}"
MAX_SAMPLES="${7:-1}"


# Match training config `configs/train/libero/frame_ada_16.yaml`:
#   mode=val
#   num_frames=17
#   mask_frame_num=1
# Rollout behavior:
#   1. first chunk uses 1 GT conditioning frame
#   2. later chunks reuse only the last predicted frame as condition
#   3. generation stops after reaching `TOTAL_FRAMES`
COMMON_ARGS=(
  --config configs/train/libero/frame_ada_16.yaml
  --checkpoint "${CHECKPOINT_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --inference-mode rollout
  --mode val
  --conditioning-frames 1
  --rollout-conditioning-frames 1
  --total-frames "${TOTAL_FRAMES}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --start-index "${START_INDEX}"
  --max-samples "${MAX_SAMPLES}"
)

if [ "${NUM_GPUS}" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc_per_node "${NUM_GPUS}" \
    examples/video2world_libero_inference.py "${COMMON_ARGS[@]}"
else
  python examples/video2world_libero_inference.py "${COMMON_ARGS[@]}"
fi
