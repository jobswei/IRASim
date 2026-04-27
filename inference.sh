#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash inference.sh <checkpoint_path> [output_dir] [num_inference_steps] [num_gpus]
#
# Example:
#   bash inference.sh work_dirs/results/03/21/libero_frame_ada_smoke_local-debug/checkpoints/0000001.pt
#   bash inference.sh work_dirs/results/03/21/libero_frame_ada_smoke_local-debug/checkpoints/0000001.pt work_dirs/inference/libero/run_0001 50 4

CHECKPOINT_PATH="${1:-results/REPLACE_ME/checkpoints/0000500.pt}"
OUTPUT_DIR="${2:-work_dirs/inference/libero/$(basename "${CHECKPOINT_PATH%.pt}")}"
NUM_INFERENCE_STEPS="${3:-50}"
NUM_GPUS="${4:-1}"


# Match training config `configs/train/libero/frame_ada.yaml`:
#   mode=val
#   num_frames=87
#   mask_frame_num=7
# Default denoising steps comes from `configs/base/diffusion.yaml`:
#   infer_num_sampling_steps=50
COMMON_ARGS=(
  --config configs/train/libero/frame_ada.yaml
  --checkpoint "${CHECKPOINT_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --inference-mode slice
  --mode val
  --conditioning-frames 7
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
)

if [ "${NUM_GPUS}" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc_per_node "${NUM_GPUS}" \
    examples/video2world_libero_inference.py "${COMMON_ARGS[@]}"
else
  python examples/video2world_libero_inference.py "${COMMON_ARGS[@]}"
fi
