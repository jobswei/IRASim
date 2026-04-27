#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash rollout_inference.sh <checkpoint_path> <total_frames> [output_dir] [num_inference_steps] [num_gpus|auto]
#
# Example:
#   bash rollout_inference.sh work_dirs/results/03/21/libero_frame_ada_smoke_local-debug/checkpoints/0000001.pt 256
#   bash rollout_inference.sh work_dirs/results/03/21/libero_frame_ada_smoke_local-debug/checkpoints/0000001.pt 256 work_dirs/inference/libero_rollout/run_0001 50 4

CHECKPOINT_PATH="${1:-results/REPLACE_ME/checkpoints/0000500.pt}"
TOTAL_FRAMES="${2:-256}"
OUTPUT_DIR="${3:-work_dirs/inference/libero_rollout/$(basename "${CHECKPOINT_PATH%.pt}")_${TOTAL_FRAMES}f}"
NUM_INFERENCE_STEPS="${4:-50}"
NUM_GPUS="${5:-auto}"

resolve_num_gpus() {
  local requested="${1}"
  if [ "${requested}" != "auto" ]; then
    echo "${requested}"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local detected
    detected="$(nvidia-smi -L | wc -l)"
    if [ "${detected}" -gt 0 ]; then
      echo "${detected}"
      return 0
    fi
  fi

  echo "1"
}

NUM_GPUS="$(resolve_num_gpus "${NUM_GPUS}")"


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
)

echo "[rollout_inference] checkpoint=${CHECKPOINT_PATH}"
echo "[rollout_inference] total_frames=${TOTAL_FRAMES} output_dir=${OUTPUT_DIR}"
echo "[rollout_inference] num_inference_steps=${NUM_INFERENCE_STEPS} num_gpus=${NUM_GPUS} dataset=full_eval"

if [ "${NUM_GPUS}" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc_per_node "${NUM_GPUS}" \
    examples/video2world_libero_inference.py "${COMMON_ARGS[@]}"
else
  python examples/video2world_libero_inference.py "${COMMON_ARGS[@]}"
fi
