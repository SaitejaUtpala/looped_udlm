#!/bin/bash
# Zero-shot multiple-choice accuracy with lm-eval (pip install lm-eval).
#
#   bash scripts/mcq.sh checkpoints/owt/uniform_step655420.ckpt 12
#   bash scripts/mcq.sh checkpoints/owt/flop_proportional_step688915.ckpt adaptive
#   bash scripts/mcq.sh checkpoints/owt/baseline_step1000000.ckpt baseline
set -euo pipefail
source "$(dirname "$0")/common.sh"

CKPT="${1:?usage: $0 <checkpoint> <baseline|S|adaptive>}"
POLICY="${2:?missing depth policy}"
case "${POLICY}" in
  baseline) LOOP_ARGS="" ;;
  adaptive) LOOP_ARGS=",loop_eval_policy=adaptive,loop_eval_count=${S_MAX:-12},loop_adaptive_epsilon=${EPSILON:-0.1}" ;;
  *) LOOP_ARGS=",loop_eval_policy=fixed,loop_eval_count=${POLICY}" ;;
esac

OUT="${OUTPUT_ROOT}/mcq/$(basename "${CKPT}" .ckpt)-${POLICY}"
mkdir -p "${OUT}"
for task_config in "arc_easy 256" "arc_challenge 256" "hellaswag 64" \
                   "winogrande 64" "piqa 64" "mathqa 64" "openbookqa 256" \
                   "boolq 64" "race 32" "social_iqa 512"; do
  read -r task batch_size <<< "${task_config}"
  python discrete_diffusion_harness.py \
    --tasks "${task}" \
    --model dlm \
    --batch_size "${batch_size}" \
    --model_args "checkpoint_path=$(abspath "${CKPT}")${LOOP_ARGS}" \
    --trust_remote_code \
    --output_path "${OUT}/${task}.json"
done
