#!/bin/bash
# Validation likelihood (NELBO perplexity bound) of a checkpoint at a chosen depth.
#
#   bash scripts/eval_ppl.sh checkpoints/owt/uniform_step655420.ckpt 12
#   bash scripts/eval_ppl.sh checkpoints/owt/uniform_step655420.ckpt adaptive
#   bash scripts/eval_ppl.sh checkpoints/owt/baseline_step1000000.ckpt baseline
#   DATA=fineweb-edu bash scripts/eval_ppl.sh <your FineWeb-Edu checkpoint> 8
#
# Metrics and (for adaptive) depth telemetry are written to ${OUTPUT_ROOT}/eval/<name>.json.
set -euo pipefail
source "$(dirname "$0")/common.sh"
source scripts/_loop_policy.sh

CKPT="${1:?usage: $0 <checkpoint> <baseline|S|adaptive>}"
POLICY="${2:?missing depth policy}"
shift 2
loop_policy_args "${POLICY}"
DATA_CFG=openwebtext
[[ "${DATA:-owt}" == "fineweb-edu" ]] && DATA_CFG=fineweb-edu

NAME="$(basename "${CKPT}" .ckpt)-${DATA:-owt}-${POLICY}"
mkdir -p "${OUTPUT_ROOT}/eval"
python -u main.py \
  mode=ppl_eval \
  algo=duo_base \
  data="${DATA_CFG}" \
  eval.checkpoint_path="$(abspath "${CKPT}")" \
  eval.metrics_path="${OUTPUT_ROOT}/eval/${NAME}.json" \
  eval.generate_samples=False \
  "${LOOP_ARGS[@]}" \
  hydra.run.dir="${OUTPUT_ROOT}/eval/${NAME}" \
  '~wandb' \
  "$@"
