#!/bin/bash
# Unconditional generation: Gen. PPL under GPT-2 Large and unigram entropy.
# Defaults follow the paper: 512 samples, nucleus p=0.9, ancestral sampler.
#
#   bash scripts/gen_ppl.sh checkpoints/owt/uniform_step655420.ckpt 12
#   bash scripts/gen_ppl.sh checkpoints/owt/flop_proportional_step688915.ckpt adaptive
#   STEPS=64 bash scripts/gen_ppl.sh checkpoints/owt/baseline_step1000000.ckpt baseline
#   PREDICTOR=psi bash scripts/gen_ppl.sh ...    # Psi-sampler
#
# Samples, metrics, and (for adaptive) depth telemetry go to ${OUTPUT_ROOT}/samples/<name>.json.
set -euo pipefail
source "$(dirname "$0")/common.sh"
source scripts/_loop_policy.sh

CKPT="${1:?usage: $0 <checkpoint> <baseline|S|adaptive>}"
POLICY="${2:?missing depth policy}"
shift 2
loop_policy_args "${POLICY}"
STEPS="${STEPS:-1024}"
PREDICTOR="${PREDICTOR:-ancestral}"
SEED="${SEED:-1}"

NAME="$(basename "${CKPT}" .ckpt)-${POLICY}-${PREDICTOR}-steps${STEPS}-seed${SEED}"
mkdir -p "${OUTPUT_ROOT}/samples"
python -u main.py \
  mode=sample_eval \
  algo=duo_base \
  seed="${SEED}" \
  eval.checkpoint_path="$(abspath "${CKPT}")" \
  eval.generated_samples_path="${OUTPUT_ROOT}/samples/${NAME}.json" \
  eval.compute_generative_perplexity=True \
  loader.eval_batch_size=64 \
  sampling.num_sample_batches=8 \
  sampling.steps="${STEPS}" \
  sampling.predictor="${PREDICTOR}" \
  sampling.p_nucleus=0.9 \
  trainer.devices=1 \
  "${LOOP_ARGS[@]}" \
  hydra.run.dir="${OUTPUT_ROOT}/samples/${NAME}" \
  '~wandb' \
  "$@"
