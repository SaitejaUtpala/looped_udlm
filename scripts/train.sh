#!/bin/bash
# Train a non-looped or looped DUO++ small model on one node.
#
#   bash scripts/train.sh owt baseline
#   bash scripts/train.sh owt uniform
#   bash scripts/train.sh fineweb-edu flop_proportional
#   bash scripts/train.sh owt noise_proportional trainer.max_steps=1000   # extra Hydra overrides
#
# Global batch is 512 sequences on any GPU count. The per-GPU micro-batch
# follows the paper's 4xH100 runs: 32 for the baseline, 16 for looped runs
# (deeper unrolls); override with BATCH_SIZE. Looped runs stop at the step
# budget whose training FLOPs match 1M baseline steps (655,420 uniform and
# noise_proportional, 688,915 flop_proportional). Re-running the same command
# resumes from ${OUTPUT_ROOT}/<data>/<schedule>_seed<seed>/checkpoints/last.ckpt.
set -euo pipefail
source "$(dirname "$0")/common.sh"

DATA="${1:?usage: $0 <owt|fineweb-edu> <baseline|uniform|flop_proportional|noise_proportional> [overrides...]}"
SCHEDULE="${2:?missing schedule}"
shift 2
SEED="${SEED:-1}"
BASELINE_STEPS="${BASELINE_STEPS:-1000000}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"

case "${DATA}" in
  owt) DATA_CFG=openwebtext ;;
  fineweb-edu) DATA_CFG=fineweb-edu ;;
  *) echo "unknown data ${DATA}" >&2; exit 1 ;;
esac

if [[ "${SCHEDULE}" == "baseline" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-32}"
  STEPS="${BASELINE_STEPS}"
  LOOP_ARGS=(model.loop.enabled=False)
else
  BATCH_SIZE="${BATCH_SIZE:-16}"
  STEPS=$(python -c "
import flops, loop_schedule as ls
arch = flops.Arch(vocab_size=50257, loop_enabled=True, loop_num_layers=2)
vals, probs = ls.schedule_pmf('${SCHEDULE}', 1, 12, arch)
print(flops.iso_flop_steps(arch, ls.expected_value(vals, probs), ${BASELINE_STEPS}))")
  LOOP_ARGS=(
    model.loop.enabled=True
    model.loop.schedule="${SCHEDULE}"
    "model.loop.eval_count_sweep=[1,6,8,12]"
  )
fi

NGPU="$(num_gpus)"
[[ "${NGPU}" -ge 1 ]] || { echo "no visible GPUs" >&2; exit 1; }
if (( GLOBAL_BATCH % (NGPU * BATCH_SIZE) )); then
  echo "GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by ${NGPU} GPUs x BATCH_SIZE=${BATCH_SIZE}" >&2
  exit 1
fi
RUN_NAME="${DATA}-${SCHEDULE}-seed${SEED}"
wandb_args "${RUN_NAME}"
OUT="${OUTPUT_ROOT}/${DATA}/${SCHEDULE}_seed${SEED}"
echo "run=${RUN_NAME} steps=${STEPS} gpus=${NGPU} micro_batch=${BATCH_SIZE}" \
     "accumulation=$(( GLOBAL_BATCH / (NGPU * BATCH_SIZE) )) out=${OUT}"

python -u main.py \
  mode=train \
  data="${DATA_CFG}" \
  seed="${SEED}" \
  trainer.devices="${NGPU}" \
  trainer.max_steps="${STEPS}" \
  loader.global_batch_size="${GLOBAL_BATCH}" \
  loader.batch_size="${BATCH_SIZE}" \
  loader.eval_batch_size="$(( 2 * BATCH_SIZE ))" \
  "${LOOP_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  hydra.run.dir="${OUT}" \
  checkpointing.save_dir="${OUT}" \
  "$@"
