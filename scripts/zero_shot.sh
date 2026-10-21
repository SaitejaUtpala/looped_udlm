#!/bin/bash
# Zero-shot likelihood (NELBO perplexity bound) on seven out-of-domain corpora,
# using the DUO zero-shot recipe (data downloaded and packed on first use).
#
#   bash scripts/zero_shot.sh checkpoints/owt/uniform_step655420.ckpt adaptive
#   bash scripts/zero_shot.sh checkpoints/owt/baseline_step1000000.ckpt baseline
set -euo pipefail
source "$(dirname "$0")/common.sh"
source scripts/_loop_policy.sh

CKPT="${1:?usage: $0 <checkpoint> <baseline|S|adaptive>}"
POLICY="${2:?missing depth policy}"
shift 2
loop_policy_args "${POLICY}"
SEED="${SEED:-1}"

mkdir -p "${OUTPUT_ROOT}/zero_shot"
for data in ptb wikitext103 lm1b-gpt2 lambada ag_news \
            scientific_papers_pubmed scientific_papers_arxiv; do
  NAME="$(basename "${CKPT}" .ckpt)-${data}-${POLICY}-seed${SEED}"
  python -u main.py \
    mode=ppl_eval \
    algo=duo_base \
    data="${data}" \
    data.insert_valid_eos=False \
    seed="${SEED}" \
    loader.eval_batch_size=16 \
    eval.checkpoint_path="$(abspath "${CKPT}")" \
    eval.metrics_path="${OUTPUT_ROOT}/zero_shot/${NAME}.json" \
    eval.generate_samples=False \
    "${LOOP_ARGS[@]}" \
    hydra.run.dir="${OUTPUT_ROOT}/zero_shot/${NAME}" \
    '~wandb' \
    "$@"
done
