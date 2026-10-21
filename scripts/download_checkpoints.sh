#!/bin/bash
# Download the three final OpenWebText checkpoints (baseline, uniform,
# flop-proportional) to ./checkpoints.
#   bash scripts/download_checkpoints.sh                         # all three
#   bash scripts/download_checkpoints.sh "owt/uniform_*"         # one file
set -euo pipefail
source "$(dirname "$0")/common.sh"

huggingface-cli download saitejautpala/looped_udlm \
  --include "${1:-*.ckpt}" --local-dir checkpoints
