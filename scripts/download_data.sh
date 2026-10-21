#!/bin/bash
# Download the pre-tokenized (GPT-2), pre-packed 1024-token training data.
#   bash scripts/download_data.sh owt            # OpenWebText (~73 GB)
#   bash scripts/download_data.sh fineweb-edu    # FineWeb-Edu sample-10BT
# Files land in ${LOOPED_UDLM_DATA:-./data}/{openwebtext,fineweb-edu}, which is
# where configs/data/*.yaml look for them. No tokenization or packing is needed.
set -euo pipefail
source "$(dirname "$0")/common.sh"

DATA_ROOT="${LOOPED_UDLM_DATA:-$(pwd)/data}"

case "${1:-owt}" in
  owt)         repo=saitejautpala/openwebtext-gpt2-packed-1024;     dir=openwebtext ;;
  fineweb-edu) repo=saitejautpala/fineweb-edu-10bt-gpt2-packed-1024; dir=fineweb-edu ;;
  *) echo "usage: $0 {owt|fineweb-edu}" >&2; exit 1 ;;
esac

huggingface-cli download "${repo}" --repo-type dataset \
  --local-dir "${DATA_ROOT}/${dir}"
echo "Downloaded ${repo} to ${DATA_ROOT}/${dir}"
