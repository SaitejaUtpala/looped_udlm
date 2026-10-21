# Sourced by every script: repo root, .env, and shared defaults.
CALLER_DIR="$(pwd)"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

# Absolute path of a file given relative to where the script was invoked.
abspath() { (cd "${CALLER_DIR}" && realpath "$1"); }

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
# Empty values in .env mean "use the default"; unset them so tools don't see "".
for var in WANDB_API_KEY WANDB_ENTITY HF_TOKEN LOOPED_UDLM_DATA OUTPUT_ROOT \
           CUDA_VISIBLE_DEVICES; do
  [[ -z "${!var:-}" ]] && unset "${var}"
done

export WANDB_PROJECT="${WANDB_PROJECT:-looped_udlm}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/outputs}"
export HYDRA_FULL_ERROR=1

num_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}"
  else
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 0
  fi
}

# Hydra overrides for the W&B logger: wandb_args <run name>.
wandb_args() {
  WANDB_ARGS=(wandb.project="${WANDB_PROJECT}" wandb.name="$1")
  [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_ARGS+=(wandb.entity="${WANDB_ENTITY}")
  if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set (see .env.example); set WANDB_MODE=offline to log locally." >&2
    exit 1
  fi
  [[ "${WANDB_MODE}" == "disabled" ]] && WANDB_ARGS=('~wandb')
  return 0
}
