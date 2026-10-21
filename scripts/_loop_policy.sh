# Shared by the evaluation scripts: turn a depth policy into Hydra overrides.
#   baseline   non-looped checkpoint
#   <S>        looped checkpoint at fixed depth S (e.g. 1, 6, 8, 12, 24)
#   adaptive   looped checkpoint, per-sample adaptive depth with cap S_MAX (default 12)
loop_policy_args() {
  case "$1" in
    baseline) LOOP_ARGS=(model.loop.enabled=False) ;;
    adaptive) LOOP_ARGS=(model.loop.enabled=True model.loop.eval_policy=adaptive
                         model.loop.eval_count="${S_MAX:-12}"
                         model.loop.adaptive_epsilon="${EPSILON:-0.1}") ;;
    ''|*[!0-9]*) echo "policy must be baseline, adaptive, or an integer depth" >&2; exit 1 ;;
    *) LOOP_ARGS=(model.loop.enabled=True model.loop.eval_policy=fixed
                  model.loop.eval_count="$1") ;;
  esac
}
