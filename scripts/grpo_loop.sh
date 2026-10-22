#!/usr/bin/env bash
# GRPO: alternate closed-loop sampling in AlpaSim and policy updates.
#
# Each iteration samples G rollouts per training scene with the current
# checkpoint (temperature sampling, CATK traffic, gsplat cameras), builds an
# immutable batch from the logged actions/log-probs and AlpaSim scores, and
# writes the next checkpoint. Wall time and rollouts are logged per iteration.
#
# Usage: scripts/grpo_loop.sh <sft-checkpoint> <run-dir> [iterations] [group-size]
# Env:   TEMPERATURE (1.0), KL_COEF (0.05), PROGRESS_WEIGHT (0.2), SPLIT (train),
#        SPLITS (splits file, e.g. a training subset), STEPS (sim steps per episode, 200),
#        TOPOLOGY / METADRIVE_RENDERER_ADDRS (see run_episode.sh),
#        EVAL_SPLIT (if set, evaluate each new checkpoint greedily on it), EVAL_ROLLOUTS (2),
#        SPLIT_CYCLE (space-separated split names; iteration i samples the (i-1) mod n-th)
set -euo pipefail

SFT=$(realpath "${1:?usage: grpo_loop.sh <sft-checkpoint> <run-dir> [iterations] [group-size]}")
RUN=$(realpath -m "${2:?}")
ITERS=${3:-3}
GROUP=${4:-4}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${ALPASIM_DIR:-$HOME/ext/alpasim}/.venv/bin/python

mkdir -p "$RUN"
policy=$SFT
for it in $(seq 1 "$ITERS"); do
  dir="$RUN/it$it"
  mkdir -p "$dir"
  cat > "$dir/policy.yaml" <<YAML
policy: {type: neural, checkpoint: $policy, temperature: ${TEMPERATURE:-1.0}}
greedy: false
decision_period_s: 0.5
feature_dir: $dir/features
YAML
  start=$(date +%s)
  split=${SPLIT:-train}
  if [[ -n "${SPLIT_CYCLE:-}" ]]; then
    read -r -a cycle <<< "$SPLIT_CYCLE"
    split=${cycle[$(( (it - 1) % ${#cycle[@]} ))]}
  fi
  echo "$split" > "$dir/split.txt"
  "$HERE/evaluate.sh" "$dir/policy.yaml" "$split" "$dir/rollouts" "$GROUP" \
    eval.video.render_video=False runtime.simulation_config.n_sim_steps="${STEPS:-200}" \
    > "$dir/collect.log" 2>&1
  collected=$(date +%s)
  "$PY" -m metadrive_rl.training.grpo build "$dir/rollouts" "$dir/batch" \
    --features "$dir/features" --progress-weight "${PROGRESS_WEIGHT:-0.2}" > "$dir/build.json"
  CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-2} "$PY" -m metadrive_rl.training.grpo update "$dir/batch" \
    --policy "$policy" --ref "$SFT" --out "$dir/policy.pt" --version "grpo:$(basename "$RUN"):it$it" \
    --kl-coef "${KL_COEF:-0.05}" > "$dir/update.json"
  done_at=$(date +%s)
  "$PY" - "$dir" "$start" "$collected" "$done_at" <<'EOF' | tee -a "$RUN/iterations.jsonl"
import json, sys
d, start, collected, done = sys.argv[1], *map(int, sys.argv[2:])
build = json.load(open(f"{d}/build.json"))
update = json.load(open(f"{d}/update.json"))
print(json.dumps({"iteration": d.rsplit("it", 1)[1], "rollouts": build["rollouts"],
                  "decisions": build["decisions"], "reward_mean": build["reward_mean"],
                  "zero_variance_groups": build["groups_with_zero_reward_variance"],
                  "collect_s": collected - start, "update_s": done - collected,
                  "rollouts_per_hour": round(build["rollouts"] * 3600 / max(collected - start, 1), 1),
                  **{k: update["final"][k] for k in ("kl_to_ref", "entropy", "clip_fraction")}}))
EOF
  policy=$dir/policy.pt
  if [[ -n "${EVAL_SPLIT:-}" ]]; then
    cat > "$dir/eval_policy.yaml" <<YAML
policy: {type: neural, checkpoint: $policy}
greedy: true
decision_period_s: 0.5
YAML
    SPLITS=${EVAL_SPLITS:-$HERE/../configs/evaluation/splits.yaml} "$HERE/evaluate.sh" \
      "$dir/eval_policy.yaml" "$EVAL_SPLIT" "$dir/eval" "${EVAL_ROLLOUTS:-2}" \
      eval.video.render_video=False > "$dir/eval.log" 2>&1
    "$PY" -m metadrive_rl.evaluation.summarize "$dir/eval=it$it" --out "$dir/eval.md"
  fi
done
