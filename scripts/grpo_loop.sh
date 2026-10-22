#!/usr/bin/env bash
# GRPO: alternate closed-loop sampling in AlpaSim and policy updates.
#
# Each iteration samples G rollouts per training scene with the current
# checkpoint (temperature sampling, CATK traffic, gsplat cameras), builds an
# immutable batch from the logged actions/log-probs and AlpaSim scores, and
# writes the next checkpoint. Wall time and rollouts are logged per iteration.
#
# Usage: scripts/grpo_loop.sh <sft-checkpoint> <run-dir> [iterations] [group-size]
# Env:   TEMPERATURE (1.0), KL_COEF (0.05), PROGRESS_WEIGHT (0.2), SPLIT (train)
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
  "$HERE/evaluate.sh" "$dir/policy.yaml" "${SPLIT:-train}" "$dir/rollouts" "$GROUP" > "$dir/collect.log" 2>&1
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
done
