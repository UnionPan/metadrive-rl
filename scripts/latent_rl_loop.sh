#!/usr/bin/env bash
# M5a on-policy latent RL in AlpaSim (metadrive_ais.training.latent_rl).
#
# Each iteration rolls out the current VaVAM-AIS bundle on a fresh seeded draw
# of train scenes (CATK, 4 s forced replay, native 8-frame context), split over
# two AlpaSim stacks, then makes one PPO update of the latent prior on the CPU.
# Bundles: <out>/bundle_v<k>.pt; logs and stats: <out>/iter_<k>/.
#
# Usage: scripts/latent_rl_loop.sh <out-dir> <bundle-v0> <iterations>
# Env:   RL_GPUS ("2 3": one stack + renderer per GPU), SCENES_PER_ITER (16),
#        ROLLOUTS (4 per scene), SEED (0), START (first iteration, to resume)
set -euo pipefail

OUT=$(realpath -m "${1:?usage: latent_rl_loop.sh <out-dir> <bundle-v0> <iterations>}")
B0=$(realpath "${2:?bundle v0}")
ITERS=${3:?iterations}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$HOME/ext/alpasim/.venv/bin/python"
read -r -a GPUS <<< "${RL_GPUS:-2 3}"
export VAVAM_MUP_SHAPES_DIR=$HOME/ext/alpasim/data/drivers/vavam/mup_shapes
cd "$REPO"
mkdir -p "$OUT"
[ -f "$OUT/bundle_v0.pt" ] || cp "$B0" "$OUT/bundle_v0.pt"

# One renderer per GPU for the whole loop.
RENDERERS=()
for i in "${!GPUS[@]}"; do
  port=$((46180 + 10 * i))
  if ! ss -ltn | grep -q ":$port "; then
    scripts/serve_gsplat_renderer.sh "$port" "${GPUS[$i]}" > "$OUT/renderer_$port.log" 2>&1 &
    RENDERERS+=($!)
  fi
done
trap 'kill "${RENDERERS[@]}" 2>/dev/null || true' EXIT
for i in "${!GPUS[@]}"; do
  until ss -ltn | grep -q ":$((46180 + 10 * i)) "; do sleep 5; done
done

for ((k = ${START:-0}; k < ITERS; k++)); do
  IT="$OUT/iter_$k"
  mkdir -p "$IT"
  # Seeded scene draw from the train split, without the two broken-route scenes.
  mapfile -t HALVES < <("$PY" - "$k" "${SEED:-0}" "${SCENES_PER_ITER:-16}" "${#GPUS[@]}" <<'EOF'
import sys, yaml, numpy as np
k, seed, n, parts = map(int, sys.argv[1:])
broken = ("clipgt-1e720496", "clipgt-387a9a98")
train = [s["scene_id"] for s in yaml.safe_load(open("configs/evaluation/splits.yaml"))["train"]
         if not s["scene_id"].startswith(broken)]
pick = np.random.default_rng([seed, k]).choice(train, size=n, replace=False)
for part in np.array_split(pick, parts):
    print(",".join(part))
EOF
)
  STACKS=()
  for i in "${!GPUS[@]}"; do
    (rc=0
     DRIVER_GPU=${GPUS[$i]} DRIVER_SOURCE=external_renderer \
      ALPASIM_RENDERER_ADDRS="[localhost:$((46180 + 10 * i))]" \
      ALPASIM_DRIVER_PORT=$((6795 + i)) BASEPORT=$((12000 + 1000 * i)) \
      VAVAM_AIS_LATENT_LOG="$IT/latent_$i.jsonl" \
      DRIVER_OVERRIDES="model.checkpoint_path=$OUT/bundle_v$k.pt inference.context_length=8" \
      scripts/run_alpasim.sh "$IT/stack_$i" vavam_ais "scenes.scene_ids=[${HALVES[$i]}]" \
        runtime.simulation_config.n_rollouts="${ROLLOUTS:-4}" trafficsim=catk \
        runtime.simulation_config.force_gt_duration_us=4000000 \
        runtime.endpoints.driver.n_concurrent_rollouts=4 \
        runtime.endpoints.renderer.n_concurrent_rollouts=4 \
        eval.allow_aggregation_with_failed_rollouts=True eval.video.render_video=False \
        > "$IT/stack_$i.log" 2>&1 || rc=$?
     echo "stack_$i exit=$rc" >> "$IT/status") &
    STACKS+=($!)
  done
  wait "${STACKS[@]}"  # not the renderers, which run for the whole loop
  "$PY" -m metadrive_ais.training.latent_rl --bundle "$OUT/bundle_v$k.pt" \
    --reference "$OUT/bundle_v0.pt" --runs "$IT"/stack_* --logs "$IT"/latent_*.jsonl \
    --out "$OUT/bundle_v$((k + 1)).pt" --seed "$k" | tee "$IT/update.log"
done
