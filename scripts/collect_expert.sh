#!/usr/bin/env bash
# Collect expert (human-log) decision data for SFT on one split.
#
# The ego replays the logged drive for the whole episode (force-GT), while the
# MetaDrive driver is still called every step: it sees gsplat-rendered frames,
# logs its 12 candidate plans per decision, and saves the decision images.
# Labels are computed offline from the executed (= logged) ego future.
#
# Usage: scripts/collect_expert.sh <splits.yaml> <split> <out-dir>
# Needs: scripts/serve_gsplat_renderer.sh running (METADRIVE_RENDERER_ADDR).
set -uo pipefail

SPLITS=${1:?usage: collect_expert.sh <splits.yaml> <split> <out-dir>}
SPLIT=${2:?}
OUT=$(realpath -m "${3:?}")
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${ALPASIM_DIR:-$HOME/ext/alpasim}/.venv/bin/python
export DRIVER_SOURCE=metadrive_external_renderer

mkdir -p "$OUT/images"
cat > "$OUT/policy.yaml" <<YAML
policy: {type: rule}
decision_period_s: 0.5
image_dir: $OUT/images
YAML

mapfile -t SCENES < <("$PY" -c "
import sys, yaml
for s in yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]: print(s['scene_id'])
" "$SPLITS" "$SPLIT")

for scene in "${SCENES[@]}"; do
  if compgen -G "$OUT/$scene/sim/rollouts/*/*/rollout.asl" > /dev/null; then
    echo "=== $scene: done"; continue
  fi
  echo "=== $scene"
  rm -rf "${OUT:?}/$scene"
  "$HERE/run_episode.sh" "$OUT/$scene" "$OUT/policy.yaml" \
    "scenes.scene_ids=[$scene]" \
    runtime.simulation_config.force_gt_duration_us=1000000000 \
    > "$OUT/$scene.log" 2>&1
  echo "exit=$?"
done
