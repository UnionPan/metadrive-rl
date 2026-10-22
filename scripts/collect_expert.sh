#!/usr/bin/env bash
# Collect expert (human-log) decision data for SFT on one split.
#
# The ego replays the logged drive for the whole episode (force-GT), while the
# MetaDrive driver is still called every step: it sees gsplat-rendered frames,
# logs its 12 candidate plans per decision, and saves the decision images.
# Labels are computed offline from the executed (= logged) ego future.
#
# All scenes without a rollout under <out-dir> run in one AlpaSim invocation
# (shared startup, concurrent rollouts); evaluation videos are skipped.
#
# Usage: scripts/collect_expert.sh <splits.yaml> <split> <out-dir>
# Needs: scripts/serve_gsplat_renderer.sh running (METADRIVE_RENDERER_ADDRS).
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

PENDING=$("$PY" - "$SPLITS" "$SPLIT" "$OUT" <<'EOF'
import sys, yaml
from pathlib import Path
splits, split, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
done = {p.parent.parent.name for p in out.glob("**/rollouts/*/*/_complete")}
print(",".join(s["scene_id"] for s in yaml.safe_load(open(splits))[split]
               if s["scene_id"] not in done))
EOF
)
if [[ -z "$PENDING" ]]; then
  echo "all $SPLIT scenes collected"; exit 0
fi
echo "collecting $(tr ',' '\n' <<< "$PENDING" | wc -l) scenes"
"$HERE/run_episode.sh" "$OUT/batch_$(date +%Y%m%d_%H%M%S)" "$OUT/policy.yaml" \
  "scenes.scene_ids=[$PENDING]" \
  runtime.simulation_config.force_gt_duration_us=1000000000 \
  eval.video.render_video=False
