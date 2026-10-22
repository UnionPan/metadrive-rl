#!/usr/bin/env bash
# Dense camera frames (every 0.1 s step) for perception datasets, over the
# train and val scenes in one AlpaSim run.
#
#   expert:      the ego replays the logged drive (force-GT); rule driver.
#   closed_loop: the given policy checkpoint drives (greedy) under CATK traffic,
#                which reaches states the expert never does (e.g. near-collisions).
#
# Usage: scripts/collect_dense.sh <out-dir> expert
#        scripts/collect_dense.sh <out-dir> closed_loop <policy-checkpoint>
# Env:   DRIVER_PORT / BASEPORT / TOPOLOGY / METADRIVE_RENDERER_ADDRS (run_episode.sh)
# Then:  python -m metadrive_rl.data.build_gap_dataset <dataset-dir> <out-dir>...
set -uo pipefail

OUT=$(realpath -m "${1:?usage: collect_dense.sh <out-dir> expert|closed_loop [checkpoint]}")
MODE=${2:?}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${ALPASIM_DIR:-$HOME/ext/alpasim}/.venv/bin/python
export DRIVER_SOURCE=metadrive_external_renderer

mkdir -p "$OUT/images"
SCENES=$("$PY" -c "
import sys, yaml
s = yaml.safe_load(open(sys.argv[1]))
print(','.join(x['scene_id'] for x in s['train'] + s['val']))
" "$HERE/../configs/evaluation/splits.yaml")

case "$MODE" in
  expert)
    policy="policy: {type: rule}"
    extra=(runtime.simulation_config.force_gt_duration_us=1000000000) ;;
  closed_loop)
    policy="policy: {type: neural, checkpoint: $(realpath "${3:?checkpoint required}")}"
    extra=(trafficsim=catk) ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
cat > "$OUT/policy.yaml" <<YAML
$policy
greedy: true
decision_period_s: 0.5
image_dir: $OUT/images
image_every_step: true
YAML

"$HERE/run_episode.sh" "$OUT" "$OUT/policy.yaml" \
  "scenes.scene_ids=[$SCENES]" eval.video.render_video=False \
  eval.allow_aggregation_with_failed_rollouts=True "${extra[@]}"
