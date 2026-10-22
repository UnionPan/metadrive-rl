#!/usr/bin/env bash
# Simulator-verified preference data: matched-state branches.
#
# For each alternative action, one AlpaSim run over all scenes of a split: the
# ego replays the logged drive until the branch time (identical state for all
# actions), then the fixed action drives for the horizon under CATK reactive
# traffic. AlpaSim scores every rollout (at-fault collision / offroad fail it;
# otherwise relative progress). Decision images are saved for the branch state.
#
# Usage: scripts/collect_branches.sh <splits.yaml> <split> <out-dir> [branch_s] [horizon_s]
# Env:   ACTIONS (space-separated; default below)
set -uo pipefail

SPLITS=${1:?usage: collect_branches.sh <splits.yaml> <split> <out-dir> [branch_s] [horizon_s]}
SPLIT=${2:?}
OUT=$(realpath -m "${3:?}")
BRANCH_S=${4:-4}
HORIZON_S=${5:-6}
ACTIONS=${ACTIONS:-"CRUISE@5 CRUISE@10 FOLLOW YIELD BRAKE CHANGE_LANE_LEFT@10 CHANGE_LANE_RIGHT@10"}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${ALPASIM_DIR:-$HOME/ext/alpasim}/.venv/bin/python
export DRIVER_SOURCE=metadrive_external_renderer

SCENES=$("$PY" -c "
import sys, yaml
print(','.join(s['scene_id'] for s in yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]))
" "$SPLITS" "$SPLIT")
STEPS=$(( (BRANCH_S + HORIZON_S) * 10 ))

mkdir -p "$OUT"
echo "{\"branch_s\": $BRANCH_S, \"horizon_s\": $HORIZON_S, \"split\": \"$SPLIT\"}" > "$OUT/branch.json"
for action in $ACTIONS; do
  label=$(echo "$action" | tr '@' '_')
  if compgen -G "$OUT/$label/sim/aggregate/results-summary.json" > /dev/null; then
    echo "=== $action: done"; continue
  fi
  mkdir -p "$OUT/$label/images"
  cat > "$OUT/$label/policy.yaml" <<YAML
policy: {type: fixed, action: "$action"}
decision_period_s: 0.5
image_dir: $OUT/$label/images
YAML
  echo "=== $action"
  rm -rf "${OUT:?}/$label/sim" "$OUT/$label/driver"
  "$HERE/run_episode.sh" "$OUT/$label" "$OUT/$label/policy.yaml" \
    "scenes.scene_ids=[$SCENES]" \
    runtime.simulation_config.force_gt_duration_us=$(( BRANCH_S * 1000000 )) \
    runtime.simulation_config.n_sim_steps=$STEPS \
    eval.video.render_video=False \
    trafficsim=catk \
    > "$OUT/$label.log" 2>&1
  echo "exit=$?"
done
