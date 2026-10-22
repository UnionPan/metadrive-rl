#!/usr/bin/env bash
# Closed-loop evaluation of one policy on one split (all scenes in one run),
# with gsplat-rendered cameras and CATK reactive traffic.
#
# Usage: scripts/evaluate.sh <policy-spec> <split> <out-dir> [rollouts-per-scene] [overrides...]
# Needs: scripts/serve_gsplat_renderer.sh running (METADRIVE_RENDERER_ADDRS).
# Then:  python -m metadrive_rl.evaluation.summarize <out-dir>
# Rollouts that fail in the simulator (e.g. CATK finds no map near the ego) are
# excluded from aggregation; compare policies with summarize --common-scenes.
set -uo pipefail

POLICY=${1:?usage: evaluate.sh <policy-spec> <split> <out-dir> [rollouts]}
SPLIT=${2:?}
OUT=$(realpath -m "${3:?}")
ROLLOUTS=${4:-1}
shift $(( $# >= 4 ? 4 : $# ))
HERE=$(cd "$(dirname "$0")" && pwd)
SPLITS=${SPLITS:-$HERE/../configs/evaluation/splits.yaml}
PY=${ALPASIM_DIR:-$HOME/ext/alpasim}/.venv/bin/python
export DRIVER_SOURCE=metadrive_external_renderer

SCENES=$("$PY" -c "
import sys, yaml
print(','.join(s['scene_id'] for s in yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]))
" "$SPLITS" "$SPLIT")

"$HERE/run_episode.sh" "$OUT" "$POLICY" \
  "scenes.scene_ids=[$SCENES]" \
  runtime.simulation_config.n_rollouts="$ROLLOUTS" \
  trafficsim=catk eval.allow_aggregation_with_failed_rollouts=True "$@"
