#!/usr/bin/env bash
# Powered paired closed-loop evaluation on the eval_big split, in disk-sized
# batches: download a batch of scenes, (re)start both gsplat renderers so they
# index it, evaluate every policy on it (CATK, 2 rollouts/scene, no videos),
# then delete that batch's scene files. Policies see identical scenes.
#
# Usage: scripts/eval_powered.sh <out-dir> LABEL=POLICY_SPEC [LABEL=POLICY_SPEC ...]
# Env:   BATCH (24), ROLLOUTS (2), SPLIT (eval_big)
# Then:  python -m metadrive_rl.evaluation.summarize <out-dir>/<label> ... --common-scenes
set -uo pipefail

OUT=$(realpath -m "${1:?usage: eval_powered.sh <out-dir> LABEL=SPEC [LABEL=SPEC ...]}")
shift
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
ALPASIM_DIR=${ALPASIM_DIR:-$HOME/ext/alpasim}
PY=$ALPASIM_DIR/.venv/bin/python
SPLIT=${SPLIT:-eval_big}
BATCH=${BATCH:-24}
export TOPOLOGY=metadrive_4gpu METADRIVE_RENDERER_ADDRS='[localhost:46100,localhost:46101]'

restart_renderers() {
  for pid in $(pgrep -f "^python -m metadrive_rl.render.server"); do kill "$pid"; done
  sleep 3
  "$HERE/serve_gsplat_renderer.sh" 46100 1 > "$OUT/renderer_1.log" 2>&1 &
  "$HERE/serve_gsplat_renderer.sh" 46101 3 > "$OUT/renderer_2.log" 2>&1 &
  for _ in $(seq 120); do
    grep -q "serving on" "$OUT/renderer_1.log" 2>/dev/null \
      && grep -q "serving on" "$OUT/renderer_2.log" 2>/dev/null && return 0
    sleep 2
  done
  echo "renderers did not start" >&2; return 1
}

mkdir -p "$OUT"
n=$("$PY" -c "import sys, yaml; print(len(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]))" \
  "$REPO/configs/evaluation/splits.yaml" "$SPLIT")
for start in $(seq 0 "$BATCH" $(( n - 1 ))); do
  b=$(( start / BATCH ))
  bdir=$OUT/batches/b$b
  mkdir -p "$bdir"
  "$PY" - "$REPO/configs/evaluation/splits.yaml" "$SPLIT" "$start" "$BATCH" "$bdir/splits.yaml" <<'EOF'
import sys, yaml
splits, name, start, size, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
yaml.safe_dump({"batch": yaml.safe_load(open(splits))[name][start : start + size]}, open(out, "w"))
EOF
  echo "=== batch $b: scenes $start..$(( start + BATCH - 1 ))"
  "$PY" "$HERE/download_scenes.py" "$bdir/splits.yaml" "$ALPASIM_DIR" batch > "$bdir/download.log" 2>&1
  restart_renderers || exit 1
  for spec in "$@"; do
    label=${spec%%=*}
    policy=${spec#*=}
    if [[ -n "$(find "$OUT/$label/b$b" -path '*/aggregate/results-summary.json' 2>/dev/null)" ]]; then
      echo "$label b$b: done"; continue
    fi
    SPLITS=$bdir/splits.yaml "$HERE/evaluate.sh" "$policy" batch "$OUT/$label/b$b" \
      "${ROLLOUTS:-2}" eval.video.render_video=False > "$OUT/$label/b$b.log" 2>&1
    echo "$label b$b exit=$?"
  done
  # Free the disk for the next batch (eval_big is disjoint from all other splits).
  "$PY" - "$bdir/splits.yaml" "$ALPASIM_DIR" <<'EOF'
import sys, yaml
from pathlib import Path
for s in yaml.safe_load(open(sys.argv[1]))["batch"]:
    Path(sys.argv[2], "data/nre-artifacts/all-usdzs", f"{s['uuid']}.usdz").unlink(missing_ok=True)
EOF
done
