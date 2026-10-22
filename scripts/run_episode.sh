#!/usr/bin/env bash
# Run closed-loop AlpaSim rollouts with the MetaDrive driver.
#
# Starts the MetaDrive driver on this host, then lets the AlpaSim wizard launch
# physics/controller/runtime in Docker against an external renderer
# (see scripts/serve_gsplat_renderer.sh).
#
# Usage: scripts/run_episode.sh <log-dir> [policy-spec] [extra wizard overrides...]
#   policy-spec  rule (default), fixed:<ACTION>, or a policy YAML path
# Env:  ALPASIM_DIR (default ~/ext/alpasim)
#       DRIVER_SOURCE (metadrive_external_renderer | metadrive_no_render)
#       DRIVER_GPU (default 2; GPU 0 runs physics/CATK, GPU 1 the renderer)
#       TOPOLOGY (1gpu | metadrive_4gpu; the latter expects two renderers, see
#                 METADRIVE_RENDERER_ADDRS in the metadrive_external_renderer preset)
#       METADRIVE_RENDERER_ADDRS (YAML list, default [localhost:46100])
set -euo pipefail

LOG_DIR=$(realpath -m "${1:?usage: run_episode.sh <log-dir> [policy-spec] [overrides...]}")
POLICY=${2:-rule}
shift $(( $# >= 2 ? 2 : 1 ))
ALPASIM_DIR=${ALPASIM_DIR:-$HOME/ext/alpasim}
export METADRIVE_RENDERER_ADDRS=${METADRIVE_RENDERER_ADDRS:-[localhost:46100]}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$ALPASIM_DIR/.venv/bin/python"

mkdir -p "$LOG_DIR"
CUDA_VISIBLE_DEVICES=${DRIVER_GPU:-2} "$PY" -m alpasim_driver.main \
  --config-path="$REPO/configs/simulator" --config-name=driver_metadrive \
  output_dir="$LOG_DIR/driver" model.checkpoint_path="$POLICY" \
  > "$LOG_DIR/driver.log" 2>&1 &
DRIVER_PID=$!
trap 'kill $DRIVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 60); do
  (exec 3<>/dev/tcp/127.0.0.1/6789) 2>/dev/null && break
  kill -0 $DRIVER_PID || { tail "$LOG_DIR/driver.log"; exit 1; }
  sleep 1
done

cd "$ALPASIM_DIR"
uv run alpasim_wizard deploy=local topology="${TOPOLOGY:-1gpu}" driver=metadrive \
  driver_source="${DRIVER_SOURCE:-metadrive_external_renderer}" wizard.debug_flags.use_localhost=True \
  wizard.log_dir="$LOG_DIR/sim" "$@"
