#!/usr/bin/env bash
# Closed-loop AlpaSim rollouts with an AlpaSim driver model (e.g. Alpamayo 1)
# running on this host and the gsplat renderer as the external sensorsim.
#
# Starts the driver from configs/driver/<driver>.yaml, waits for its port, then
# lets the AlpaSim wizard run physics/controller/trafficsim/runtime in Docker.
#
# Usage: scripts/run_alpasim.sh <log-dir> <driver> [wizard overrides...]
#   e.g. scripts/run_alpasim.sh runs/smoke alpamayo1 "scenes.scene_ids=[clipgt-...]"
# Env:   ALPASIM_DIR (~/ext/alpasim), DRIVER_GPU (0), TOPOLOGY (single_gpu),
#        ALPASIM_STACK_GPU (single_gpu topology; default DRIVER_GPU),
#        ALPASIM_DRIVER_PORT (6789), BASEPORT (6000),
#        ALPASIM_RENDERER_ADDRS (YAML list, default [localhost:46100]),
#        ALPAMAYO_DTYPE (float16 on Turing; see patches/alpasim/0003)
set -euo pipefail

LOG_DIR=$(realpath -m "${1:?usage: run_alpasim.sh <log-dir> <driver> [overrides...]}")
DRIVER=${2:?driver config name under configs/driver}
shift 2
ALPASIM_DIR=${ALPASIM_DIR:-$HOME/ext/alpasim}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$ALPASIM_DIR/.venv/bin/python"
export ALPASIM_DRIVER_PORT=${ALPASIM_DRIVER_PORT:-6789}
export ALPASIM_RENDERER_ADDRS=${ALPASIM_RENDERER_ADDRS:-[localhost:46100]}
export ALPASIM_STACK_GPU=${ALPASIM_STACK_GPU:-${DRIVER_GPU:-0}}

mkdir -p "$LOG_DIR"
CUDA_VISIBLE_DEVICES=${DRIVER_GPU:-0} "$PY" -m alpasim_driver.main \
  --config-path="$REPO/configs/driver" --config-name="$DRIVER" \
  output_dir="$LOG_DIR/driver" port="$ALPASIM_DRIVER_PORT" > "$LOG_DIR/driver.log" 2>&1 &
DRIVER_PID=$!
trap 'kill $DRIVER_PID 2>/dev/null || true' EXIT

# Large checkpoints take a few minutes to load before the port opens.
for _ in $(seq 600); do
  (exec 3<>/dev/tcp/127.0.0.1/"$ALPASIM_DRIVER_PORT") 2>/dev/null && break
  kill -0 $DRIVER_PID || { tail "$LOG_DIR/driver.log"; exit 1; }
  sleep 1
done

# Concurrent stacks must not share a Compose project (it defaults to "sim").
export COMPOSE_PROJECT_NAME="as-$(echo "$LOG_DIR" | md5sum | cut -c1-10)"
cd "$ALPASIM_DIR"
uv run alpasim_wizard "hydra.searchpath=[file://$REPO/configs/alpasim]" \
  deploy=local topology="${TOPOLOGY:-single_gpu}" driver="$DRIVER" \
  driver_source=external_alpamayo wizard.debug_flags.use_localhost=True \
  wizard.log_dir="$LOG_DIR/sim" wizard.baseport="${BASEPORT:-6000}" "$@"
