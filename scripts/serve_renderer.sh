#!/usr/bin/env bash
# Serve the NVIDIA NRE renderer for AlpaSim on an Ampere-or-newer GPU machine.
#
# NRE has no Turing (sm_75) kernels, so on RTX 8000 hosts the renderer runs on
# a second machine. It binds to localhost only; reach it via an SSH tunnel:
#   ssh -N -L 46100:localhost:46100 <render-host>
#
# Usage: scripts/serve_renderer.sh <scene-dir> [port] [gpu]
#   scene-dir  directory containing the scene .usdz files (same files as the
#              AlpaSim host's data/nre-artifacts/all-usdzs)
set -euo pipefail

SCENE_DIR=$(realpath "${1:?usage: serve_renderer.sh <scene-dir> [port] [gpu]}")
PORT=${2:-46100}
GPU=${3:-0}
IMAGE=${NRE_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}

ls "$SCENE_DIR"/*.usdz >/dev/null

exec docker run --rm --init --name "nre-${PORT}" \
  --gpus "device=${GPU}" --network host \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e OMP_NUM_THREADS=1 \
  -v "${SCENE_DIR}:/mnt/nre-data:ro" \
  --entrypoint /app/internal/scripts/pycena/runtime/pycena_nrm_full \
  "$IMAGE" serve-grpc \
  --port="$PORT" --host=127.0.0.1 \
  "--artifact-glob=/mnt/nre-data/**/*.usdz" \
  --no-enable-nrend --cache-size=3 --max-workers=2 --enable-editing-actors
