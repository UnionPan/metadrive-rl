#!/usr/bin/env bash
# Serve NuRec scenes to AlpaSim with the gsplat renderer (Turing-compatible).
#
# Usage: scripts/serve_gsplat_renderer.sh [port] [gpu]
# Env:   RENDER_ENV      conda env with torch 2.4 + gsplat (default metadrive-render)
#        SCENE_GLOB      default ~/ext/alpasim/data/nre-artifacts/all-usdzs/*.usdz
#        RENDER_CACHE    extracted checkpoints (default ~/ext/render-cache)
set -euo pipefail

PORT=${1:-46100}
GPU=${2:-1}
RENDER_ENV=${RENDER_ENV:-metadrive-render}
SCENE_GLOB=${SCENE_GLOB:-$HOME/ext/alpasim/data/nre-artifacts/all-usdzs/*.usdz}
RENDER_CACHE=${RENDER_CACHE:-$HOME/ext/render-cache}

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RENDER_ENV"
export CUDA_VISIBLE_DEVICES=$GPU
exec python -m metadrive_rl.render.server --port "$PORT" \
  --artifact-glob "$SCENE_GLOB" --cache-dir "$RENDER_CACHE"
