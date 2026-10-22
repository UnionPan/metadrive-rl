#!/usr/bin/env bash
# Create the conda env for the gsplat renderer service.
#
# gsplat publishes prebuilt CUDA wheels (with sm_75 kernels) only for
# Python 3.10 / torch <= 2.4, so the renderer gets its own env. The AlpaSim
# gRPC package declares Python >= 3.11 but is pure Python once built, so its
# wheel is built with AlpaSim's toolchain and installed here as-is.
#
# Usage: scripts/setup_render_env.sh [env-name] [alpasim-dir]
set -euo pipefail

ENV=${1:-metadrive-render}
ALPASIM_DIR=$(realpath "${2:-$HOME/ext/alpasim}")
REPO=$(cd "$(dirname "$0")/.." && pwd)
WHEELS=$(mktemp -d)

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -q -n "$ENV" python=3.10
conda activate "$ENV"
pip install -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -q gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt24cu124 \
  --extra-index-url https://pypi.org/simple
(cd "$ALPASIM_DIR" && uv build --wheel src/grpc -o "$WHEELS")
pip install -q grpcio "protobuf>=4,<5"
pip install -q --ignore-requires-python --no-deps "$WHEELS"/alpasim_grpc-*.whl
pip install -q -e "$REPO[render]" pytest
python -c "import gsplat, torch; assert 'sm_75' in torch.cuda.get_arch_list(); print('render env ready')"
