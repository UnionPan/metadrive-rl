#!/usr/bin/env bash
# Clone AlpaSim at the pinned commit, apply local patches, and build its uv
# workspace (protos, services, Alpamayo driver plugins).
#
# Usage: scripts/setup_alpasim.sh [alpasim-dir]   (default ~/ext/alpasim)
set -euo pipefail

ALPASIM_COMMIT=affc2eab209fa43bdfa2f26c0f8d437922d78a68
ALPASIM_DIR=$(realpath -m "${1:-$HOME/ext/alpasim}")
REPO=$(cd "$(dirname "$0")/.." && pwd)

if [[ ! -d "$ALPASIM_DIR/.git" ]]; then
  git clone https://github.com/NVlabs/alpasim.git "$ALPASIM_DIR"
fi
cd "$ALPASIM_DIR"
git checkout -q "$ALPASIM_COMMIT"
for patch in "$REPO"/patches/alpasim/*.patch; do
  if git apply --check "$patch" 2>/dev/null; then
    git apply "$patch"
  elif ! git apply --reverse --check "$patch" 2>/dev/null; then
    echo "patch does not apply: $patch" >&2; exit 1
  fi
done

# setup_local_env.sh compiles protos and syncs the workspace (needs uv + cargo).
bash -c 'source setup_local_env.sh'
echo "AlpaSim ready at $ALPASIM_DIR"
