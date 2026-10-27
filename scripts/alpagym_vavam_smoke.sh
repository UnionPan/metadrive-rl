#!/usr/bin/env bash
# M5b: a VaVAM-AIS latent-policy GRPO run through AlpaGym on this host
# (experiment vavam_ais_local_2gpu_smoke from metadrive_ais.alpagym.configs).
#
# Prerequisites (docs/PROGRESS.md, M5b):
#   - AlpaGym at ~/ext/alpagym with vam, mup, timm, lightning, torchmetrics,
#     torchdata, matplotlib, pyquaternion and this package installed (--no-deps)
#   - a dedicated AlpaSim copy at ~/ext/alpasim-alpagym (AlpaGym `uv sync`s it)
#     with patches/alpasim/0006 (driver_source external_dynamic_renderer)
#   - redis-server in ~/ext/tools-env (conda-forge), the gsplat renderer serving
#     on $RENDER_PORT, and the policy exported with
#     `python -m metadrive_ais.alpagym.model <bundle.pt> <checkpoint-dir>`
#
# Usage: scripts/alpagym_vavam_smoke.sh <checkpoint-dir> [extra Hydra overrides]
# Env:   GPUS (Cosmos policy + rollout GPUs, default 1,2), RENDER_PORT (46300)
set -euo pipefail
CKPT=$(realpath "${1:?usage: alpagym_vavam_smoke.sh <checkpoint-dir> [overrides...]}")
shift
NV=$HOME/ext/alpagym/.venv/lib/python3.12/site-packages/nvidia
export PATH=$HOME/ext/tools-env/bin:$PATH
export CPATH=$NV/cudnn/include:$NV/nccl/include
export VAVAM_MUP_SHAPES_DIR=$HOME/ext/alpasim/data/drivers/vavam/mup_shapes
export CUDA_VISIBLE_DEVICES=${GPUS:-1,2}
export ALPASIM_RENDERER_ADDRS="[localhost:${RENDER_PORT:-46300}]"
cd "$HOME/ext/alpagym"
exec uv run --no-sync --all-packages python -m alpagym_host.cli \
  experiment=vavam_ais_local_2gpu_smoke policy.model.path="$CKPT" reward=progress_safety "$@"
