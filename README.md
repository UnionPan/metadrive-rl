# metadrive-rl

Workspace for autonomous-driving experiments with NVIDIA's Alpamayo models in
the AlpaSim closed-loop simulator, on 4× Quadro RTX 8000 (Turing, sm_75,
48 GB each).

The project is being restarted under a new plan. The previous pipeline (a
small MLP policy head over a hand-written mode-conditioned trajectory planner,
with SFT/DPO/GRPO/AIS experiments) has been removed; it is preserved under the
Git tag `archive/v0-mlp-planner`.

## Verified on this machine (2026-10-22)

| component | status |
|---|---|
| Alpamayo 1 (`nvidia/Alpamayo-R1-10B`) in AlpaSim | closed-loop episode runs: 1 GPU, fp16, ~3–6 s per inference |
| Alpamayo 2 Super (`nvidia/Alpamayo2-Super`, 36B) in AlpaSim | closed-loop episode runs: 2 GPUs (`ALPAMAYO_DEVICE_MAP=layer-split`), fp16, ~13 s per inference |
| Alpamayo 1.5 (`nvidia/Alpamayo-1.5-10B`) | blocked: needs access to the gated `nvidia/Cosmos-Reason2-8B` (auto-approved on request) |
| AlpaGym (`NVlabs/alpagym`) | environment builds (user-space CUDA 12.8 toolkit for `grouped-gemm`); 469 unit tests pass. The 2-GPU smoke run needs `redis-server`, cuDNN and NCCL headers (sudo), Cosmos-Reason2 access, and its policy hard-codes bf16 (emulated on Turing) |
| alpamayo-recipes (`NVlabs/alpamayo-recipes`) | repository tests pass (32). The recipes themselves target 8× H100 (SFT), ≥ 5× 80 GB (RL), or Blackwell (FP8/NVFP4 quantization) |

Smoke test (one scene, 6 s):

```bash
scripts/serve_gsplat_renderer.sh 46100 1 &
DRIVER_GPU=1 ALPAMAYO_DTYPE=float16 scripts/run_alpasim.sh runs/smoke/a1 alpamayo1 \
  "scenes.scene_ids=[clipgt-b0499b70-4b20-4389-8e0f-cd9fce07e7b2]" \
  runtime.simulation_config.n_rollouts=1 runtime.simulation_config.n_sim_steps=60
# Alpamayo 2 on two GPUs:
DRIVER_GPU=2,3 ALPASIM_STACK_GPU=1 ALPAMAYO_DTYPE=float16 ALPAMAYO_DEVICE_MAP=layer-split \
  scripts/run_alpasim.sh runs/smoke/a2 alpamayo2 ...
```

## What is here

- `src/metadrive_rl/render/`: a gsplat 3DGUT reimplementation of AlpaSim's
  sensorsim service. NVIDIA's NRE renderer ships no Turing kernels, so AlpaSim
  renders NuRec scenes on this hardware through this server (f-theta cameras,
  rolling shutter, dynamic actors).
- `scripts/setup_alpasim.sh`: AlpaSim at a pinned commit plus `patches/alpasim/`.
- `scripts/setup_render_env.sh`, `scripts/serve_gsplat_renderer.sh`: renderer
  environment and server (`serve_gsplat_renderer.sh <port> <gpu>`).
- `scripts/download_scenes.py`: NuRec scenes into AlpaSim's scene cache.
- `scripts/run_alpasim.sh`, `configs/driver/`, `configs/alpasim/`: AlpaSim with
  an AlpaSim driver model (Alpamayo 1 / 2) on this host and the gsplat renderer
  as the external sensorsim (four Alpamayo cameras at 576×328).
- `patches/alpasim/0003-...`: Alpamayo drivers on Turing: `ALPAMAYO_DTYPE`,
  SDPA attention for Alpamayo 1, and a layer-aligned two-GPU device map for
  Alpamayo 2 (`ALPAMAYO_DEVICE_MAP=layer-split`).
- `scripts/bench_alpamayo.py`: loads an Alpamayo checkpoint with SDPA attention
  and times one inference.

## Hardware notes

- FlashAttention 2 has no Turing kernels; Alpamayo checkpoints must run with
  SDPA attention.
- bf16 is emulated on Turing. Alpamayo-R1-10B takes ~87 s per inference in
  bf16 and ~6.4 s in fp16, with a ~24 GB peak in either.

## License

Code: Apache-2.0. NuRec scenes and Alpamayo checkpoints are under NVIDIA's
licenses.
