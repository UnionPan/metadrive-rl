# metadrive-rl

Workspace for autonomous-driving experiments with NVIDIA's Alpamayo models in
the AlpaSim closed-loop simulator, on 4× Quadro RTX 8000 (Turing, sm_75,
48 GB each).

The project is being restarted under a new plan. The previous pipeline (a
small MLP policy head over a hand-written mode-conditioned trajectory planner,
with SFT/DPO/GRPO/AIS experiments) has been removed; it is preserved under the
Git tag `archive/v0-mlp-planner`.

## What is here

- `src/metadrive_rl/render/`: a gsplat 3DGUT reimplementation of AlpaSim's
  sensorsim service. NVIDIA's NRE renderer ships no Turing kernels, so AlpaSim
  renders NuRec scenes on this hardware through this server (f-theta cameras,
  rolling shutter, dynamic actors).
- `scripts/setup_alpasim.sh`: AlpaSim at a pinned commit plus `patches/alpasim/`.
- `scripts/setup_render_env.sh`, `scripts/serve_gsplat_renderer.sh`: renderer
  environment and server (`serve_gsplat_renderer.sh <port> <gpu>`).
- `scripts/download_scenes.py`: NuRec scenes into AlpaSim's scene cache.
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
