# Hardware — `larx`

Census and measured limits (2026-10-22). See also guideline §8.1.

## Machine

- 4× Quadro RTX 8000, 48 GB each, Turing (compute capability 7.5); driver
  580.178.04 (CUDA 13.0 driver API).
- NVLink pairs: **GPU0–GPU3** and **GPU1–GPU2** (NV2); other pairs go through
  PCIe. Put two-GPU models (Alpamayo 2 Super) on an NVLink pair.
- AMD Threadripper 3990X (128 threads), 251 GB RAM, 1.8 TB NVMe (~270 GB free).
- Docker 29.9 + Compose v5.6, uv 0.13.4. No CUDA toolkit in the system;
  a user-space CUDA 12.8 toolkit is at `~/ext/cuda128` (conda), for
  building CUDA extensions (`CUDA_HOME=~/ext/cuda128`).
- No sudo for the agent. Still missing for AlpaGym's local smoke run:
  `redis-server`, cuDNN dev headers, NCCL dev headers.

## Kernel and precision limits

- **bf16 is emulated**: a 4096² bf16 matmul is ~3× slower than fp16; Alpamayo-R1-10B
  takes 87 s per inference in bf16 vs 6.4 s in fp16. Run inference and training in
  fp16 (with gradient scaling for training).
- **FlashAttention 2 has no Turing kernels**: use SDPA.
- **NVIDIA's NRE sensorsim has no sm_75 kernels**: AlpaSim renders through the
  in-repo gsplat sensorsim (`src/metadrive_ais/render/`).

## Measured costs

| workload | GPUs | memory | speed |
|---|---|---|---|
| gsplat sensorsim (6-scene cache) | 1 | ≤ 15 GB (memory returned on eviction) | — |
| VaVAM-B driver (fp16) | 1 | small (model 1.75 GB) | 50 s episode in ~40 s wall time (smoke run) |
| Alpamayo 1 driver (fp16) | 1 | ~24 GB | 3–6 s per inference |
| Alpamayo 1.5 driver (fp16) | 1 | ~24 GB | ~5 s per inference |
| Alpamayo 2 Super driver (fp16, `layer-split`) | 2 | 33 + 35 GB | ~13 s per inference |
| AlpaSim physics + CATK per stack | shared | a few GB | the busiest GPU once several stacks share it |

## Running stacks concurrently

Each stack needs its own driver port, `BASEPORT` and renderer port
(`scripts/run_alpasim.sh` env vars). The Docker Compose project name is derived
from the log directory. `configs/alpasim/topology/single_gpu.yaml` places a whole stack on
one GPU (`ALPASIM_STACK_GPU`). Check `nvidia-smi` before adding a stack. Two renderers
plus a stack on one card has run out of memory before.
