# Sources, versions and licenses

Pinned upstream code, checkpoints and data used on `larx` (checked 2026-10-22).
Local pinned commits take precedence over upstream documentation.

## Code

| component | location | version / commit | notes |
|---|---|---|---|
| AlpaSim | `~/ext/alpasim` (NVlabs/alpasim) | `affc2ea` (2026-08-18) | `scripts/setup_alpasim.sh` + `patches/alpasim/0001–0003` |
| VideoActionModel (`vam`) | AlpaSim venv | 1.0.0, git `738050ec70` | VaVAM inference and action-learning code |
| alpamayo-r1 / alpamayo1-5 / alpamayo2-super | AlpaSim venv | 0.1.0 (`11a0e01c13` / `36aeb4c593` / `59776920bb`) | Alpamayo model packages used by AlpaSim's drivers |
| AlpaGym | `~/ext/alpagym` (NVlabs/alpagym) | `972d160` (2026-07-21) | Env via `uv sync --all-packages` with `CUDA_HOME=~/ext/cuda128` |
| alpamayo-recipes | `~/ext/alpamayo-recipes` | `2e907da` (2026-10-16) | Recipes need H100/Blackwell; not run |
| torch / transformers (AlpaSim venv) | | 2.8.0 / 4.57.1 | |
| torch / gsplat (renderer env `metadrive-render`) | | 2.4.1+cu124 / 1.5.3 | `scripts/setup_render_env.sh` |

## Checkpoints

| checkpoint | source | revision / SHA-256 (prefix) | license |
|---|---|---|---|
| VaVAM-B `VAM_width_1024_pretrained_139k.pt` | valeoai/VideoActionModel release v1.0.0 | `129e887db30477d4` | **research-only** (VideoActionModel License) |
| VQ tokenizer encoder / decoder (`VQ_ds16_16384_llamagen_*.jit`) | same release | `b5e2c47d04f8ff8a` / `1606c10e8dbc6bcc` | research-only |
| Alpamayo-R1-10B | `nvidia/Alpamayo-R1-10B` | `dd4a24cace` | NVIDIA model license (non-commercial) |
| Alpamayo-1.5-10B | `nvidia/Alpamayo-1.5-10B` | `7aba8293c0` | NVIDIA model license |
| Alpamayo 2 Super (36B) | `nvidia/Alpamayo2-Super` | `00554695e7` | NVIDIA model license |
| Cosmos-Reason2-8B (A1.5 backbone config) | `nvidia/Cosmos-Reason2-8B` (gated, accepted) | `a9fae2cf89` | NVIDIA Open Model License |
| DINOv3 ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` (gated, accepted) | `5931719e67` | DINOv3 License |
| DINOv2-base | `facebook/dinov2-base` | `f9e44c814b` | Apache-2.0 |

VaVAM code is MIT; its weights are research-only, and its nuScenes-derived
data is CC BY-NC-SA 4.0. Do not redistribute the weights.

## Data

| data | source | revision | license |
|---|---|---|---|
| NuRec scenes (`.usdz`) | `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`, suite `26.04` | dataset `be4644840b` | NVIDIA AV Dataset License (gated) |
| Scene splits | `configs/evaluation/splits.yaml` (frozen v0, seed 0) | train 64 / val 4 / test 4 / val_cl 16 / test_cl 16 / eval_big 96 | — |

152 scenes are on disk (`~/ext/alpasim/data/nre-artifacts/all-usdzs`, ~1.6 GB
each): all of train/val/test/val_cl/test_cl and 48 of eval_big.
