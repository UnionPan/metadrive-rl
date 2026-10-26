# Roadmap — MetaDrive-AIS

Milestones from `PROJECT_GUIDELINES.md` §9. Status as of 2026-10-26.

| milestone | status | next concrete step |
|---|---|---|
| M0 repo/source audit | done (`docs/SOURCES.md`, `docs/HARDWARE.md`) | — |
| M1 VaVAM-B baseline | done. AlpaSim drops VaVAM's muP base shapes (flow velocity 4×); corrected baselines B0a\*/B0b\* are the reference. Offroad 44–53% is not caused by that bug | — |
| M2 dataset + AIS representation | done: CATK and expert-replay episode datasets with DINOv3-S features and VQ tokens; no predictive benefit from memory (frame ≈ GRU) | memory-dependent scenario tests (guideline §6) |
| M3 AIS-conditioned VaVAM-B | done: zero-gate parity, VaVAM-AIS AlpaSim driver, adapter SFT. Open-loop gain (frame ≫ GRU) but **no closed-loop gain** over B0b\* (`reports/m3_closed_loop.md`) | — |
| M4 ablations B0a–B6 | running: stack (B2) and motion-only control (B1m) seeds, B4 (frozen AIS-trained GRU) | ablation report; B6 (VaViM features) if justified |
| M5a standalone latent-policy RL | code and tests done, protocol declared (`docs/PROGRESS.md`) | latent-channel pretraining → sensitivity gate → 8 PPO iterations → val_cl v8 vs v0 |
| M5b AlpaGym interop | environment builds, unit tests pass; smoke run blocked on sudo packages | install `redis-server`, cuDNN/NCCL headers; VaVAM policy bundle (fp16) |
| M6 demo/portfolio | not started | — |

Open risks: the gsplat renderer is a reimplementation (photometric gap,
degradation off the recorded path, no pedestrians), which affects VaVAM's
inputs. The 16-scene val_cl detects only large effects; claims need eval_big
(~100 scenes). VaVAM's failures come early (median incident ~3.5 s after
control starts), which leaves few decisions per episode for credit assignment.
