# Roadmap — MetaDrive-AIS

Milestones from `PROJECT_GUIDELINES.md` §9. Status as of 2026-10-22.

| milestone | status | next concrete step |
|---|---|---|
| M0 repo/source audit | done (`docs/SOURCES.md`, `docs/HARDWARE.md`) | — |
| M1 VaVAM-B baseline | done: pinhole sensorsim validated; B0a vs B0b on val_cl (score n.s.; B0b −0.19 progress); offroad 41–53% with a rightward bias (open loop and mirror test) | collision-partner breakdown (optional) |
| M2 dataset + AIS representation | not started | episode recorder (frames, commands, executed egomotion, rewards) from AlpaSim logs; DINOv3-S cache; GRU + reward/next-embedding heads |
| M3 AIS-conditioned VaVAM-B | adapter + zero-gate parity tests pass on the real checkpoint | custom AlpaSim driver plugin owning per-session AIS state; adapter SFT on expert trajectories |
| M4 ablations B0a–B6 | not started | needs M3 |
| M5a standalone latent-policy RL | not started | needs M3 |
| M5b AlpaGym interop | environment builds, unit tests pass; smoke run blocked on sudo packages | install `redis-server`, cuDNN/NCCL headers; VaVAM policy bundle (fp16) |
| M6 demo/portfolio | not started | — |

Open risks: the gsplat renderer is a reimplementation (photometric gap,
degradation off the recorded path, no pedestrians), which affects VaVAM's
inputs. Expert demonstrations for SFT need a licensed source (guideline §4).
The 16-scene val_cl detects only large effects; claims need eval_big (~100 scenes).
