# Roadmap — MetaDrive-AIS

Milestones from `PROJECT_GUIDELINES.md` §9. Status as of 2026-10-27.

| milestone | status | next concrete step |
|---|---|---|
| M0 repo/source audit | done (`docs/SOURCES.md`, `docs/HARDWARE.md`) | — |
| M1 VaVAM-B baseline | done. AlpaSim drops VaVAM's muP base shapes (flow velocity 4×); corrected baselines B0a\*/B0b\* are the reference. Offroad 44–53% is not caused by that bug | — |
| M2 dataset + AIS representation | done: CATK and expert-replay episode datasets with DINOv3-S features and VQ tokens; no predictive benefit from memory (frame ≈ GRU) | memory-dependent scenario tests (guideline §6) |
| M3 AIS-conditioned VaVAM-B | done: zero-gate parity, VaVAM-AIS AlpaSim driver, adapter SFT. Open-loop gain, but closed loop: null on val_cl and **worse than B0b\*** on eval_big-48 (paired score −0.09 [−0.16, −0.02]; `reports/m3_eval_big48.md`) | — |
| M4 ablations B0a–B6 | done for B1/B1m/B2/B3/B4 (`reports/m4_open_loop_ablation.md`): ~90% of the open-loop gain is the executed-motion input; no memory or AIS-objective benefit; GRU slower to train (16-epoch check: 0.056 vs frame 0.037). B4 at 16 epochs running | B6 not justified by these results; memory-dependent scenario tests (guideline §6) remain the open test of memory |
| M5a standalone latent-policy RL | machinery done and validated (exact log-probs, driver checkpoints, KL-controlled PPO, closed-loop eval). Learned latent channels fail the sensitivity gate (CVAE collapse; hindsight prior too narrow). Fixed-channel RL v3 on stock VaVAM: no effect (`reports/m5a_v3_rl.md`) | v3b (held latents, 2 s) running; then report |
| M5b AlpaGym interop | environment builds, unit tests pass; smoke run blocked on sudo packages | install `redis-server`, cuDNN/NCCL headers; VaVAM policy bundle (fp16) |
| M6 demo/portfolio | not started | — |

Open risks: the gsplat renderer is a reimplementation (photometric gap,
degradation off the recorded path, no pedestrians), which affects VaVAM's
inputs. The 16-scene val_cl detects only large effects; claims need eval_big
(~100 scenes). VaVAM's failures come early (median incident ~3.5 s after
control starts), which leaves few decisions per episode for credit assignment.
