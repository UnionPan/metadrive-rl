# Roadmap — MetaDrive-AIS

Milestones from `PROJECT_GUIDELINES.md` §9. Status as of 2026-10-27.

| milestone | status | next concrete step |
|---|---|---|
| M0 repo/source audit | done (`docs/SOURCES.md`, `docs/HARDWARE.md`) | — |
| M1 VaVAM-B baseline | done. AlpaSim drops VaVAM's muP base shapes (flow velocity 4×); corrected baselines B0a\*/B0b\* are the reference. Offroad 44–53% is not caused by that bug | — |
| M2 dataset + AIS representation | done: CATK and expert-replay episode datasets with DINOv3-S features and VQ tokens; no predictive benefit from memory (frame ≈ GRU) | — |
| M3 AIS-conditioned VaVAM-B | done: zero-gate parity, VaVAM-AIS AlpaSim driver, adapter SFT. Open-loop gain, but closed loop: null on val_cl and **worse than B0b\*** on eval_big-48 (paired score −0.09 [−0.16, −0.02]; `reports/m3_eval_big48.md`) | — |
| M4 ablations B0a–B6 | done: B1/B1m/B2/B3/B4 open loop (`reports/m4_open_loop_ablation.md`), 2× budget check, and the §6 memory-dependent tests (`reports/m4_memory_scenarios.md`): no memory or AIS-objective benefit, even on memory-relevant windows; long GRU history hurts in closed loop. B5/B6 not run: B5 (latent context modes) is covered in spirit by the M5a latent channels, and B6 (VaViM features instead of DINOv3) has no memory or DINOv3 benefit to explain (B1m ≈ B1) | occlusion/merge/reactivity cases need scenario controls AlpaSim lacks |
| M5a standalone latent-policy RL | done (negative): valid latent-policy PPO through AlpaSim with exact log-probs, restored checkpoints, closed-loop evaluation and an equal-budget AWR comparator. Learned latent channels fail the sensitivity gate; RL on a fixed channel (per-step v3, held v3b) does not beat stock VaVAM; RL ≥ AWR | — |
| M5b AlpaGym interop | done: VaVAM-AIS policy bundle for AlpaGym (`metadrive_ais.alpagym`); an episode, scoring, verified optimizer step (PPO ratio 1 on-policy), weight sync and a rerun on the synced weights, with export back to an AlpaSim bundle (`scripts/alpagym_vavam_smoke.sh`). 2-GPU throughput measured; AlpaGym's 1-GPU colocated layout starts no rollout replica | — |
| M6 demo/portfolio | done: technical README, `reports/summary.md`, side-by-side gallery (`docs/demo_gallery.md`, videos in `outputs/demo_gallery/`), failure taxonomy, reproduce command (verified end to end), per-run artifact exports (`scripts/export_runs.py`) | optional replay UI |

Open risks: the gsplat renderer is a reimplementation (photometric gap,
degradation off the recorded path, no pedestrians), which affects VaVAM's
inputs. The 16-scene val_cl detects only large effects; claims need eval_big
(~100 scenes). VaVAM's failures come early (median incident ~3.5 s after
control starts), which leaves few decisions per episode for credit assignment.
