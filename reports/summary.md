# MetaDrive-AIS: results so far (2026-10-27)

Question (`PROJECT_GUIDELINES.md`): does an approximate-information-state
(AIS) memory, trained on frozen DINOv3 features and executed actions, make
VaVAM-B a better closed-loop driver in AlpaSim, beyond its own 8-frame
context, and can simulator-reward learning improve it further?

Short answer on this hardware and data: **no.** The reasons are measured, not
assumed. Every comparison below was declared in `docs/PROGRESS.md` before it
ran; negative results are kept.

## Setup facts that shape everything

- **AlpaSim drops VaVAM's muP base shapes**, so the shipped driver runs the
  flow head with a 4× velocity. Corrected baselines B0a\*/B0b\*
  (`reports/m1_corrected_baselines.md`) are the reference.
- **VaVAM-B fails mostly by leaving the road**: offroad in 44–53% of val_cl
  rollouts, with an open-loop rightward bias. Incidents come early (median
  ~3.5 s after control starts).
- **Conditioning VaVAM's action expert needs large residuals.** Action tokens
  have norm ~4e3; the 3 s waypoint moves ~0.2 m per 100 units of residual.
- The renderer is our gsplat reimplementation of NVIDIA's sensorsim (no Turing
  kernels for NRE). Two eval_big scenes are not servable by it.

## Results

| question | result | evidence |
|---|---|---|
| Does memory predict outcomes better than the current frame? (M2) | No: frame ≈ GRU on collision/offroad AUROC and next-embedding NLL | `reports/m2_memory_comparison.md` |
| Does an adapter on VaVAM's action expert improve imitation? (M3) | Yes, open loop: held-out flow loss 0.086 → 0.037, ADE 3.34 → 2.25 m | `reports/m4_open_loop_ablation.md` |
| Where does that gain come from? (M4) | ~90% from the executed-motion input (speed), which VaVAM never sees; DINOv3 adds a little; memory (stack, GRU) and the AIS objective (B4) add nothing, also at 2× budget | `reports/m4_open_loop_ablation.md` |
| Does it drive better? (M3, closed loop) | No. val_cl: +0.005 [−0.14, +0.13]. **eval_big-48: −0.088 [−0.161, −0.023]**, worse than stock B0b\* | `reports/m3_closed_loop.md`, `reports/m3_eval_big48.md` |
| Can a learned latent action channel be trained for RL? (M5a) | Not with the gate met: the CVAE collapses (2 configs), and the hindsight PCA channel carries information but its prior explores only 0.14 m | `docs/PROGRESS.md` (M5a) |
| Does closed-loop RL through a latent residual improve stock VaVAM? (M5a v3) | No: v8 mean policy vs B0b\* +0.014 [−0.108, +0.13]; per-step exploration shows no learnable signal (max \|r\| = 0.09, p = 0.39) | `reports/m5a_v3_rl.md` |
| Same with temporally persistent exploration (v3b) | No: v8 mean policy vs B0b\* −0.025 [−0.21, +0.14]; held perturbations hurt (v0 held −0.21), and RL only learned to perturb less | `reports/m5a_v3b_rl.md` |
| Equal-budget non-RL comparator (AWR on the v3b rollouts) | Worse than stock (−0.09 [−0.28, +0.07]); RL ≥ AWR (+0.07 [−0.00, +0.15]) | `reports/m5a_v3b_rl.md` |

## What works (reusable)

- VaVAM-B, Alpamayo 1 / 1.5 / 2 running closed loop in AlpaSim on Turing GPUs
  (fp16, SDPA, layer split), with the corrected VaVAM loader.
- An AlpaSim driver plugin (`vavam_ais`): per-session memory, zero-gate parity
  with stock VaVAM, seeded latent policies with exact log-prob logging.
- Episode datasets from AlpaSim logs (frames, commands, executed motion,
  outcomes), DINOv3/VQ caches, flow-matching SFT through the frozen expert,
  scene-cluster bootstrap evaluation with paired comparisons and abort
  accounting, and an on-policy PPO loop over AlpaSim rollouts.

Failure taxonomy: `reports/failure_taxonomy.md`. Demo gallery: `docs/demo_gallery.md`.

## Open directions

- **Memory-dependent scenarios** (guideline §6): val_cl scenes are short and
  rarely need history, so the memory question is not yet tested where it should
  matter.
- **Offroad is the binding failure**, and it is lateral. Neither imitation of
  human replays nor small-budget RL moved it. Dense lateral or route-following
  rewards, or on-policy corrective data (DAgger-style, from VaVAM's own failure
  states), are the natural next levers.
- The full eval_big (48 more scenes, ~77 GB) and test_cl stay unused.
