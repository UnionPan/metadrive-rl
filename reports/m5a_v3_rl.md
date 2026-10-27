# M5a v3: latent-policy RL on stock VaVAM-B (per-step latent draws)

Policy: stock VaVAM-B (muP base shapes, 8 frames) plus a latent residual
`300 · W c_t` (W a fixed random orthonormal 256×4 map), with
`c_t ~ q_phi(c | z_t) = N(mu(z_t), diag sigma(z_t)^2)` drawn every query (2 Hz).
z_t comes from the flow-SFT frame encoder (DINOv3-S + executed motion); the
adapter gate is 0. q_phi starts at N(0, I), so the v0 mean policy is stock
VaVAM. PPO on exact log-probs: 8 iterations × 16 train scenes × 4 rollouts
(512 rollouts), AlpaSim score, scene-group baseline, lr 3e-5, target KL 0.02,
KL 0.1 to v0. Logs: `runs/rl/v3/`, `runs/rl/v3_eval/`.

## Training (`scripts/rl_progress.py runs/rl/v3`)

| iter | rollouts | aborted | mean score | signal scenes | queries | PPO epochs | approx KL | KL to v0 | prior std |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 64 | 0 | 0.199 | 10/16 | 547 | 4 | 0.0074 | 0.010 | 1.00 |
| 1 | 64 | 0 | 0.253 | 10/16 | 586 | 3 | 0.0203 | 0.024 | 0.99 |
| 2 | 63 | 1 | 0.170 | 9/16 | 482 | 4 | 0.0090 | 0.024 | 1.00 |
| 3 | 64 | 0 | 0.184 | 11/16 | 606 | 4 | 0.0036 | 0.053 | 1.01 |
| 4 | 64 | 0 | 0.130 | 9/16 | 278 | 2 | 0.0326 | 0.028 | 0.99 |
| 5 | 64 | 0 | 0.073 | 8/16 | 345 | 3 | 0.0216 | 0.033 | 0.97 |
| 6 | 64 | 0 | 0.133 | 8/16 | 421 | 4 | 0.0117 | 0.026 | 0.98 |
| 7 | 64 | 0 | 0.096 | 8/16 | 361 | 4 | 0.0083 | 0.016 | 0.98 |

Mean scores are on a different scene draw each iteration (diagnostic only). The
policy barely moved: KL to v0 ≤ 0.053, prior-head weights ~2e-4. Offline over
all iterations (292 rollouts with group signal), no latent dimension
correlates with the advantage: max |r| = 0.093, permutation p = 0.39.

## Evaluation (val_cl, M1 protocol, 2 rollouts)

| run | scenes | rollouts | aborted | score | pass rate | at-fault collision | offroad | progress | distance (m) |
|---|---|---|---|---|---|---|---|---|---|
| B0b* | 16 | 32 | 0 | 0.247 [0.0815, 0.432] | 0.281 [0.0938, 0.469] | 0.219 [0.0625, 0.438] | 0.531 [0.312, 0.75] | 0.518 [0.364, 0.682] | 86.3 [65.8, 105] |
| v0-sample | 16 | 32 | 0 | 0.175 [0.0582, 0.318] | 0.281 [0.125, 0.469] | 0.219 [0.0625, 0.406] | 0.5 [0.281, 0.688] | 0.399 [0.279, 0.541] | 70.3 [49.7, 93.2] |
| v8-sample | 16 | 32 | 0 | 0.116 [0.0328, 0.208] | 0.188 [0.0625, 0.312] | 0.219 [0.0625, 0.406] | 0.625 [0.438, 0.781] | 0.39 [0.276, 0.519] | 67.9 [50.1, 86.2] |
| v8-mean | 16 | 32 | 0 | 0.261 [0.0897, 0.452] | 0.344 [0.156, 0.562] | 0.188 [0, 0.375] | 0.5 [0.281, 0.719] | 0.509 [0.365, 0.659] | 84.6 [65.2, 103] |

Paired v8-mean − B0b* over 16 matched scenes:
- score: +0.0135 [-0.108, +0.13] (6 scenes differ)
- pass rate: +0.0625 [-0.125, +0.25] (4 scenes differ)
- at-fault collision: -0.0312 [-0.0938, +0] (1 scenes differ)
- offroad: -0.0312 [-0.219, +0.156] (3 scenes differ)
- progress: -0.00944 [-0.0658, +0.0495] (13 scenes differ)
- distance (m): -1.76 [-12.3, +7.69] (16 scenes differ)
Paired v8-sample − v0-sample over 16 matched scenes:
- score: -0.0589 [-0.153, +0.0151] (6 scenes differ)
- pass rate: -0.0938 [-0.25, +0] (2 scenes differ)
- at-fault collision: +0 [-0.0938, +0.0938] (2 scenes differ)
- offroad: +0.125 [+0, +0.281] (3 scenes differ)
- progress: -0.00987 [-0.0528, +0.0317] (16 scenes differ)
- distance (m): -2.43 [-13.8, +6.79] (16 scenes differ)
Paired v0-sample − B0b* over 16 matched scenes:
- score: -0.0721 [-0.239, +0.0655] (7 scenes differ)
- pass rate: +0 [-0.188, +0.188] (7 scenes differ)
- at-fault collision: +0 [-0.0938, +0.0938] (2 scenes differ)
- offroad: -0.0312 [-0.219, +0.188] (8 scenes differ)
- progress: -0.119 [-0.215, -0.03] (15 scenes differ)
- distance (m): -16 [-34.8, +3.07] (16 scenes differ)

## Reading

- **No RL effect.** The trained mean policy equals stock VaVAM within noise
  (+0.014 [−0.108, +0.13]), and at equal exploration noise v8 is no better
  than v0 (−0.059 [−0.153, +0.015]).
- **Exploration noise costs progress** (v0 sampled vs B0b\*: progress −0.12
  [−0.22, −0.03]), as expected for a random residual.
- **Why nothing was learned:** the per-iteration signal is weak (8–11 of 16
  scenes have non-identical group scores; incidents come ~3 s after control,
  leaving ~6 credited decisions per rollout). Per-step draws re-randomise every
  0.5 s, so a perturbation never changes behaviour consistently, and over 292
  rollouts no latent direction relates to outcome. The machinery is validated:
  exact log-probs (logged = recomputed to 7e-6), checkpoints restored by the
  AlpaSim driver, KL-controlled PPO, real closed-loop evaluation.
- Follow-up declared: v3b with latents held for 2 s (temporally persistent
  exploration).
