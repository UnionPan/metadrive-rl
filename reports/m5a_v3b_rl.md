# M5a v3b: latent-policy RL with temporally persistent exploration

As v3 (`reports/m5a_v3_rl.md`): stock VaVAM-B plus `300 · W c` with a fixed
random orthonormal W, prior q_phi(c | z_t) starting at N(0, I), and the same PPO
settings and scene draws. The difference: `VAVAM_AIS_LATENT_HOLD=4`, so a latent
is drawn every 4 queries (2 s) and held, with log-probs only at decision points
(an option policy). Logs: `runs/rl/v3b/`, `runs/rl/v3b_eval/` (sampled runs hold
latents for 4 queries).

## Training

| iter | rollouts | aborted | mean score | signal scenes | queries | PPO epochs | approx KL | KL to v0 | prior std |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 64 | 0 | 0.123 | 7/16 | 108 | 2 | 0.0646 | 0.028 | 0.98 |
| 1 | 64 | 0 | 0.278 | 14/16 | 214 | 2 | 0.0246 | 0.049 | 0.94 |
| 2 | 63 | 1 | 0.116 | 8/16 | 104 | 1 | 0.0214 | 0.098 | 0.93 |
| 3 | 64 | 0 | 0.167 | 11/16 | 177 | 1 | 0.0392 | 0.157 | 0.93 |
| 4 | 64 | 0 | 0.139 | 9/16 | 111 | 1 | 0.0624 | 0.264 | 0.91 |
| 5 | 64 | 0 | 0.076 | 9/16 | 125 | 1 | 0.0365 | 0.108 | 0.94 |
| 6 | 64 | 0 | 0.109 | 8/16 | 118 | 3 | 0.0302 | 0.145 | 0.96 |
| 7 | 64 | 0 | 0.182 | 10/16 | 133 | 4 | 0.0271 | 0.045 | 0.95 |

Offline over all iterations (decision points, 304 rollouts with group
signal): no latent direction correlates with the advantage (max |r| = 0.104,
p = 0.24). Larger held perturbations tend to score worse (corr(|c|, A) = −0.11,
p = 0.057); per-step v3 shows neither (+0.05, p = 0.45).

## Evaluation (val_cl, M1 protocol, 2 rollouts)

| run | scenes | rollouts | aborted | score | pass rate | at-fault collision | offroad | progress | distance (m) |
|---|---|---|---|---|---|---|---|---|---|
| B0b* | 16 | 32 | 0 | 0.247 [0.0815, 0.432] | 0.281 [0.0938, 0.469] | 0.219 [0.0625, 0.438] | 0.531 [0.312, 0.75] | 0.518 [0.364, 0.682] | 86.3 [65.8, 105] |
| v0-held | 16 | 32 | 0 | 0.0355 [0, 0.08] | 0.0938 [0, 0.188] | 0.25 [0.0938, 0.469] | 0.656 [0.438, 0.844] | 0.354 [0.257, 0.465] | 63.2 [45.7, 81.4] |
| v8-held | 16 | 32 | 0 | 0.115 [0.0124, 0.259] | 0.156 [0.0312, 0.312] | 0.25 [0.0938, 0.438] | 0.594 [0.375, 0.781] | 0.376 [0.266, 0.506] | 67.3 [47.9, 88.5] |
| v8-mean | 16 | 32 | 0 | 0.222 [0.0662, 0.4] | 0.281 [0.0938, 0.5] | 0.188 [0.0312, 0.375] | 0.531 [0.312, 0.75] | 0.476 [0.336, 0.62] | 78 [60.2, 94.7] |

Paired v8-mean − B0b* over 16 matched scenes:
- score: -0.0247 [-0.209, +0.141] (6 scenes differ)
- pass rate: +0 [-0.25, +0.219] (5 scenes differ)
- at-fault collision: -0.0312 [-0.188, +0.156] (4 scenes differ)
- offroad: +0 [-0.188, +0.188] (4 scenes differ)
- progress: -0.0426 [-0.138, +0.0433] (15 scenes differ)
- distance (m): -8.28 [-24.3, +5.44] (16 scenes differ)
Paired v8-held − v0-held over 16 matched scenes:
- score: +0.0792 [-0.036, +0.232] (6 scenes differ)
- pass rate: +0.0625 [-0.0938, +0.25] (5 scenes differ)
- at-fault collision: +0 [-0.156, +0.156] (6 scenes differ)
- offroad: -0.0625 [-0.25, +0.0938] (5 scenes differ)
- progress: +0.0216 [-0.0107, +0.0614] (16 scenes differ)
- distance (m): +4.1 [-0.984, +9.99] (16 scenes differ)
Paired v0-held − B0b* over 16 matched scenes:
- score: -0.212 [-0.408, -0.038] (8 scenes differ)
- pass rate: -0.188 [-0.406, +0.0312] (7 scenes differ)
- at-fault collision: +0.0312 [-0.0625, +0.126] (3 scenes differ)
- offroad: +0.125 [-0.125, +0.375] (9 scenes differ)
- progress: -0.164 [-0.263, -0.0761] (16 scenes differ)
- distance (m): -23.1 [-38.9, -9.45] (16 scenes differ)

## Reading

- **No improvement over stock VaVAM:** the v8 mean policy scores −0.025
  [−0.21, +0.14] vs B0b\*.
- **Held exploration is costly:** persistent ~1 m perturbations drop the score
  from 0.25 to 0.04 (−0.21 [−0.41, −0.04]), mostly through offroad (66%). The
  channel does change driving; per-step dithering (v3) did not.
- **What RL learned is "perturb less":** v8 with held sampling recovers part of
  that cost (+0.08 [−0.04, +0.23]; prior std 1.00 → ~0.93). The only signal in
  the data is the perturbation's size, not a direction that improves on stock
  behaviour.
- In this channel, at 512 rollouts, stock VaVAM is a local optimum: deviations
  along these fixed residual directions mostly hurt. Improving it would need a
  channel aligned with the failure (lateral position relative to the route) or
  a denser, lateral-aware reward. A blind random residual of this budget does
  not find one.
