# M3 closed-loop check: adapter bundles vs B0b* (val_cl)

Protocol: M1 (val_cl, 16 scenes x 2 rollouts, CATK, 4 s forced replay, route commands), `inference.context_length=8`, `VAVAM_MUP_SHAPES_DIR` set. Bundles: `runs/sft/abl_frame_s0` (best epoch 3), `runs/sft/abl_gru_s0` (best epoch 8). Logs and videos: `runs/ais/m3_cl/{frame_s0,gru_s0}/sim/`.

| run | scenes | rollouts | aborted | score | pass rate | at-fault collision | offroad | progress | distance (m) |
|---|---|---|---|---|---|---|---|---|---|
| B0b* | 16 | 32 | 0 | 0.247 [0.0815, 0.432] | 0.281 [0.0938, 0.469] | 0.219 [0.0625, 0.438] | 0.531 [0.312, 0.75] | 0.518 [0.364, 0.682] | 86.3 [65.8, 105] |
| B1-frame | 16 | 32 | 0 | 0.252 [0.0993, 0.43] | 0.375 [0.188, 0.562] | 0.156 [0.0312, 0.312] | 0.5 [0.312, 0.688] | 0.453 [0.326, 0.588] | 78 [58.5, 97.1] |
| B3-gru | 16 | 32 | 0 | 0.202 [0.0248, 0.4] | 0.312 [0.125, 0.562] | 0.188 [0.0312, 0.375] | 0.531 [0.312, 0.75] | 0.478 [0.335, 0.629] | 83.1 [60.6, 107] |

Paired B1-frame − B0b* over 16 matched scenes:
- score: +0.00523 [-0.139, +0.132] (10 scenes differ)
- pass rate: +0.0938 [-0.156, +0.281] (10 scenes differ)
- at-fault collision: -0.0625 [-0.188, +0.0625] (4 scenes differ)
- offroad: -0.0312 [-0.219, +0.188] (8 scenes differ)
- progress: -0.065 [-0.119, -0.0143] (15 scenes differ)
- distance (m): -8.28 [-19.5, +1.98] (16 scenes differ)

Paired B3-gru − B0b* over 16 matched scenes:
- score: -0.0448 [-0.228, +0.124] (7 scenes differ)
- pass rate: +0.0312 [-0.25, +0.281] (7 scenes differ)
- at-fault collision: -0.0312 [-0.281, +0.188] (6 scenes differ)
- offroad: +0 [-0.188, +0.188] (7 scenes differ)
- progress: -0.0404 [-0.0913, +0.00616] (15 scenes differ)
- distance (m): -3.25 [-14.5, +9.38] (16 scenes differ)

## Speed profile (`metadrive_ais.evaluation.speed_profile`)

| run | 0–4 s | 4–8 s | 8–12 s | 12–16 s | planned-on-current slope |
|---|---|---|---|---|---|
| B0a* | 8.4 / 5.7 | 6.8 / 6.1 | 6.7 / 5.9 | 6.2 / 5.5 | 0.40 |
| B0b* | 8.6 / 5.9 | 5.5 / 4.9 | 4.8 / 4.2 | 4.0 / 3.6 | 0.53 |
| B1-frame | 9.8 / 7.6 | 8.5 / 7.2 | 6.6 / 5.5 | 5.0 / 4.7 | 0.62 |
| B3-gru | 9.3 / 6.8 | 6.8 / 5.7 | 5.8 / 5.1 | 5.2 / 4.7 | 0.64 |

Cells: mean executed / planned speed (m/s) over policy queries in the bin.

## Reading

- **Null on the primary metric.** Neither adapter changes the closed-loop
  score against B0b\*: frame +0.005 [−0.139, +0.132], GRU −0.045
  [−0.228, +0.124]. 16 scenes rule out only large gains (about +0.13 for
  frame).
- **The open-loop speed gain partly transfers.** Both adapters plan and drive
  faster than B0b\*: in the first 4 s of control they plan 7.6 (frame) and
  6.8 (GRU) m/s vs 5.9. All VaVAM variants still slow down over the episode,
  and B0b\* slows the most.
- **Faster driving does not buy progress.** Frame progress is slightly lower
  (−0.065 [−0.119, −0.014]). Progress counts only up to the first incident,
  and the binding failure is offroad (50–53% for all three runs), a lateral
  error that the expert-replay flow loss barely moved (open-loop lateral
  median ≈ 0 before and after).
- **Open-loop flow loss on expert replay does not predict closed-loop
  score here.** The adapters never see the states VaVAM drives into (offroad
  drift, late corrections): the training data are human replays only.
