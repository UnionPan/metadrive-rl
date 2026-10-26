# Progress — MetaDrive-AIS (VaVAM-B + AIS)

Plan: `PROJECT_GUIDELINES.md` v1.2. Machine: `larx`, 4× Quadro RTX 8000 (Turing).
The previous MLP/planner project is archived (git tag `archive/v0-mlp-planner`).

## M0 — repo and source audit (in progress)

- AlpaSim pinned at `affc2eab` with `patches/alpasim/0001–0003`; gsplat
  sensorsim; Alpamayo 1 / 1.5 / 2 closed loop verified (README, guideline §8.1).
- VaVAM-B assets (`data/download_vavam_assets.sh --model VaVAM-B`, SHA-verified):
  `VAM_width_1024_pretrained_139k.pt` (1.75 GB), VQ tokenizer encoder/decoder
  (`~/ext/alpasim/data/drivers/vavam/`). License: code MIT, **weights
  research-only** (VideoActionModel License), nuScenes-derived data
  CC BY-NC-SA 4.0. Weights must not be redistributed.
- Remaining: `docs/SOURCES.md`, `docs/HARDWARE.md`.

## M1 — real VaVAM-B AlpaSim baseline (in progress)

**Pinhole sensorsim (done).** AlpaSim's VaVAM preset renders a synthetic
1920×1080 OpenCV-pinhole front camera with radial/tangential distortion. The
gsplat sensorsim now renders `opencv_pinhole_param` cameras through gsplat's
3DGUT pinhole path. Validation (`metadrive_ais.render.validate.pinhole_consistency`,
test `test_pinhole_matches_opencv_and_ftheta`), two val scenes:

| check | b0499b70 | 300465f7 |
|---|---|---|
| distorted render vs NumPy OpenCV projection | 48.6 dB | 43.5 dB |
| control: distortion ignored | 30.0 dB | 20.5 dB |
| pinhole vs photo-validated f-theta camera, same pose | 39.1 dB | 30.3 dB |

**Smoke episodes (done).** `scripts/run_alpasim.sh` with `configs/driver/vavam.yaml`,
`DRIVER_SOURCE=external_renderer`, scene `clipgt-b0499b70`, 1 rollout, no reactive
traffic, 4 s forced replay. B0a (`context_length=1`) and B0b
(`context_length=8`) both completed; both collide into the slow lead car
early (progress 0.04 / 0.06). VaVAM-B loads in ~10 s; a 50 s episode takes
~40 s wall time.

**Pre-declared B0a vs B0b comparison (declared before running, 2026-10-22):**
- Scenes: `val_cl` (16 scenes, `configs/evaluation/splits.yaml`, frozen v0 splits).
- 2 rollouts per scene, CATK reactive traffic (`trafficsim=catk`), forced
  replay 4 s for both (so B0b has its 8 frames when control starts), stock
  100 steps at 2 Hz (50 s).
- B0a: `inference.context_length=1` (AlpaSim preset). B0b: `inference.context_length=8` (as trained).
- Primary: paired B0b − B0a score on matched scenes (scene-cluster bootstrap,
  `metadrive_ais.evaluation.summarize --common-scenes --paired B0b B0a`).
  Secondary: at-fault collision, offroad, progress. Simulator aborts excluded and counted.
- 16 scenes resolve only large effects (§7); a null result here is not
  evidence of no effect. test_cl and eval_big stay untouched.

**Result of the pre-declared comparison** (`reports/m1_b0a_b0b_valcl.md`;
logs, metrics and 63 MP4s in `runs/vavam/m1/{b0a,b0b}/sim/`):

| | B0a (1 frame) | B0b (8 frames) | paired B0b − B0a (16 scenes) |
|---|---|---|---|
| rollouts | 31 of 32 | 32 of 32 | |
| score | 0.352 [0.142, 0.568] | 0.231 [0.063, 0.421] | −0.121 [−0.389, +0.156] |
| pass rate | 0.41 | 0.38 | −0.03 |
| offroad | **0.41** | **0.53** | +0.125 [−0.219, +0.438] |
| at-fault collision | 0.19 | 0.13 | −0.063 [−0.25, +0.125] |
| progress | 0.61 | 0.42 | **−0.19 [−0.32, −0.06]** |
| distance | 102 m | 70 m | **−31 m [−52, −11]** |

- **Context length does not resolve on score.** Native 8-frame VaVAM-B makes
  significantly less progress and drives shorter distances; score, offroad and
  collision differences are within noise on 16 scenes.
- **One B0a rollout is missing.** Scene `81a1d7cc` failed three times in
  physics ("There are only 0 points in 3 dimensions": no ground mesh under the
  ego), i.e. the policy drove off the reconstructed area. This is a
  policy-dependent loss, not a CATK abort. The launch script lacked
  `eval.allow_aggregation_with_failed_rollouts=True`, so B0a was aggregated
  afterwards with `alpasim-aggregation` and that flag
  (`runs/vavam/m1/b0a/eval-config-allow-failed.yaml`).
- The two rollouts per scene are nearly identical: CATK adds little variance
  for this policy.
- **GPU profile**: one stack per GPU (renderer + VaVAM driver + physics + CATK)
  peaked at 12.8 GiB (B0a) / 15.8 GiB (B0b), mean utilisation 12% / 38%;
  32 rollouts took 13 / 18 min. Several VaVAM stacks fit per GPU.

**Failure taxonomy: offroad dominates (41–53%), with a rightward bias.**
- Videos: on straight roads with command STRAIGHT the planned trajectory
  veers right toward the road edge (`4e05c75c`, `81a1d7cc`, `561c5a26`,
  `e244984f`). At a highway exit (`e244984f`) the route-derived command
  switches to RIGHT and VaVAM turns sharply off the road, as it would at an
  intersection. AlpaSim derives commands from a 3 m lateral offset 20 m ahead.
- **Not a pipeline error**: the preset camera points straight ahead (yaw
  −0.15°, pitch −0.38°); VaVAM's trajectories use x forward, y left
  (`vam.datalib.ego_trajectory_dataset`), as AlpaSim assumes; the command map
  (RIGHT 0, LEFT 1, STRAIGHT 2) matches; pinhole rendering is validated (above).
- **Open loop on the human path** (`metadrive_ais.evaluation.forced_replay`,
  driver queries during the 4 s forced replay, B0a, 62 predictions over 3 s):
  the final waypoint is 0.44 m right of the human (median), 79% of predictions
  right; longitudinal error is unbiased (median +0.17 m).
- **Mirror test** (`scripts/vavam_mirror_probe.py`, 248 logged frames, same
  command and noise): as logged, median −0.29 m, 81% right; mirrored left-right,
  median −0.11 m, 69% right; correlation(y, −y_mirrored) = 0.74. Most of the
  bias follows the image content, and a smaller part (about −0.2 to −0.4 m
  at 3 s) is a model prior that survives mirroring.
- Implication for M3: the dominant failure is lateral, not memory-related.
  AIS memory can only help if history carries lateral information (e.g.
  executed drift); report lateral error separately in all ablations.

## AIS conditioning interface in VaVAM-B (brief step 4)

Read from the pinned `vam` 1.0.0 code and the VaVAM-B checkpoint (AlpaSim's
loader `alpasim_driver.models.vam_model.load_inference_VAM`):

| | value |
|---|---|
| total parameters | 357.3M |
| VaViM GPT (`joint_model.gpt`) | 319.4M (318.8M non-embedding); 24 layers, width 1024; vocabulary 16384 VQ codes + 1 |
| action expert (`joint_model.action_expert`) | 37.9M; 24 layers; action-token width 256 |
| context | 8 frames × 576 tokens (32×18 VQ grid per frame) |
| action | horizon 6 waypoints × (x, y), scaled by 70; 3 navigation commands |
| sampling | flow matching, 10 Euler steps, σ_min 0.001 |

The forward pass (`JointModel.forward`): visual tokens → VaViM embeddings;
noisy actions + command + diffusion step → `ActionEncoder` → action tokens
`[B, t, 6, 256]`. Then 24 joint layers with a mutual-attention mask: visual
tokens are causal over frames and **never attend to action tokens**; action
tokens of timestep `t` attend to visual tokens of frames `≤ t` and to their own
6-token block. Finally `ln_f` → `action_decoder` → flow velocity. At inference
only one action block is predicted (attending to all context frames) and the
visual KV cache is reused across the 10 Euler steps.

**Adapter injection point (first choice in guideline §2):** a zero-initialized
gated residual `δ_t = gate(z_t) · W z_t ∈ R^256`, added to the `ActionEncoder`
output for timestep `t` and broadcast over its 6 action tokens. This is the
same place the navigation command embedding enters. Consequences:
- Gate 0 reproduces stock VaVAM-B exactly (same tensors, same noise seed),
  which is the parity test.
- With VaViM frozen, no gradient needs to flow through the 319M GPT: visual
  keys/values do not depend on action tokens, so they can be computed under
  `no_grad`. Only the action expert's activations need storing for backward.
- In the 8-timestep training forward each action block `t` gets the AIS state
  `z_t` computed from history up to `t`. This keeps teacher forcing causal.
- Driver integration: a custom AlpaSim model plugin wrapping
  `VideoActionModelInference` that owns per-session AIS state, resets it per
  episode, and sets the residual before each `forward_inference` call.

## M1 follow-up diagnostic: route-derived commands (declared before running)

Episode records (`metadrive_ais.data.episodes`, `data/episodes/m1`) show:
- AlpaSim's route is often off the human path. During the forced replay, 29% of
  steps have the route > 1 m to the right at the 20 m lookahead (`451cbe44`
  −8 m, `4e05c75c` −3 m).
- In closed loop the derived command is RIGHT on 56% of policy steps (10% on the human path).
- RIGHT is the command on 64% of the 4 steps before each first offroad step,
  and in most episodes the first RIGHT precedes the first offroad step.

Diagnostic: rerun B0a and B0b on val_cl, 2 rollouts, CATK, 4 s replay, with
`route.use_waypoint_commands=false` (command fixed to STRAIGHT). Compare
offroad and score against M1 on matched scenes. This tests the command
channel only. It is not a candidate baseline: scenes with genuine turns need
commands.
