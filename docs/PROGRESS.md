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

**Result (`reports/m1_command_diagnostic.md`): negative. Route commands are
not the cause.** With the command fixed to STRAIGHT, offroad goes *up* and
score *down* on matched scenes:

| | Δ score | Δ offroad | Δ at-fault collision |
|---|---|---|---|
| B0a straight − route | −0.075 [−0.231, +0.056] | +0.063 [0, +0.156] | 0 |
| B0b straight − route | −0.047 [−0.141, 0.000] | +0.156 [0, +0.344] | 0 |

The route-derived commands carry real road geometry (curves, exits). The
association between RIGHT and offroad was correlational. The failure source
remains VaVAM-B's rightward bias. **Baselines keep the stock route commands.**
Scene `81a1d7cc` again fails in physics (no ground under the ego) for B0a.

## M2 — dataset and AIS representation (in progress)

**Built:** episode recorder (`metadrive_ais.data.episodes`, leakage test) and
frozen DINOv3 feature cache (`metadrive_ais.vision`; DINOv3-S
`facebook/dinov3-vits16-pretrain-lvd1689m` rev `114c1379`, 448×256,
[CLS, mean], 768-d, fp16; 52 frames/s on one RTX 8000).

**Collection (declared 2026-10-26), train split only (64 scenes):**
1. `train_b0b_catk`: stock VaVAM-B B0b (8 frames, preset route commands),
   CATK, 2 rollouts per scene: on-policy states, executed actions and
   per-step outcomes for the reward and next-observation objectives.
2. `train_expert_replay`: the whole episode forced along the human path
   (VaVAM-B B0b still queried), 1 rollout per scene: frames along the human
   path and the human future trajectory as flow-matching SFT targets.

3. `train_b0a_catk` (added 2026-10-26 when GPUs freed up): stock VaVAM-B B0a
   (1 frame), CATK, 2 rollouts per scene. A second behaviour policy gives the
   action diversity the action-conditioned AIS objectives need.

val, val_cl, test, test_cl and eval_big stay out of all training data.

## M3 — AIS-conditioned VaVAM-B (groundwork)

- `metadrive_ais.models.vavam_conditioning`: zero-initialized gated residual
  on VaVAM's action tokens. On the real checkpoint (`tests/test_vavam_conditioning.py`):
  gate 0 is bit-identical to stock inference, an open gate changes the
  trajectory, and the flow-matching loss reaches only the adapter.
- `metadrive_ais.drivers.vavam_ais`: AlpaSim driver plugin `vavam_ais`
  (entry point), configured by an AIS bundle (VaVAM checkpoint + DINOv3
  encoder + memory + adapter). Keeps per-session memory using the session id
  from `patches/alpasim/0004`. Tests (`tests/test_vavam_ais_driver.py`, real
  frames): zero-gate driver output equals the stock `vam` driver; interleaved
  sessions equal isolated ones; with an open gate, history changes the trajectory.
- Memory `step()` reproduces the training-time sequence forward for frame,
  stack and GRU encoders (`test_step_matches_sequence_forward`).
- Closed-loop smoke (`runs/ais/smoke_zero_gate`, zero-gate bundle,
  `clipgt-b0499b70`): runs end to end. Its outcome (pass, 0.89 progress)
  differs from three stock runs (collision at progress 0.04–0.06). The runs
  match step by step for 16 steps (speeds, plans, obstacle distance), and at
  step 15 the ego is 0.1–0.2 m from the lead car in all of them. The split is
  a knife-edge near-collision, not a parity bug. Single smoke runs cannot be
  compared; only matched multi-scene evaluations can.


## Finding: AlpaSim runs VaVAM-B with a 4× flow velocity (muP base shapes dropped)

While checking the SFT loss (stock VaVAM-B scored 9.4 on human futures and
8.6 even on its own samples, worse than predicting zero), the velocity field
turned out to be 4× too large. AlpaSim's `load_inference_VAM` sets
`gpt_mup_base_shapes` / `action_mup_base_shapes` to `None`, but the
checkpoint was trained with `mup_shapes/actionexpert_24layers_baseattentiondim256_baseembeddingdim64.bsh`
and `gpt2_24layers_basewidth256.bsh` (valeoai/VideoActionModel `738050ec70`,
the same commit as the installed `vam`). Without them muP's readout width
multiplier is 1 instead of 4 (action width 256 / base 64).

Expert-replay windows (train scenes, 8-frame context; `metadrive_ais.models.vavam_loading`):

| | AlpaSim loader | with base shapes |
|---|---|---|
| velocity at t=0 (noise scale 0.77) | 3.10 | **0.77** |
| flow loss on human futures | 8.77 | **0.018** |
| open-loop single-sample ADE / FDE | 2.72 / 4.89 m | 3.20 / 5.74 m |

The model as trained fits human driving very well (flow loss 0.018). The
inflated velocity collapses sampling toward the conditional mean, which
lowers single-sample ADE but is not the model as trained.

Consequences:
- `patches/alpasim/0005`: `VAVAM_MUP_SHAPES_DIR` makes AlpaSim load the base
  shapes (unset = AlpaSim's original behaviour). Files:
  `~/ext/alpasim/data/drivers/vavam/mup_shapes/`.
- All M1 results describe **VaVAM-B as shipped by AlpaSim** (mis-scaled).
- The first SFT smoke run (`runs/sft/smoke_gru`) trained against the
  mis-scaled model and is void. Flow SFT now loads VaVAM with the base shapes.
- VaVAM-AIS bundles record this and the driver refuses to run them without
  `VAVAM_MUP_SHAPES_DIR`.
- The M2 collections in progress use the as-shipped VaVAM as the behaviour
  policy. That is fine for representation learning (observations, executed
  actions, outcomes), but they are not on-policy for the corrected model.

**Corrected baselines (declared before running):** B0a\* and B0b\*, the M1
protocol (val_cl, 2 rollouts, CATK, 4 s replay, route commands) with
`VAVAM_MUP_SHAPES_DIR` set. These become the reference for every AIS
comparison. The as-shipped M1 results stay as a separate row.

**Corrected baselines result** (`reports/m1_corrected_baselines.md`; 32/32
rollouts each, no failures):

| | B0a (shipped) | B0a\* | B0b (shipped) | B0b\* |
|---|---|---|---|---|
| score | 0.352 | 0.437 | 0.231 | 0.247 |
| pass rate | 0.41 | 0.44 | 0.38 | 0.28 |
| offroad | 0.41 | 0.44 | 0.53 | 0.53 |
| at-fault collision | 0.19 | 0.13 | 0.13 | 0.22 |
| progress | 0.61 | 0.63 | 0.42 | 0.52 |

- Paired corrected − shipped: B0a score +0.085 [−0.026, +0.24]; B0b score
  +0.016 [−0.17, +0.22], progress +0.10 [−0.003, +0.20]. Not resolved on 16 scenes.
- B0b\* − B0a\*: score −0.19 [−0.43, +0.02]; progress **−0.11 [−0.20, −0.03]**.
  Native 8-frame context still drives less far.
- **Offroad (44–53%) does not come from the muP bug.** The open-loop rightward
  bias shrinks but remains: B0a\* final lateral median −0.26 m (72% right) vs
  −0.44 m (79%) as shipped (`forced_replay`). B0a\* also predicts slower
  driving (longitudinal median −2.5 m).
- B0a\* and B0b\* replace B0a/B0b as the reference for all AIS comparisons.

**M2 datasets built** (`data/episodes/`, DINOv3-S features, VQ tokens):
`train_b0b_catk` 124 episodes / 4,836 steps / 62 scenes;
`train_b0a_catk` 119 / 4,641 / 60; `train_expert_replay` 62 / 2,418 / 62.
Closed-loop policy steps: offroad 0.19 / 0.16, collision 0.10 / 0.09.
Failures: the two broken-route scenes (`1e720496`, `387a9a98`) plus
occasional failed attempts on `577a5904`, `a72b3cac`, `623d5516`.

**M2 representation comparison (declared before running):**
frame (no memory), stack-4 and GRU memory, identical heads and training,
trained on `train_b0b_catk` + `train_b0a_catk`, the same 12 held-out train
scenes, 3 seeds each, 300 epochs, best epoch by held-out total loss.
Primary metrics: held-out next-step collision and offroad AUROC, and
next-embedding NLL. Memory helps only if GRU/stack beat frame beyond the seed spread.

**M2 result (`reports/m2_memory_comparison.md`): no measurable memory benefit
for prediction.** 197 train / 46 held-out episodes (12 held-out train scenes), 3 seeds:

| memory | collision AUROC | offroad AUROC | next-embed NLL | next-embed cosine |
|---|---|---|---|---|
| frame | 0.70 ± 0.10 | 0.78 ± 0.04 | **−0.160** | 0.735 |
| stack-4 | 0.55 ± 0.10 | 0.76 ± 0.05 | −0.138 | 0.724 |
| GRU | 0.69 ± 0.07 | 0.76 ± 0.05 | −0.150 | 0.741 |

- Frame-only matches the GRU on outcome prediction and predicts the next
  embedding slightly better. On this data the current frame plus executed
  motion (which includes speed) is about as informative as the history.
- Low power: 18 collision and 19 offroad events in the held-out set; AUROC
  spreads ±0.05–0.10 across seeds.
- Behaviour data come from the as-shipped VaVAM (two context lengths). The
  scenes are short (~20 s, 39 queries), with few situations where history
  matters (guideline §6 lists the memory-dependent tests still to build).

## M3 — first adapter SFT (flow loss only, VaVAM with muP base shapes)

`runs/sft/gru_b3_s0`: GRU memory + gated adapter, flow-matching loss on
expert-replay windows, VaVAM frozen, 6 epochs; held-out = 12 train scenes (312 windows):

| epoch | held-out flow loss | open-loop ADE / FDE (m) | gate |
|---|---|---|---|
| 0 (= stock VaVAM-B\\*) | 0.0858 | 3.34 / 5.69 | 0 |
| 6 | 0.0726 (−15%) | 3.00 / 5.26 | 0.25 |

Lateral median stays ~0. Still improving at epoch 6.

**Caveat:** the adapter gives VaVAM the executed ego motion (incl. speed),
which VaVAM-B never sees (images + command only). The gain may come from
that input rather than from memory.

**Ablation (declared before running):** memory ∈ {frame (current frame +
executed motion, no history), stack-4, GRU}, flow loss only, 8 epochs, the
same 12 held-out scenes; frame and GRU with 2 seeds, stack with 1. Memory
helps only if GRU/stack beat frame on held-out flow loss and ADE beyond the
seed spread. The report now also gives longitudinal error (speed bias).

**M3 closed-loop check (declared before running, 2026-10-26):** do the
open-loop gains carry over to driving? Bundles from `abl_frame_s0` (best
epoch 3) and `abl_gru_s0` (best epoch 8), seed 0 fixed in advance (no
seed picking), converted with `python -m metadrive_ais.drivers.vavam_ais`.
- Protocol identical to B0b\*: val_cl, 2 rollouts, CATK, 4 s forced replay,
  route commands, `inference.context_length=8`, `VAVAM_MUP_SHAPES_DIR` set.
- Primary: paired score vs B0b\* (`summarize --common-scenes --paired`).
  Secondary: progress (the open-loop gain is mostly longitudinal), offroad,
  at-fault collision.
- Checked before launch: the driver's executed motion uses the same poses as
  training (pose timestamps equal query times, no later poses), and frames are
  the same JPEG bytes decoded to RGB.
- 16 scenes resolve only large effects; a null result is not evidence of no
  effect. test_cl and eval_big stay untouched.

**Motion-only control B1m (declared before running, 2026-10-26):** every
adapter (B1–B4) receives the executed motion u_{t-1}, which includes speed,
and VaVAM-B never sees speed. `--memory motion` uses u_{t-1} alone (no DINOv3
features, no memory) with the same mixer, adapter and protocol as the
ablation (8 epochs, 12 held-out scenes, 2 seeds). If motion ≈ frame on
held-out flow loss and ADE (within seed spread), the open-loop gain is the
speed input, not visual features or memory. Queued on GPUs 0/1 after the
stack runs.

**M3 closed-loop result (`reports/m3_closed_loop.md`; 32/32 rollouts each):**

| | B0b\* | B1-frame | B3-GRU |
|---|---|---|---|
| score | 0.247 | 0.252 | 0.202 |
| paired score vs B0b\* | | +0.005 [−0.139, +0.132] | −0.045 [−0.228, +0.124] |
| offroad | 0.53 | 0.50 | 0.53 |
| progress (paired) | | −0.065 [−0.119, −0.014] | −0.040 [−0.091, +0.006] |
| planned speed, first 4 s of control | 5.9 m/s | 7.6 | 6.8 |

- No closed-loop gain from either adapter. The open-loop gain (mostly
  longitudinal) shows up as faster driving, but failures are dominated by
  offroad (lateral), which expert-replay SFT does not touch.
- New diagnostic: `metadrive_ais.evaluation.speed_profile` (executed vs
  planned speed by time since control start).
- Consequence for the plan: training only on human replays cannot correct
  VaVAM's own failure states. The next lever is data from VaVAM's closed-loop
  states: AIS losses on the CATK collections (B4, offroad/collision heads), and
  then closed-loop RL (M5a) whose reward sees offroad directly.

## M4 — ablations

**B4 (declared before running, 2026-10-26):** GRU memory trained with the AIS
losses on closed-loop CATK data (M2 checkpoints `runs/ais/m2_gru_s{0,1}`),
frozen; only the adapter trains, with the flow loss on expert replay (8
epochs, the M3 protocol). Seed i uses `m2_gru_s{i}`. M2 and flow-SFT hold out
the same 12 scenes, so the memory never saw the SFT validation scenes. With
expert replay, the collision/offroad heads would see almost no positives,
which is why the AIS objective comes from the CATK data. Compare with B3 (GRU
trained end-to-end by the flow loss) and B1 (frame): the AIS objective helps
the action head only if B4 beats B3 on held-out flow loss and ADE beyond the
seed spread. Caveat: B4 trains fewer parameters (adapter only).

## M5a — latent-policy RL (declared before running, 2026-10-26)

Design (guideline Stage E, "stochastic AIS-conditioned latent control"):
the adapter residual gets `W_c c_t` with `c_t ~ q_phi(c | z_t)`, a 4-d
diagonal Gaussian (`models/latent_policy.py`). The channel is learned
CVAE-style during flow SFT: a posterior sees the human's next 6 waypoints,
the flow loss is the reconstruction, plus `0.01 · KL(posterior || prior)`.
RL then trains only the prior: memory, adapter, decoder and VaVAM are frozen,
so the logged `z_t` is exact and `log q_phi(c_t | z_t)` is recomputed
for PPO (`training/latent_rl.py`, `scripts/latent_rl_loop.sh`).

1. **Channel pretraining:** `flow_sft --memory frame --latent-dim 4
   --kl-weight 0.01`, 8 epochs, seed 0 (frame: the best memory in M2/M3).
2. **Gate before any RL:** prior latents must move the trajectory
   (`latent_spread_final_m` ≥ 0.5 m at the 3 s waypoint with the flow noise
   fixed), and the posterior must fit the human better than prior samples
   (`flow_loss_posterior` < `flow_loss`). If it fails: one retry with
   `--kl-weight 0.001 --latent-dim 8`, then stop and report.
3. **RL:** 8 iterations. Each draws 16 train scenes (seeded, without the
   two broken-route scenes) × 4 rollouts, CATK, 4 s replay, 8-frame
   context. Reward: AlpaSim score per rollout, scene-group baseline. Credit:
   queries after the replay up to the first incident. PPO: 4 epochs, clip 0.2,
   lr 3e-4, `0.1 · KL(q_phi || q_v0)`. Checks: the logged log-probs match the
   recomputed ones (≤ 1e-3) and the policy versions match.
4. **Evaluation (val_cl, M1 protocol):** v0 (supervised prior, sampled
   latents) vs v8 (after RL). Primary: paired score v8 − v0 (the RL effect,
   same architecture and supervised budget). Both are also compared with B0b\*.
   Per-iteration training scores are on different scenes and only diagnostic.

**M4 budget check (declared before running, 2026-10-26):** B3 (GRU) and B4
were still improving at epoch 8 (best epoch 7–8), while frame/stack peaked
at epoch 2–3. B3 is rerun for 16 epochs, seed 0, otherwise the same protocol.
If it approaches frame's held-out flow (0.037), the 8-epoch memory result is a
budget artifact. B4 at 16 epochs follows if B3 moves materially.

**M4 open-loop ablation result (`reports/m4_open_loop_ablation.md`, 2 seeds each):**
held-out flow loss: motion-only 0.041, frame 0.037, stack 0.037, GRU 0.069,
B4 (frozen AIS GRU) 0.073; stock 0.086. About 90% of the adapter's gain comes
from the executed motion (speed) that VaVAM-B never sees; DINOv3 adds a little,
and history nothing. The AIS objective does not help the action head. Caveat:
the GRUs were still improving at epoch 8 (16-epoch check running).

**M5a amendment before any RL run (2026-10-26), from a plumbing smoke test**
(1 iteration, 2 train scenes × 2 rollouts, a provisional epoch-2 latent
checkpoint; not a result). The driver logged 128 draws, and the recomputed
log-probs matched to 7e-6. Changes:
- `latent_rl_loop.sh` waited on its renderer and hung after the rollouts; it
  now waits only on the AlpaSim stacks.
- PPO lr 3e-4 → **3e-5**: z_t is unnormalised, and a single Adam step at 3e-4
  moved the prior by approximate KL 0.12. Added early stopping at approximate KL
  > 0.02 (estimator `mean((r−1) − log r)`).
- Observed: that checkpoint's prior std sits at the log-std clamp (e¹), so
  the std gets no gradient. RL can move only the latent means (fixed
  exploration noise). Reported as `prior_std_at_clamp` in the update stats.

**M5a step 1 result: the CVAE latent channel fails the gate (posterior
collapse).** `runs/sft/m5a_latent_s0`, best epoch 3: latent spread 0.056 m at
3 s (gate ≥ 0.5 m), posterior flow 0.0466 vs prior 0.0468; KL fell to 0.006
by epoch 8. As an adapter it still learns (flow 0.047, like frame). Diagnosis:
flow gradients into the latent path are tiny (decoder ~1e-6 vs prior-NLL ~3).
Decoder gradients scale with c and encoder gradients with W_c, so a learned
encoder starts at a saddle, and the injected posterior noise only hurts until
c is informative. The decoder learns to mute it. The declared retry
(`--kl-weight 0.001 --latent-dim 8`) is collapsing the same way (spread 0.054 m
at epoch 3); its final result follows.

**M5a channel v2 (declared before running, 2026-10-26): hindsight latent with
a fixed PCA encoder.** c_t = the standardized top-4 principal components of
the human's next 6 waypoints (PCA fitted on the training futures). Training
uses c_t deterministically (no noise, no KL), so the decoder learns to steer
VaVAM from an informative latent (decoder gradient ~600× larger in a check).
The prior q_phi(c | z_t) is fitted by maximum likelihood to those detached
latents (`--latent-objective hindsight --latent-encoder pca --latent-dim 4
--prior-weight 0.1`, frame memory, 8 epochs, seed 0). Sampling the prior then
spreads plans over the futures that are plausible given z_t. Same gate (spread
≥ 0.5 m, posterior flow < prior flow). If the CVAE retry passes after all, it
stays the declared choice for RL and v2 is reported alongside. If v2 fails,
latent RL stops here and the report records the negative result.

**M5a v2 stopped at epoch 2 (design bug found); v2b declared (2026-10-27).**
v2 did not learn (held-out flow 0.0852 at epoch 2; frame was 0.042), and
the latent residual had no effect on the flow loss. Cause: VaVAM's action
tokens have norm ~4.2e3. In a direct probe, the 3 s waypoint moves ~0.2 m per
100 units of residual norm (≈ linear: 0.02 m at 10, 2 m at 1000). The latent
decoder starts at norm ~6, and Adam's step size grows it only ~0.05 per
weight per epoch, far from a useful magnitude. The same limit explains the
CVAE collapse. **v2b** = v2 with the decoder output scaled by 100 and
zero-initialised (exact parity at the start); `--latent-decoder-scale 100`,
otherwise identical (hindsight, PCA encoder, latent dim 4, prior weight 0.1,
frame memory, 8 epochs, seed 0), same gate. In a 40-step check the spread
grew 0 → 0.10 m and the posterior beat the prior (0.0415 vs 0.0418).

**M5a CVAE retry result: fails the gate as well.** `runs/sft/m5a_latent_retry_s0`
(`--kl-weight 0.001 --latent-dim 8`), best epoch 8: spread 0.056 m, posterior
flow 0.0426 vs prior 0.0428, KL 0.17. Both declared CVAE configurations are
negative results with the same cause (decoder scale vs. token norm, above).

**M3 confirmation on eval_big-48 (declared before running, 2026-10-27):**
the val_cl null (frame adapter vs B0b\*) has CIs of ±0.13–0.2. Rerun on the 48
eval_big scenes whose artifacts are on disk (selected by availability, before
any run; the other 48 would need ~77 GB). Same M1 protocol (2 rollouts, CATK,
4 s replay, route commands, 8 frames, `VAVAM_MUP_SHAPES_DIR`), B0b\* (stock
VaVAM) vs B1-frame (`runs/ais/m3_cl/frame_s0.bundle.pt`). Primary: paired
score. Secondary: progress, offroad, at-fault collision. test_cl stays untouched.

**M4 budget check result:** B3 (GRU) at 16 epochs reaches held-out flow
0.0556 (best epoch 16, still improving ~0.002/epoch), vs 0.069 at 8 epochs and
0.037 for frame (best epoch 3). Twice the budget does not close the gap: the GRU
optimises far more slowly but is not shown to be worse in the limit. B4 at 16
epochs follows (declared rule: B3 moved materially).

**eval_big-48 result (`reports/m3_eval_big48.md`): the frame adapter is worse
than B0b\* in closed loop.** Paired score −0.088 [−0.161, −0.023] over 43
matched scenes (−0.086 [−0.156, −0.023] when its 4 off-map aborts count as
failures); pass rate −0.10; at-fault collision +0.07. The open-loop gain from
the executed-motion input does not transfer; on more scenes it hurts. Four
eval_big scenes abort for every policy (renderer exception on dynamic nodes,
broken routes).

**M5a v3: RL-only latent channel on stock VaVAM-B (declared before running,
2026-10-27).** Independent of v2b: can closed-loop RL through a latent residual
improve stock VaVAM-B? Bundle (`drivers/vavam_ais.py --rl-latent 4 300` on
`abl_frame_s0`): its frame encoder supplies z_t as features only, and the
adapter gate is set to 0, so the mean policy is stock VaVAM-B (= B0b\*). The
latent decoder is a fixed random orthonormal map × 300, and the prior starts
at N(0, I). Sizing probe (`scripts/latent_gate.py`, 64 held-out windows):
scale 150 / 300 / 600 → spread 0.47 / 0.93 / 1.69 m at 3 s; held-out ADE
3.68 / 3.74 / 3.86 m. Scale 300 passes the 0.5 m gate.
- RL as declared for M5a: 8 iterations × 16 train scenes × 4 rollouts, CATK,
  AlpaSim score, scene-group baseline, PPO (lr 3e-5, target KL 0.02, KL 0.1 to
  v0), GPUs 0/1.
- Evaluation on val_cl (M1 protocol): primary = paired score of the v8
  **mean** policy (`VAVAM_AIS_LATENT=mean`) vs B0b\*. Secondary: v8 sampled vs
  v0 sampled (the RL effect at equal exploration noise), and v0 sampled vs
  B0b\* (the cost of exploration noise).

**M5a v2b result: fails the gate on spread.** `runs/sft/m5a_v2b_pca_s0`, best
epoch 8: spread 0.138 m at 3 s (lateral 0.038, longitudinal 0.130; gate ≥ 0.5),
posterior flow 0.0646 < prior 0.0693 (passes), open-loop ADE 2.85 (posterior)
vs 2.99 m (prior). With the residual scale fixed, the hindsight channel carries
real information about the future. Its maximum-likelihood prior, though, is
confident (held-out NLL rising), so prior samples barely explore, and almost
not laterally, where VaVAM fails. As declared, RL on a learned latent stops here.
The separately declared v3 (fixed random channel on stock VaVAM) is the RL arm.

Summary of the latent-channel attempts (all frame memory, 8 epochs):

| channel | spread at 3 s | posterior vs prior flow | verdict |
|---|---|---|---|
| CVAE, KL 0.01, d=4 | 0.056 m | 0.0466 vs 0.0468 | fail (collapse) |
| CVAE, KL 0.001, d=8 | 0.056 m | 0.0426 vs 0.0428 | fail (collapse) |
| hindsight PCA, decoder ×1 | stopped at epoch 2 (no learning) | — | design bug |
| hindsight PCA, decoder ×100 | 0.138 m | 0.0646 vs 0.0693 | fail (prior too narrow) |
| fixed random ×300 (v3, RL only) | 0.93 m (probe) | n/a | passes; RL running |

**M5a v3 training diagnosis (before its evaluation):** the policy barely moved
(KL to v0 0.010–0.053; v8 prior-head weights ~2e-4). Offline check over all 8
iterations (292 rollouts with group signal): no latent dimension correlates
with the advantage (max |r| = 0.093, permutation p = 0.39). With a fresh
latent every 0.5 s, the exploration is temporal dithering: plans are re-drawn
before a perturbation can change behaviour consistently, so outcome differences
are noise from the policy's point of view.

**M5a v3b: temporally persistent exploration (declared before running,
2026-10-27).** Same as v3 (bundle `runs/rl/v3_bundle_v0.pt`, 8 iterations × 16
train scenes × 4 rollouts, the same PPO settings and scene seeds), but
`VAVAM_AIS_LATENT_HOLD=4`. A latent is drawn every 4 queries (2 s) and held,
an option policy with log-probs only at decision points (credit: decisions whose
hold window overlaps control before the first incident). Evaluation as for v3:
primary = v8 mean policy vs B0b\* (paired score, val_cl). Secondary: v8 held
sampling vs v0 held sampling, plus the same offline signal check.

**M5a v3 result (`reports/m5a_v3_rl.md`): no RL effect.** Primary: v8 mean
policy vs B0b\* +0.014 [−0.108, +0.13] (it behaves like stock VaVAM; the policy
barely moved). v8 vs v0, both sampled: −0.059 [−0.153, +0.015]. Exploration
noise costs progress (v0 sampled vs B0b\*: −0.12 [−0.22, −0.03]). The RL
machinery works end to end, but 512 rollouts of per-step dithering carry no
learnable signal. v3b (held latents) is running.

**M4 budget check, B4 at 16 epochs:** held-out flow 0.060 (best epoch 16) vs
B3-16 0.056 and frame 0.038 (seed 0 each). Twice the budget keeps the
ordering frame ≈ stack < GRU ≤ B4. Neither memory nor the AIS objective helps
the action head on this data and budget.

**M5a v3b result (`reports/m5a_v3b_rl.md`): no improvement over stock VaVAM.**
Primary: v8 mean policy vs B0b\* −0.025 [−0.209, +0.141]. Held exploration
is costly (v0 held vs B0b\* −0.21 [−0.41, −0.04], offroad 66%). RL recovered
part of that (v8 vs v0 held +0.08 [−0.04, +0.23]) by exploring less, not by
finding a better direction (no directional correlation: max |r| = 0.10,
p = 0.24; size of perturbation vs advantage r = −0.11, p = 0.057). In this
channel, stock VaVAM is a local optimum at this budget.

**M5a equal-budget non-RL comparator (declared before running, 2026-10-27).**
Advantage-weighted regression (`training/latent_awr.py`), labelled
reward-weighted imitation, not RL. Data: the same 512 rollouts as v3b (all 8
iterations, decision points, per-iteration scene-group advantages). One offline
weighted maximum-likelihood fit of q_phi(c | z) from the v0 prior, weights
exp(A) clipped at 20, 300 full-batch Adam steps at lr 1e-3. Evaluation: mean
policy on val_cl (M1 protocol), paired vs B0b\* and vs v3b v8 mean.

**Credit-window correction (latent_rl):** AlpaSim scores a rollout from control
start for `duration_frac_20s · 20 s`, which ends at the first incident or
earlier where the scene's metrics stop being valid (e.g. 12.5 s while the video
runs 16 s). `latent_rl` had ended credit at the first raw offroad/collision in
`metrics.parquet`, which can fall after the scored window. It now uses the
scored window. Rechecked offline: v3 no signal (max |r| 0.089, p 0.44); v3b
no direction (max |r| 0.125, p 0.11), but perturbation size hurts (r = −0.17,
p = 0.004). The v3/v3b conclusions stand.

**M5a AWR comparator result (reward-weighted imitation, not RL):** mean
policy vs B0b\* −0.093 [−0.278, +0.071], progress −0.07 [−0.145, −0.007]. RL
v3b v8 (mean) vs AWR: +0.068 [−0.000, +0.15]. On the same 512 rollouts,
on-policy RL is at least as good as the offline weighted fit, which moved the
latent means (|mu| 0.13) and drove slightly worse. Neither improves on stock
VaVAM-B. M5a is complete as specified (valid latent-policy RL, restored
checkpoints, closed-loop evidence, equal-budget non-RL comparison), with a
negative result.

## M5b — AlpaGym feasibility (2026-10-27)

- Host prerequisites without sudo: `redis-server` 8.10 (conda-forge,
  `~/ext/tools-env`), plus `cudnn.h`/`nccl.h` from the AlpaGym venv's NVIDIA wheels
  via `CPATH`. Only the pre-flight script's system-path check still fails.
- Upstream smoke run (`alpamayo_1_5_local_2gpu_smoke`): one unsharded policy
  replica (`dp_shard_size: 1`). **Correction (same day):** the recipe's Cosmos
  wrapper (`alpamayo1_x_rl/models/expert_model/cosmos_wrapper.py`) freezes the
  language model and vision tower, so only the action expert trains. Memory is
  ~20 GB of frozen bf16 weights plus the expert and its optimizer, which may fit
  in 48 GB; bf16 is emulated (slow). AlpaGym's AlpaSim launch would need our
  gsplat renderer instead of NRE.
- VaVAM integration (the M5b deliverable) fits easily (319M + 38M parameters);
  started below.

## M4 — memory-dependent scenario tests (guideline §6; declared before running, 2026-10-27)

All cases come from actual AlpaSim rollouts (held-out expert-replay windows,
held-out CATK steps) or supported simulator controls (driver memory reset).
- **T1 history length (open loop):** B3 (GRU, 16 epochs) and B4 (frozen AIS
  GRU, 16 epochs) on the 312 held-out windows, with the memory re-run over only
  the last k ∈ {1, 2, 4, 8, 16, all} queries (observation timestamps
  unchanged). History matters if held-out flow loss falls as k grows.
- **T2 memory-relevant slices (open loop):** held-out windows classified by
  the ego's own speed history over the last 2 s: stop-and-go (stopped at some
  point, now > 3 m/s), decelerating (speed fell > 2 m/s), accelerating (rose >
  2 m/s), steady. motion-only, frame, stack, GRU-16 and B4-16 (seed 0) are
  compared on flow loss and ADE within each slice, with scene-cluster bootstrap
  CIs on the GRU − frame difference.
- **T3 history in closed loop:** the B3 GRU-16 adapter bundle in AlpaSim
  (val_cl, M1 protocol) with its memory reset every k queries:
  `VAVAM_AIS_MEMORY_RESET` = 1 (current step only), 4 (2 s), and never.
  Primary: paired score never − reset-1.
- **T4 predictive memory on informative steps:** the M2 AIS heads (frame vs
  GRU, 3 seeds each) on held-out CATK steps whose speed changed > 3 m/s over the
  previous 2 s: collision/offroad AUROC and next-embedding NLL on that subset,
  with the event count reported (power).
