# Progress log

Living record of what is done, measured, blocked, and next. Plan and milestones:
[PROJECT_GUIDELINES.md](../PROJECT_GUIDELINES.md).

## Environment (pinned 2026-10-21)

| Item | Value |
|---|---|
| GPUs | 4× Quadro RTX 8000, 48 GB, compute capability 7.5 (Turing) |
| NVIDIA driver | 580.178.04 (CUDA driver API 13.0) |
| Host RAM / disk | 251 GB / ~310 GB free on `/` |
| AlpaSim | `NVlabs/alpasim@affc2eab` + `patches/alpasim/*.patch` |
| AlpaSim venv | Python 3.12, torch 2.8.0+cu128 (`~/ext/alpasim/.venv`): driver plugin, analysis |
| Renderer env | conda `metadrive-render`: Python 3.10, torch 2.4.1+cu124, gsplat 1.5.3 (sm_75 kernels) |
| Scene data | HF `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (gated, access granted) |

Setup: `scripts/setup_alpasim.sh` (clone, pin, patch, build, install plugin).

Artifacts: checkpoints (SFT seed 0, GRPO main it8, AIS encoders), dataset v2 and result tables
are in the Hugging Face repo `unionpan/metadrive-rl` (public card, gated files with automatic approval; round-trip load verified);
run directories, videos and logs stay under `runs/` and `demos/` on the training host.

GPU allocation while simulating: GPU 0 AlpaSim physics (Docker), GPU 1 gsplat
renderer, GPUs 2–3 free for training.

## M0 — feasibility gate

**NVIDIA NRE cannot run here.** Scene load fails with `CUDA error: no kernel
image is available for execution on the device`: NRE's own CUDA/Slang
extensions are compiled for sm_80+ only (the container's PyTorch does include
sm_75, so the error is a sticky failure from an extension kernel). The image is
closed-source. No newer GPU is available, so rendering is reimplemented.

**gsplat renderer (`src/metadrive_rl/render/`).** NuRec scenes store 3D
Gaussians in their checkpoint. We load them without unpickling code
(`checkpoint.py`), and rasterise them with gsplat's 3DGUT path using the
scene's exact f-theta camera models and rolling shutter. Vehicles are
reconstructed in cuboid-local frames; cuboid *i* maps to the track whose time
span equals `time_embed.timestamps_us_ranges[i]` (all 96 matched, pose counts
sum to the 7527 calibration rows). `server.py` serves AlpaSim's sensorsim
gRPC API (version, scenes, cameras, ego masks, `render_rgb`, `batch_render_rgb`).

Validation against the photo shipped in scene `clipgt-01d503d4…` (front wide
camera, recorded pose, recorded actor poses; bottom 10% hood band excluded):

| Check | Result |
|---|---|
| Full frame PSNR, 1920×1080 | 24.70 dB |
| PSNR on vehicle pixels (6.6% of frame), without → with vehicles | 16.03 → 20.78 dB |
| 512×320 render vs. downsampled 1920×1080 render | > 24 dB (test threshold) |
| Over gRPC at 512×320 vs. downsampled photo | 21.38 dB |
| Latency over gRPC, 512×320, 36 actors, JPEG | 46.6 ms (was 153 ms before vectorising) |

**Generalisation across the 24 split scenes** (`python -m metadrive_rl.render.validate`,
`reports/render_validation_24_scenes.json`, `reports/render_validation_montage.jpg`):
full-frame PSNR median 18.0 dB (min 11.8, max 25.7); placing actors improves
vehicle pixels by a median 2.2 dB (22 scenes with visible actors). Geometry and
actor placement hold everywhere; the worst scenes are a night drive (our render
is much brighter) and a sunny drive (flat sky misses the sun and gradients).

Known gaps: (1) the sky is a flat colour: `model.background.textures` holds one
panorama per camera whose parameterisation we could not identify; no mapping
tried beat a flat colour. (2) No camera response / exposure model. A per-scene
affine colour fit on the other cameras' photos made the held-out front camera
*worse* (median −3.2 dB, worse in 18/24 scenes), consistent with per-camera
responses (`crf_params` is per camera); with one photo per camera there is no
honest held-out check for a same-camera fit, so no calibration is applied.
(3) No ego-hood overlay. (4) NRE's per-track pose corrections are not applied;
right-multiplying them lowered PSNR, so their convention is unknown. (5) Only
f-theta cameras. (6) Views far from the recorded path degrade (smears, floaters)
— e.g. the rule policy 60–90 m off the logged position — so rendered
observations are only trustworthy near the log.

**Off-coverage degradation: cause and handling.** The reconstruction only
covers what the recording drive saw. Along the CRUISE@10 rollout (human drove
74 m, policy 163 m), static Gaussians in a 40 m view cone ahead drop from
98,571 at the start to 45,958 at the end of the logged path, 5,611 at 120 m and
593 at 150 m; the remaining large semi-transparent Gaussians smear. AlpaSim
already stops scoring a rollout once the ego is ≥ 4 m from the logged
trajectory (`max_dist_to_gt_trajectory`); GRPO batches now also drop decisions
after that exit. Lane changes (3.5 m) stay inside; the binding limit is
longitudinal: at 10 m/s the ego passes the end of this 74 m log at ~11 s, so
faster-than-human policies are scored on that window only.

Renderer-side pruning, validated on 24 scenes (full-frame PSNR vs. photos):
dropping Gaussians larger than 5 or 10 m is worse in 12/24 scenes (up to −1 dB;
many are real background) and its off-coverage effect cannot be measured, so it
is not used. Dropping near-transparent Gaussians (opacity < 0.01) removes 41%
of static Gaussians, renders 12% faster (40.0 → 35.2 ms at 512×320) with no PSNR
change (worse in 0/24) and is the server default. Beyond the corridor only a
generative model could fill in content (NVIDIA uses a diffusion "fixer"); that
would be plausible, not recorded, imagery.

**No-render mode.** Before the renderer existed, closed-loop runs used
AlpaSim's renderer skip mode, patched to report locally defined cameras and
return black frames (`metadrive_no_render`). Those runs have black camera
images (`img_is_black = 1`) and are valid only for perception-free policies.

## M1 — mode-conditioned driver

**Done**
- 12-action schema and deterministic planner (route following, committed lane
  offset, jerk/accel limits, look-ahead curvature speed cap).
- Driver plugin returns all 12 candidates and logs a JSON decision record
  (action, log-prob, policy version, decision id) into each ASL drive response.
- ASL reader, per-rollout metric reduction, and the M1 probe analysis
  (`python -m metadrive_rl.evaluation.mode_probe`).

**Exit gate result (no-render mode, rule + fixed actions, one scene, 20 s):**
see `reports/m1_mode_probe_table.md` and `reports/m1_mode_probe_paths.png`.
Distance travelled: BRAKE 4.8 m, YIELD 22.7 m, FOLLOW 31.7 m, CRUISE@10 164 m;
lane changes end ±3.3 m from the CRUISE path. Same scene + different mode →
different executed paths: **met**. The rule policy reproduces CRUISE@10 with
0.00 m divergence (deterministic replay).

**Bugs found by closed-loop runs and fixed**
- Session state reset every call: the servicer rebuilds its pose list per
  update; sessions are now keyed by the first pose object.
- Lane changes approached their target asymptotically (>100 m) because each
  replanning call restarted the lateral blend; the blend now resumes from the
  distance driven since the lane target changed. Closed loop after the fix:
  both lane changes reach 90% of a lane width within 24 m.
- AlpaSim's `MAP` route generator re-anchors the route to the ego's current
  lane mid-episode, which made a held left lane change swing 7 m to the right
  at ~140 m. The plugin now detects such lateral route jumps (> 1.5 m between
  calls, only while the ego projects onto the given waypoints) and moves the
  committed offset with the route. Closed loop: max divergence 3.56 m (one lane).

**Known limitation: lane offsets are relative to a route the runtime
regenerates.** After the fix, the held left lane change still drifts back over
~138–165 m while its offset from the route stays at 3.50 m, i.e. the `MAP`
route itself moves relative to the original lane. The `RECORDED` route is no
alternative: it ends where the logged drive ended (74 m in this scene), and an
ego faster than the human runs off its end (the left lane change looped off-road).
A robust fix needs lane-graph information, which the deployable driver
interface (route waypoints only) does not provide.

**Finding: collision metrics under log-replay traffic.** Every probe run
reports front and rear collisions, including BRAKE (rear-ended at 4.1 s by the
logged follower). Without a traffic model, recorded actors ignore the ego, so
any deviation from the human log causes contact.

**Reactive traffic (CATK).** `trafficsim=catk` (weights via Git LFS) runs on
GPU 0 with physics. With CATK all 62 compared actors move differently from log
replay and the rule policy is no longer rear-ended (`collision_rear` 0); its
at-fault front collision with the truck it drives into remains. Direct test of the
replay failure (same scene, gsplat cameras):

| Probe | Log replay | CATK |
|---|---|---|
| BRAKE (stops after 4.9 m) | front + rear collision | no collision; score 0.083 (progress 0.066) |
| YIELD (23 m) | front + rear collision | no collision; score 0.394 (progress 0.315) |
| rule ≈ CRUISE@10 (drives into the truck) | collisions | at-fault collision, score 0 |

Evaluation therefore uses CATK and AlpaSim's aggregated scoring: a rollout
fails on an at-fault collision or offroad, otherwise scores its clipped
relative progress, so stopping is safe but scores low
(`python -m metadrive_rl.evaluation.summarize`). Remaining limits: CATK only
takes over agents within 100 m at handover (1.7 s); the rest are held static.
The 343 m pose difference seen between replay and CATK runs is such frozen
agents on the opposite carriageway.

## M2 — SFT (exit gate met; no closed-loop gain)

- Frozen splits: `configs/evaluation/splits.yaml`, public_2604 seed 0,
  16 train / 4 val / 4 test (the development scene is in train). 24 scenes, 40 GB.
- Expert collection (`scripts/collect_expert.sh`): the ego replays the human log
  for the whole episode while the driver is called every step on gsplat frames;
  images saved per decision (0.5 s). Labels: candidate with the lowest ADE to the
  logged rig future over the 4 s plan (`metadrive_rl.data.labels`).
- **Bug found by the label audit:** during long force-GT replays the runtime
  stops sending dynamic state and the stock servicer reports speed 0 (its
  docstring promises a finite-difference fallback the code lacks). Candidates
  were planned from standstill and the median best-candidate ADE was 8.5 m with
  52% lane-change labels. The plugin now falls back to pose-history finite
  differences (logged as `speed_source`); median ADE on the first scene 0.46 m,
  labels FOLLOW / CRUISE@5 for this slow-traffic drive. Data re-collected.
- Encoder: DINOv2-small FP16, CLS + mean patch tokens (768-d), 88 ms per batch
  of 32 frames on one RTX 8000.
- Head: MLP on [image features, state features]; checkpoint selection by
  validation regret (ADE of the chosen action − best ADE).

**SFT round 1 — negative result (16 train scenes, 512 / 128 decisions, 3 seeds):**

| Model | Val regret (m) | Val top-1 |
|---|---|---|
| always FOLLOW (best constant) | 0.801 | 0.328 |
| soft-label CE, image + state | 1.580 ± 0.053 | 0.159 |
| soft-label CE, state only | 1.546 ± 0.046 | 0.182 |
| ADE regression, image + state | 0.801 ± 0.000 | 0.328 |
| ADE regression, state only | 0.801 ± 0.000 | 0.328 |

Cross-entropy heads overfit (train regret 0.16 m) and pick expensive actions on
validation scenes (whose speeds are also lower: mean 3.8 vs 8.1 m/s). ADE
regression, which targets the selection metric, never does worse than the
constant baseline, but its best checkpoint *is* the constant baseline: 16
scenes contain no generalisable signal, visual or not. Response: training split
extended to 64 scenes (val/test unchanged, verified) and ADE regression made
the default objective.

**Data scale-up.** 48 training scenes added (64 total); 62 collected — two
scenes fail AlpaSim's own route sanity checks on every attempt and are
excluded. Collecting all pending scenes in one AlpaSim run (4 concurrent
rollouts, no videos) took 18 min for 46 scenes vs ~3 min per scene before
(~7.7× throughput), after fixing renderer cache thrashing (2-scene GPU cache
vs 4 concurrent scenes: 600 reloads in 10 min; default cache now 6).
Dataset v2: 1984 train / 128 val decisions, FOLLOW 49% of labels.

**Bug: NaN route waypoints.** AlpaSim routes can end in NaN waypoints; the
route features turned them into NaN inputs for 24 rows, which silently voided a
full SFT sweep (every configuration collapsed to the same 1.93 m regret).
Features now drop non-finite waypoints and training data is checked for
non-finite values (planner was already robust).

**SFT round 2 — 62 train scenes, 3 seeds** (`reports/m2_sft_regret.md`,
`reports/m2_sft_regret.png`):

| Model | Val regret (m) |
|---|---|
| always FOLLOW | 0.801 |
| ADE regression, state only | 0.846 ± 0.079 |
| ADE regression, DINOv2-small (any regularisation/context) | 0.798–0.800 |
| ADE regression, DINOv2-base | 0.731 ± 0.022 |
| ADE regression, DINOv2-base, 3-decision context | **0.680 ± 0.023** |
| soft-label CE, DINOv2-small | 1.100 ± 0.064 |

Images only help with the larger encoder, and a 3-decision frame stack helps
further (−15% regret vs. the best constant action), an early data point for
H1 (history helps). Caveat: models were selected and compared on the same 4
validation scenes; claims go to the test split. The closed-loop checkpoint is
fixed in advance as seed 0 (val regret 0.682), not the best seed.

**M2 exit gate: closed loop through the driver API.** SFT seed 0 (fixed in
advance) runs as `policy: {type: neural}` in the AlpaSim plugin with DINOv2-base
and its 3-decision feature stack. Closed loop on the 4 val scenes (CATK,
2 rollouts each, `reports/m2_val_closed_loop.md`): rule 0.25, always-FOLLOW
0.064, SFT 0.088 mean score with scene-level CIs spanning ~[0, 0.75]; on the 16
val_cl scenes (M4 section) SFT is indistinguishable from the baselines. The
offline regret gain does not translate into measurable closed-loop gains.

## M3 — preference optimisation (in progress)

**DPO on ADE-derived preferences — negative result.** Pairs (5902): preferred =
lowest-ADE candidate, rejected = candidates ≥ 1 m worse; reference and init =
SFT seed 0 (val regret 0.682). Three DPO seeds each:

| Setting | Val regret (m) | KL to ref | Train pair acc. |
|---|---|---|---|
| β 0.1, lr 1e-4, 100 epochs | 1.77–1.85 (rises from epoch 1) | 0.70 | 0.99 |
| β 1, lr 1e-5, 60 epochs | 0.752 ± 0.019 | 0.22 | 0.97 |
| β 5, lr 1e-5, 60 epochs | 0.791 ± 0.061 | 0.12 | 0.97 |

These preferences carry the same information as SFT's ADE-regression targets,
so DPO can only overfit them. Preference optimisation needs a *new* signal:
next step is simulator-verified preferences from matched-state branches
(force-GT to a branch time, then alternative actions under CATK, ranked by
AlpaSim's at-fault/offroad and progress outcomes). These are simulator-derived
preferences, not RLHF.

**Branch preferences** (`scripts/collect_branches.sh`,
`metadrive_rl.data.branch_prefs`). For each scene, the expert log runs under
force-GT to 4 s. Then each of 7 actions (BRAKE, YIELD, FOLLOW, CRUISE@5/10,
CHANGE_LANE_LEFT/RIGHT@10) runs from that same state for 6 s under CATK.
Ranking: a passing rollout beats a failing one; between two passing rollouts,
more progress wins. Train: 62 scenes, 969 pairs (426 safety, 543 progress).
val_cl: 16 scenes, 221 pairs (114 / 107), with no simulator failures.

**Branch DPO** (β 0.1, lr 1e-4, init and reference SFT seed 0; the epoch is
chosen on val_cl pair accuracy, 3 seeds):

| | val_cl pair acc. | train pair acc. | KL to SFT | expert val regret (m) |
|---|---|---|---|---|
| SFT | 0.624 | 0.680 | 0 | 0.680 |
| branch DPO (epoch 20, all seeds) | 0.733 | 0.81 | 1.2–1.5 | 0.51–0.57 |

Unlike ADE-derived pairs, these carry signal SFT lacks: held-out pair
accuracy rises by 11 points for every seed, and expert-imitation regret also
falls. The policy changes substantially. On expert val states, greedy
CRUISE@10 goes from 5% (SFT) to ~50%, FOLLOW stays at 47–58%, and BRAKE,
YIELD and CRUISE@15 mostly disappear. Caveat: epoch selection uses val_cl
pairs, so a val_cl closed-loop score for DPO is mildly optimistic. A clean
comparison needs eval_big or test_cl.

Closed loop, val_cl, 2 rollouts, seed 0, on the 11 scenes every policy completed:

| run | score | pass rate | at-fault collision | progress | distance (m) |
|---|---|---|---|---|---|
| SFT | 0.504 [0.274, 0.734] | 0.591 | 0.409 | 0.668 | 81 |
| GRPO-it8 | 0.549 [0.277, 0.818] | 0.636 | 0.364 | 0.673 | 82 |
| branch DPO | 0.541 [0.268, 0.814] | 0.545 | 0.455 | 0.758 | 98 |

Paired DPO − SFT: score +0.037 [−0.106, +0.176], at-fault collision
+0.045 [−0.136, +0.273], progress +0.090 [+0.001, +0.209]. Outcomes differ
in only 3–4 scenes. DPO drives further. Without a lead-gap input, the
faster CRUISE@10 choices do not reduce rear-end collisions, so the
preferences buy progress, not safety. No score claim at this sample size.

## M4 — distributed RLVR (in progress)

**Topology benchmark** (identical workload: rule policy, 4 val scenes × 2
rollouts, CATK, gsplat cameras). `metadrive_4gpu`: physics + CATK on GPU 0,
renderers on GPUs 1 and 3, driver on GPU 2, second CATK replica on GPU 3,
8 concurrent rollouts (vs. 4 on one GPU per service).

| Topology | Videos | Simulation | Steps/s | Post-processing | Wall |
|---|---|---|---|---|---|
| 1gpu | on | 563 s | 2.83 | 102 s | 749 s |
| metadrive_4gpu | on | 310 s | 5.14 | 330 s | 721 s |
| 1gpu | off | 504 s | 3.16 | 60 s | 653 s |
| metadrive_4gpu | off | 317 s | 5.02 | 212 s | 602 s |

Simulation throughput improves 1.6–1.8×; end-to-end only 1.04–1.08× on this
small batch because all rollouts finish together and their evaluation runs
after the last step (with 4 slots it overlapped with the next batch). Larger
batches (GRPO: 64 rollouts) amortise this.

**GPU utilisation and concurrent stacks.** During single-stack branch
collection the GPUs averaged ~25% (10/29/0/60%) with the CPU mostly idle: the
pipeline is latency-bound (serial gRPC steps per rollout), not compute-bound.
Stacks can now run side by side (per-stack driver port, wizard base port and
Docker Compose project; evaluation renderers on their own ports). Measured with
branch collection and the 96-scene evaluation running concurrently: 7.2 + 6.2 =
**13.4 simulation steps/s vs ~5.0 for one stack (2.7×)**, still with >20 GB free
per GPU and ~10 of 128 cores busy.

**GRPO smoke run** (`scripts/grpo_loop.sh`, `runs/grpo/smoke/iterations.jsonl`):
SFT seed 0 as init and reference; 16 fixed training scenes (first 16 of the
train split), groups of 4 rollouts per scene, 12 s episodes, temperature 1.0,
`metadrive_4gpu`, reward = AlpaSim score + 0.2 × progress.

| Iteration | Rollouts | Decisions | Mean reward (train, on-policy) | Collect | Update | Rollouts/h | KL to ref | Entropy | Clip frac. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 1536 | 0.560 | 1036 s | 11 s | 222 | 0.070 | 0.80 | 0.40 |
| 2 | 64 | 1536 | 0.603 | 975 s | 11 s | 236 | 0.087 | 0.75 | 0.32 |

The loop ran collection → immutable batch → update twice without
intervention, with the sampled actions' logged log-probabilities and correct
policy versions (iteration 2's batch was sampled by `grpo:smoke:it1`). One
group had zero reward variance in iteration 2. The reward change is on-policy
training data and within noise; not a result. Changes after the smoke run:
floor the group reward std at 0.05 (small groups were amplified), 2 update
epochs instead of 4 (clip fraction 0.3–0.4), optional greedy evaluation of
every checkpoint on a held-out split.

**Closed-loop comparison on val_cl** (16 scenes, 2 rollouts each, CATK,
greedy policies, `metadrive_4gpu`; scene-level bootstrap CIs).
Some rollouts fail inside CATK ("no usable map geometry was found within
100 m of the current ego position"). Which scenes fail depends on the policy:
1 scene for the rule policy, the same 4 other scenes for FOLLOW, SFT and GRPO
(slow/holding policies), on both CATK replicas, so this is a policy-dependent
simulator failure, not a missing map. Both views are reported.

All completed rollouts (`reports/valcl_closed_loop_all.md`):

| run | scenes | rollouts | score | pass rate | at-fault collision | offroad | progress | distance (m) |
|---|---|---|---|---|---|---|---|---|
| rule | 15 | 30 | 0.386 [0.151, 0.625] | 0.467 [0.2, 0.733] | 0.467 [0.2, 0.733] | 0.0667 [0, 0.2] | 0.614 [0.426, 0.8] | 105 [72, 141] |
| always-FOLLOW | 12 | 24 | 0.491 [0.253, 0.709] | 0.667 [0.417, 0.875] | 0.333 [0.125, 0.583] | 0 [0, 0] | 0.641 [0.431, 0.817] | 80.7 [51.2, 112] |
| SFT | 12 | 24 | 0.504 [0.254, 0.717] | 0.583 [0.333, 0.833] | 0.417 [0.167, 0.667] | 0 [0, 0] | 0.685 [0.461, 0.876] | 87.9 [54.8, 122] |
| GRPO-smoke-it2 | 12 | 24 | 0.462 [0.211, 0.709] | 0.542 [0.25, 0.792] | 0.458 [0.208, 0.75] | 0 [0, 0] | 0.667 [0.453, 0.857] | 84.8 [54, 117] |

Matched scenes, completed for every policy (`reports/valcl_closed_loop_matched.md`):

| run | scenes | rollouts | score | pass rate | at-fault collision | offroad | progress | distance (m) |
|---|---|---|---|---|---|---|---|---|
| rule | 11 | 22 | 0.449 [0.177, 0.721] | 0.455 [0.182, 0.727] | 0.545 [0.273, 0.818] | 0 [0, 0] | 0.735 [0.504, 0.921] | 94.2 [56.2, 131] |
| always-FOLLOW | 11 | 22 | 0.49 [0.26, 0.728] | 0.682 [0.409, 0.909] | 0.318 [0.0909, 0.591] | 0 [0, 0] | 0.638 [0.41, 0.837] | 76.5 [47.4, 110] |
| SFT | 11 | 22 | 0.504 [0.274, 0.734] | 0.591 [0.318, 0.864] | 0.409 [0.136, 0.682] | 0 [0, 0] | 0.668 [0.437, 0.879] | 81 [49.9, 115] |
| GRPO-smoke-it2 | 11 | 22 | 0.504 [0.274, 0.734] | 0.591 [0.318, 0.864] | 0.409 [0.136, 0.682] | 0 [0, 0] | 0.667 [0.436, 0.878] | 81.1 [50, 116] |

No policy is distinguishable. The SFT and GRPO-smoke checkpoints give
identical metrics: two small updates on 16 scenes did not change any greedy
choice. Always-FOLLOW has the lowest at-fault collision rate; the rule policy
the highest.

**GRPO main run** (`reports/m4_grpo_main.md`, `reports/m4_grpo_main.png`): 8
iterations from SFT seed 0, rotating through four disjoint training subsets
(62 scenes), groups of 4, 12 s episodes, `metadrive_4gpu`. 472 rollouts and
11,328 sampled decisions, ~2.3 h of collection (180–220 rollouts/h, 10 s
updates), 24 simulator-aborted rollouts excluded. KL to the SFT reference rose
0.08 → 0.16; on-policy reward on the same subset four iterations apart
changed by +0.02, +0.10, +0.01, −0.06 (noise-level).

Greedy evaluation on val_cl, matched scenes (completed for every policy):

| run | scenes | rollouts | score | pass rate | at-fault collision | offroad | progress | distance (m) |
|---|---|---|---|---|---|---|---|---|
| rule | 11 | 22 | 0.449 [0.177, 0.721] | 0.455 [0.182, 0.727] | 0.545 [0.273, 0.818] | 0 [0, 0] | 0.735 [0.504, 0.921] | 94.2 [56.2, 131] |
| always-FOLLOW | 11 | 22 | 0.49 [0.26, 0.728] | 0.682 [0.409, 0.909] | 0.318 [0.0909, 0.591] | 0 [0, 0] | 0.638 [0.41, 0.837] | 76.5 [47.4, 110] |
| SFT | 11 | 22 | 0.504 [0.274, 0.734] | 0.591 [0.318, 0.864] | 0.409 [0.136, 0.682] | 0 [0, 0] | 0.668 [0.437, 0.879] | 81 [49.9, 115] |
| GRPO-it8 | 11 | 22 | 0.549 [0.277, 0.818] | 0.636 [0.364, 0.909] | 0.364 [0.0909, 0.636] | 0 [0, 0] | 0.673 [0.44, 0.891] | 81.9 [50.7, 117] |

Paired GRPO-it8 − SFT over the same 11 scenes: +0.045 [+0.000, +0.136]
(bootstrap over scenes) — driven by a single scene that improved; 10 scenes
scored identically. Not a claim.

**Statistical power of the closed-loop evaluation.** From paired per-scene
score differences on val_cl: policies that behave alike (SFT vs always-FOLLOW:
paired SD 0.043) are resolved by a few scenes, but behaviourally different
policies (SFT vs rule: paired SD 0.264) need ~27 scenes to resolve ±0.10 and
~107 for ±0.05 (95% CI half-width), before CATK's run-to-run traffic noise.
The 16-scene val_cl/test_cl splits can only detect large effects; a final
comparison needs ~100 held-out scenes with repeated rollouts.

**Demo / noise illustration** (`demos/d5cc3763_rule_sft_grpo.mp4`, not in Git;
`reports/demo_rule_sft_grpo.gif`): the one val_cl scene where GRPO-it8 beat
SFT (both GRPO rollouts passed, SFT 1 of 2, rule 0 of 2) was re-run with one
rollout per policy for video: now rule passes and SFT and GRPO collide. Single
rollouts under CATK's stochastic traffic flip outcomes; the video illustrates
the system, not a ranking. Late frames degrade once the ego is 70–85 m from the
logged position (renderer limit noted under M0).

**Simulator-aborted rollouts are excluded, not scored 0.** When aggregation
is allowed despite failures, CATK-aborted rollouts appear in
results-summary.json with score 0 and an error instead of metrics. The first
main GRPO run crashed on them (missing `progress`); counting them as reward 0
would have penalised the policy for a simulator error. GRPO batches and
summaries now skip them and log the count (iteration 1: 4 of 64, one scene).

**Evaluation noise: CATK samples.** Reactive traffic draws each agent's next
motion token from the top-5 at temperature 1.0, so the same deterministic
policy can get different outcomes in different runs (rule policy: at-fault
collision rate 0.50 in one run, 0.75 in another, on the same 4 scenes).
Comparisons must average repeated runs; single-run differences on few scenes
are noise.

## M5 — AIS study (offline, first pass)

Matched-budget comparison of history encoders on expert decision sequences
(`python -m metadrive_rl.training.ais_offline`, `reports/m5_ais_offline.md`):
same DINOv2-base features, cost objective, optimiser, epochs, and splits; the
AIS variant adds a transition head predicting the next observation embedding
(fixed 64-d PCA target) from (z_t, a_t).

| encoder (DINOv2-base features, 3 seeds) | params | val regret (m) | val transition MSE |
|---|---|---|---|
| frame (memoryless) | 202656 | 0.695 ± 0.072 | — |
| stack-3 | 251936 | 0.660 ± 0.160 | — |
| GRU, task loss | 301728 | 0.759 ± 0.057 | — |
| AIS-GRU, λ=1 | 328032 | 0.796 ± 0.008 | 0.871 |
| AIS-GRU, λ=0.3 | 328032 | 0.750 ± 0.045 | 0.731 |
| AIS-GRU, λ=0.1 | 328032 | 0.772 ± 0.025 | 0.841 |

**Conditioned on history ambiguity** (`reports/m5_ais_ambiguity.md`): a
decision is ambiguous when the expert labels of its 20 most similar training
frames (current image embedding only) disagree (≥ 0.6).

Ambiguous = current-frame kNN (k=20) expert-label disagreement >= 0.6; 48% of val decisions.

| encoder (3 seeds) | val regret all (m) | ambiguous frames | unambiguous frames |
|---|---|---|---|
| frame | 0.695 ± 0.072 | 0.548 ± 0.063 | 0.834 ± 0.127 |
| stack | 0.660 ± 0.160 | 0.481 ± 0.122 | 0.829 ± 0.196 |
| gru | 0.759 ± 0.057 | 0.510 ± 0.077 | 0.992 ± 0.071 |
| ais (λ=0.3) | 0.750 ± 0.045 | 0.603 ± 0.020 | 0.887 ± 0.091 |

On ambiguous frames the memory models are slightly better than the memoryless
one (stack-3 −0.07 m, GRU −0.04 m vs. frame), the direction H1 predicts, but
within one seed SD; AIS is worst there. H1: weak, non-significant hint;
H2: not supported.

No encoder separates from the memoryless baseline (seed spread up to ±0.16 m on
4 val scenes); AIS at λ=1 collapses to the constant policy (the transition loss
dominates), at λ=0.3 it matches the plain GRU. H1/H2 are **not supported in
this setting**. Expert-replay data is mostly smooth following with little
history-dependent ambiguity; the meaningful test needs history-sensitive
scenarios (aliasing, occlusion, changing traffic behaviour) evaluated in
closed loop.

## Lead-vehicle perception and speed governor (in progress)

**Failure mode.** On val_cl, 38 of 40 at-fault collisions across rule,
always-FOLLOW, SFT and GRPO-it8 are `collision_front` (2 lateral, rule only):
the ego drives into the car ahead. The cause is structural: no input carries
the distance to a lead vehicle, and FOLLOW holds the current speed. Neither
imitation nor GRPO can learn to keep a gap they cannot observe. The demo GIF
shows the same thing. The actor directly ahead at contact is always a car,
but the 38 front collisions split into two kinds:

- **Moving cars, 22 collisions in 5 scenes.** The rear-end failure. In 8
  of these (2 per policy) the car has no Gaussians, so it is invisible to
  the camera and no camera-based policy can avoid it.
- **Parked cars, 16 collisions in 2 scenes** (`048b974e`, `ad55f513`).
  Every policy, both rollouts, collides identically, early (progress 0.16
  in `048b974e`), about 1 m from the human's path. The planner tracks
  the route's lane centre; in these narrow streets the human drove offset
  from it, past parked cars whose boxes reach into the lane. This is a
  lateral-path problem: no choice of high-level action or speed cap avoids
  it. These scenes are a constant floor shared by all policies.

**Labels.** `metadrive_rl.data.lead_gap`: bumper-to-bumper gap to the nearest
actor ahead within a ±1.6 m corridor, from logged actor poses and the scene's
cuboid sizes (60 m = no lead). These labels are privileged; they train a
perception head and never reach the policy.

**Camera gap head** (`metadrive_rl.perception.lead_gap`): MLP on frozen
features + ego speed. Trained on expert_v2 decision frames (2 Hz, 4 val scenes,
3 seeds):

| features | val MAE (m) | MAE, lead < 30 m | recall, lead < 10 m | dangerous miss (lead < 10 m, predicted > 20 m) |
|---|---|---|---|---|
| DINOv2-base pooled, 3 frames | 12.9 ± 0.2 | 21.2 ± 1.4 | 0.49 | 0.50 |
| DINOv2-base 3×4 grid, 1 frame | 12.8 ± 0.3 | 11.2 ± 1.0 | 0.66 | 0.25 |
| DINOv2-base 3×4 grid, 3 frames | 19.9 ± 4.7 | 39 ± 15 | 0.25 | 0.75 (unstable) |

Pooled features lose where things are in the image; keeping a coarse
spatial grid halves the near-lead error. A 25% dangerous-miss rate is still
too high to brake on, and the expert data contains few close-lead frames.

**Dense perception data** (`scripts/collect_dense.sh`,
`metadrive_rl.data.build_gap_dataset`): one camera frame per 0.1 s step
(`image_every_step`) over the train+val scenes, from (a) expert force-GT
replays (done: 66 scenes, 13,134 frames; 2 scenes fail AlpaSim's route sanity
check) and (b) SFT driving in closed loop under CATK (collecting). Closed-loop
driving reaches the close-lead and near-collision states the governor must
recognise.

Denser frames alone barely helped. On the 4 unseen val scenes the dense-expert
heads score about like the 2 Hz heads (dangerous miss ~24% for both; recall
within 10 m 0.72 vs 0.68): consecutive frames of a scene are near-duplicates.

**Most of the remaining "misses" were label errors.** Every dangerous miss on
the val scenes came from one scene where the ego waits at a crossing and the
"lead" 2–6 m ahead is a pedestrian. NuRec renders no pedestrians (0 of 604
tracks in 12 scenes), riders (0/35) or strollers, nor moving vehicle tracks
without dynamic Gaussians (140 of 1,388 vehicle tracks). Parked vehicles are
part of the static background and do appear. The labels now count only
vehicles the renderer draws (`visible_tracks`): 2.5% of train labels and 18%
of val labels change.

| gap head (DINOv2-base grid, 1 frame; visible-vehicle labels) | data | recall < 10 m | dangerous miss | false alarm |
|---|---|---|---|---|
| 4 unseen val scenes, 2 Hz expert heads (3 seeds) | expert_v2 | 0.88–0.93 | 0 | 0 |
| 4 unseen val scenes, dense expert heads (3 seeds) | gap_dense_expert | 0.93–0.96 | 0 | 0 |
| 12 held-out scenes (8 train held out + 4 val) | gap_dense_expert | 0.892 ± 0.007 | 0.065 ± 0.001 | 0.001 |

(false alarm: predicted < 10 m when the true gap is ≥ 30 m.)

**Corridor along the driven path.** A straight corridor sweeps over cars
parked along a bend. Labels now follow the path the ego actually drives next
(`future_path`/`path_gap`; 1,393 train labels change by > 0.5 m, mostly at
long range; close-lead counts barely move). Heads trained on path labels fit
them better (held-in MAE 0.4 vs 3.2 m), but on held-out frames they are not
better: scored on path labels, straight-trained heads reach recall 0.97–0.98
with no dangerous misses, path-trained heads 0.88 with 11%. All of the
path-trained misses come from one val scene (36 frames; predictions ~42 m vs
8 m). One scene decides this, so the comparison is inconclusive. The governor
keeps the straight-label head chosen in advance; path labels are used for
evaluation.

The governor's main risk is now phantom engagement, not missed leads. On the
12 held-out scenes, frame by frame: the true gap would engage the governor on
5.3% of frames; the predicted gap engages it on 12%. Phantoms make up 7–8% of
frames, with a mean speed cap 7 m/s below the current speed; misses are 0.6%.
The head places a lead at 20–45 m when none exists. With
`governor_range_m = 25` (estimates beyond 25 m count as no lead), phantoms
fall to 3–5% and close-lead misses stay at 0.6–0.7%.

**Speed governor** (`planner/trajectory.py`, `PolicySpec.gap_head`): given a
perceived gap, every mode except BRAKE is capped at
`min(sqrt(2 b g), g / T)` on the free gap `g` beyond a 2 m standstill
distance (b = 3 m/s², T = 1.5 s). Within the horizon the lead is assumed
stationary; the plan is recomputed every 0.1 s with a fresh camera estimate.
The planner uses whichever acceleration is lower, the mode's or the
governor's. Plans without a gap are unchanged. Unit tests: from 5–20 m/s it stops behind a
stationary lead within 2–3 m of the 4 m/s² physical limit, and it settles at
s0 + T·v behind a moving lead. The decision record logs the estimate
(`lead_gap_est_m`), so closed-loop perception error can be measured from the
logs.

This is a hand-designed prior, so the evaluation must separate it from the
learned policy: each policy with and without the governor, plus
always-FOLLOW + governor, on matched scenes.

**Ablation:** always-FOLLOW, SFT and GRPO-it8, each with the governor
(gap head `gap_dense_expert_vis_grid_s0`, seed 0 chosen in advance; range
25 m), on val_cl with 2 rollouts, on a single-GPU stack (GPU 1), compared on
matched scenes with the existing no-governor runs. A grid encoder's output
begins with the pooled features, so the policies get exactly their training
features from the same forward pass.

*always-FOLLOW + governor* (12 matched scenes): score +0.000 [−0.125,
+0.125], at-fault collision +0.000, progress −0.036 [−0.092, 0.000]. Only
2 scenes change, one each way (CATK-level noise). **No detectable effect.**

Why: the camera head degrades in closed loop. Measured against path labels
over all 4,776 driver calls, recall within 10 m is 0.56 and the
dangerous-miss rate is 36% (offline on expert frames: ~0.9 and ~6%). By scene:

- Rear-end scene `17ab62e4` (the lead is a rendered car): the true gap closes
  16 → 0 m over 4 s while the estimate trails it by 10–20 m (36 → 13 m). The
  governor engages at 5.2 s and brakes from 9.4 to 5.9 m/s, too late. The
  head overestimates the gap at close range, which is the unsafe direction.
- Waiting behind a car at 1.8 m (`20a9fb48`): estimates ~5 m, recall 0.99.
- Parked-car scenes (`048b974e`, `ad55f513`): the ego's path runs into
  parked cars at the side; the head (trained on expert driving that never
  does this) does not report them. A speed governor would not fix these anyway.

**Perception v2.** Changes are judged on *closed-loop* frames: the dense SFT
collection (61 train+val scenes, 14k frames, CATK). Validation is the
closed-loop frames of the 4 val scenes plus 8 held-out train scenes (1,725
frames). Frames after a rollout's first contact are dropped: the rollout has
already failed, and the camera may sit inside the other car (89% "misses"
there, mean bias +45 m), which is not a state a governor acts in.

| gap head (3 seeds; closed-loop val frames, pre-contact) | recall < 10 m | dangerous miss | false alarm | bias, lead < 15 m |
|---|---|---|---|---|
| v1: MLP on 3×4 grid, expert frames (used in the first ablation) | 0.86 | 0.11 | 0.015 | +5.6 m |
| MLP on 3×4 grid, expert + closed-loop frames | 0.82 | 0.11 | 0.012 | +5.2 m |
| conv on 3×4 grid, expert + closed-loop | 0.86 | 0.070 | 0.012 | +3.7 m |
| **conv on 6×8 grid, expert + closed-loop** | **0.87** | **0.062** | 0.014 | **+2.7 m** |
| + min over the last 5 estimates (0.5 s) | 0.89–0.90 | 0.049–0.057 | 0.011–0.040 | +1.7–2.3 m |

Missed close cars were plain in the images (a car filling the lower frame at
3–7 m read as 20–25 m). A coarse 3×4 average pool blurs the strongest
monocular distance cue, the image row of the car's bottom edge; a 6×8 grid
with a small conv net keeps it. Closed-loop training frames alone did not
help. A low-quantile (pinball) head did not reduce the bias either, so it is
not used. Some misses are renders that break down far from the recorded path
and cannot be fixed by perception.

Governor v2 = conv 6×8 head (`arch_conv6x8_s0`, seed 0), min filter over 5
calls, range 25 m.

**Closed-loop results** (val_cl, 2 rollouts, 12 matched scenes, scene-bootstrap CIs):

| run | score | pass rate | at-fault collision | progress |
|---|---|---|---|---|
| always-FOLLOW | 0.491 [0.253, 0.709] | 0.667 | 0.333 | 0.641 |
| always-FOLLOW + governor v1 | 0.491 [0.259, 0.699] | 0.667 | 0.333 | 0.605 |
| **always-FOLLOW + governor v2** | **0.603 [0.353, 0.824]** | **0.792** | **0.208** | 0.625 |
| SFT | 0.504 [0.254, 0.717] | 0.583 | 0.417 | 0.685 |
| SFT + governor v1 | 0.504 [0.250, 0.754] | 0.583 | 0.417 | 0.646 |

Paired against always-FOLLOW:
- **v1:** score +0.000, progress −0.036. The 2 scenes that change go in
  opposite directions.
- **v2:** score +0.111 [−0.027, +0.292], at-fault collision −0.125 [−0.292,
  0.000], progress −0.016 [−0.061, +0.014]. All 3 changed scenes are
  moving-car rear-end scenes (see the collision split above): `451cbe44`
  2/2 collisions → 2/2 passes, `d5cc3763` 1 collision fixed, `17ab62e4` one
  fixed and one new.

Closed-loop perception error before contact, from the driver's own logs:
- v1: recall 0.78, dangerous miss 15%, bias +8.8 m.
- v2: recall 0.83, dangerous miss 9%, bias +5.9 m, false alarm 0.2%.

The effect is where the mechanism predicts and in the predicted direction,
but the CI still touches zero on 12 scenes. The parked-car scenes are
untouched, as expected. A claim needs eval_big. The GRPO-it8 + v1 run was
stopped after v1 showed exactly zero effect for both always-FOLLOW and SFT.
Running: SFT + v2 (GPU 1 stack) and GRPO-it8 + v2 (GPU 2 stack).

## Tests

- `pytest` in the AlpaSim venv: planner (incl. governor), plugin, labels,
  lead-gap labels, SFT/DPO/GRPO (42 passed, renderer test skipped there).
- `pytest tests/test_gsplat_render.py` in `metadrive-render` with
  `METADRIVE_TEST_USDZ` set: renders vs. photo, resolution scaling.

## Next actions

1. Lead perception: finish dense collection (expert, then SFT closed loop),
   build `gap_dense`, extract grid features, retrain the gap head with 8
   held-out train scenes; target a dangerous-miss rate well below 25%.
   DINOv3 grid features once access to the gated weights is granted.
2. Governor ablation on matched val_cl scenes: {always-FOLLOW, SFT, GRPO-it8}
   × {no governor, camera-gap governor}, 2 rollouts each; read the logged
   estimate against the true gap for closed-loop perception error.
3. Powered SFT vs GRPO-it8 comparison on `eval_big` (96 scenes, running via
   `scripts/eval_powered.sh`), summarised with `--common-scenes`.
4. GRPO continuation from it8 at LR 5e-5 (`runs/grpo/launch_cont.sh`, starts
   when branch collection frees its renderers).
5. M3: branch DPO once val_cl branches finish (`branch_prefs` → `dpo --pairs`).
6. Learnable headway (FOLLOW@T actions) only if the fixed governor helps.
