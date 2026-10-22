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

## M2 — SFT (in progress)

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

## Tests

- `pytest` in the AlpaSim venv: planner, plugin, labels, SFT (21 passed,
  renderer test skipped there).
- `pytest tests/test_gsplat_render.py` in `metadrive-render` with
  `METADRIVE_TEST_USDZ` set: renders vs. photo, resolution scaling.

## Next actions

1. Collect expert data on the 48 added training scenes; retrain SFT and
   re-check against always-FOLLOW and the state-only ablation.
2. If images still add nothing, try a larger encoder (DINOv2-base) and denser
   decisions before blaming the representation.
3. Closed-loop val evaluation with CATK: rule vs. SFT (`scripts/evaluate.sh`).
4. M3: preference pairs from per-candidate ADE and closed-loop outcomes; DPO.
