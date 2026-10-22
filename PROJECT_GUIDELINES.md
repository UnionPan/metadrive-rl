# MetaDrive-AIS: Fine-Tuning VaVAM-B with Decision-Relevant Visual Memory

**Project guidelines · v1.2 · 22 October 2026 (`larx` verification revision)**  
**Primary machine:** `larx` — 4 × NVIDIA Quadro RTX 8000 (48 GB each, Turing / compute capability 7.5)  
**Core stack:** NVIDIA AlpaSim · Valeo VaVAM-B · optional DINOv3 · PyTorch · AIS · planned continuous-action RL / AlpaGym integration

> **One-line objective:** Build and evaluate an AIS-conditioned VaVAM-B end-to-end driving agent in real AlpaSim closed loop; then extend it with a valid continuous-action simulator-reward training loop, optionally integrated into AlpaGym/Cosmos-RL, without assuming native VaVAM GRPO support.

> **v1.2 changes (checked against the local checkouts and hardware on 22 Oct 2026):**
> 1. VaVAM-B was trained with an **8-frame context** (`finetuning_timesteps=8`, 2 Hz ≈ 3.5 s); only the AlpaSim wizard preset sets `context_length=1`. The memory baseline AIS must beat is therefore native 8-frame VaVAM-B, not the 1-frame preset (§1, §6).
> 2. Stock AlpaSim rendering (NRE) has **no Turing kernels**; on `larx` AlpaSim renders through the in-repo gsplat sensorsim, which today supports only f-theta cameras. VaVAM's preset uses a synthetic **OpenCV-pinhole** camera, so pinhole support is an M1 prerequisite (§5 Stage A, §8.1, §9).
> 3. Measured `larx` facts, evaluation-statistics lessons from the archived v0 project, and the verified status of Alpamayo 1/1.5/2, AlpaGym and alpamayo-recipes are recorded in §7 and §8.1.

## 0. Design contract — read before implementing

1. **Preserve end-to-end learned trajectory generation.** VaVAM-B predicts continuous ego trajectories through its own learned flow-matching action expert. Do **not** introduce an MPC planner, CBF controller, handcrafted route optimizer, or hand-coded driving modes as the architecture of this project.
2. **Use the actual pretrained VaVAM-B checkpoint.** Keep its existing video-feature pathway and trajectory-output format; treat AlpaSim's already-existing low-level tracking controller as part of the simulator, not as a claimed research contribution.
3. **Learn memory, not a replacement perception system.** DINOv3 is an optional second frozen visual feature extractor. A recurrent AIS summarizes longer observation–action history and conditions the existing action generator.
4. **Separate the simulator from the RL framework.** VaVAM-B is supported as an **AlpaSim driver**, but NVIDIA's public **AlpaGym policy/training path currently targets Alpamayo 1.5**. VaVAM-to-AlpaGym integration is a new policy/inference/trainer implementation, not a config switch. Standard text-token GRPO does **not** directly implement RL over VaVAM's continuous flow-generated trajectories.
5. **Use real evidence.** No milestone is complete on the basis of mock rollouts, fabricated metric files, passing shape tests alone, or a dashboard with illustrative graphics. Save genuine closed-loop replay videos, logs, metrics, checkpoint hashes, and commands.
6. **Optimize for four older GPUs.** Do not make retraining the 318M video backbone, original large-scale video pretraining, Alpamayo-10B fine-tuning, or Alpamayo-10B evaluation a prerequisite. Profile first; prefer FP16-compatible kernels and frozen backbones. (Frozen Alpamayo 1 / 1.5 / 2 inference in AlpaSim *is* verified on `larx`, §8.1, and may be run as an optional external reference baseline; it is never part of the AIS claim.)
7. **No fake RL support.** A pretrained flow-matching trajectory generator is not automatically a stochastic policy with a tractable trajectory log density. Implement a valid sampled-action distribution and estimator before claiming PPO/GRPO/RLVR.
8. **Avoid data leakage.** At policy decision time, only use camera observations, allowed ego state / route information, and past executed actions. Simulator ground truth may be used for targets, evaluation, and diagnostics **only**, unless explicitly running a labeled privileged-oracle baseline.

---

### Workstream split (hard boundary)

| Track | Environment | Model training | Dependency |
|---|---|---|---|
| **A — Core driving/AIS** | Native **AlpaSim** (`driver=vavam` or custom gRPC driver) | Frozen-backbone AIS/flow SFT | Required; independent of AlpaGym |
| **B — Simulator-reward RL** | AlpaSim episodes + reward exporter | Valid stochastic latent-policy REINFORCE/PPO (or separately labeled AWR) | After Track A drives |
| **C — Distributed RL platform** | **AlpaGym + Cosmos-RL** with a new VaVAM model/policy bundle | New VaVAM-specific trainer and data pipeline | After Track B is correct; significant adaptation |

This separation prevents the most costly architectural confusion: **AlpaSim's native VaVAM driver is not evidence that VaVAM supports AlpaGym's stock GRPO implementation.**

---

## 1. Why VaVAM-B?

The original **VaViM / VaVAM** research was released by Valeo.ai. VaViM is a video-generative backbone; VaVAM attaches an action expert that generates driving trajectories conditioned on video features and high-level navigation commands. **VaVAM-B = 318M-parameter video backbone + 38M-parameter action expert (~356M total).** The original action-learning phase keeps the video model frozen, giving a close precedent for lightweight downstream adaptation. Its trajectory expert uses **flow matching**, producing a continuous trajectory from noise through iterative denoising; this is not an autoregressive LLM.

AlpaSim supports the VaVAM driver as its tutorial default, downloads `vavam-b` assets, and allows either a different mounted model directory or a custom driver container image. In the documented default VaVAM configuration, the driver uses the **latest camera frame** (`context_length=1`).

**Verified in the local checkout (v1.2):** VaVAM-B (`VAM_width_1024_*`) was trained with an **8-frame context** (`vam` `finetuning_timesteps=8`), and AlpaSim's `VAMModel` defaults to 8 frames when `context_length` is unset; only the wizard's `driver/vavam.yaml` sets 1. The stock preset also runs the policy and camera at **2 Hz** (`control_timestep_us=500000`), with 3 s of forced log replay and 100 steps (50 s) per episode, and renders a synthetic **1920×1080 OpenCV-pinhole** front camera with radial/tangential distortion (`extra_cameras`), not the scene's recorded f-theta camera. Images enter VaViM as LlamaGen VQ tokens (16384 codes, 16× downsampling). So VaVAM already has a ~3.5 s causal visual memory when run as trained. The research opportunity is **longer horizons, executed-action history, and decision-relevant compression beyond that window**, measured against native 8-frame VaVAM-B (B0b, §6).

### Feasibility boundaries

| Work item | Position | Rationale / gate |
|---|---|---|
| Pretrained VaVAM-B closed-loop inference | **Supported upstream; on `larx` needs pinhole support in the gsplat renderer** | NRE cannot run on Turing (§8.1). Run an actual AlpaSim episode and export MP4 before changing architecture. |
| Freeze video backbone; supervised adaptation of 38M action expert | **Strongly feasible in principle** | Action-learning recipe exists; fit and throughput still require profiling on `larx`. |
| Frozen DINOv3-S/B visual embeddings + GRU | **Strongly feasible** | Train and cache embeddings offline; no full video foundation-model training. |
| AIS-to-action-expert conditioning | **Moderate integration work** | Must modify model forward / training / driver inference and show that AIS affects actions. |
| Joint AIS + action expert fine-tuning | **Plausible** | Backpropagation and checkpointing require validation; keep VaViM frozen first. |
| Native AlpaSim reward collection | **Supported in simulator; adapter work for training** | Record outcomes and metrics from real closed-loop VaVAM episodes. |
| AlpaGym/Cosmos-RL with VaVAM-B | **Custom engineering, not upstream support** | Must integrate model-specific inference, payloads, checkpointing and training; validate end to end. |
| Online simulator-reward policy improvement | **Research/engineering extension** | Need a real sampled policy and correct RL estimator; not plug-and-play GRPO. |
| LLM-style reasoning SFT, RLHF, chain-of-thought GRPO | **Out of scope for this model** | VaVAM-B is not a language-generating VLA. Optional separate VLM project later. |

**Not the aim:** reproduce the original multi-node, many-GPU VaViM pretraining or large-scale Valeo action-learning recipe. Instead, start from released weights and adapt a small number of trainable parameters to AlpaSim and/or properly licensed trajectory demonstrations.

---

## 2. End-to-end architecture

```text
AlpaSim camera stream, permitted ego-motion history, route command
     │
     ├── [Original VaVAM-B vision pathway] ──> VaViM context features ──┐
     │                                                                   │
     └── [DINOv3-S/B, frozen] ──> visual embeddings e_t                  │
                                      │                                  │
                       past executed ego actions + timestamps           │
                                      │                                  │
                                recurrent AIS z_t                       │
                                      │                                  │
                            learned projection / gated adapter          │
                                      │                                  │
                                      └─────────> VaVAM-B action expert <┘
                                                (flow matching)
                                                        │
                                               continuous ego trajectory
                                                        │
                                        AlpaSim native low-level tracker
                                                        │
                                     physics / reactive traffic / next image
                                                        └── closed-loop feedback
```

The original VaVAM features **and** AIS-conditioned features reach the *same existing action generator*. The AIS model never independently outputs a hand-coded driving plan.

### Precise variables and update

At decision step `t`, denote the observable camera frame(s) by `o_t`, the executed ego-motion history by `u_{<t}`, and the navigation command by `n_t`.

```text
e_t = frozen_DINOv3(o_t)
z_t = AIS_GRU(z_{t-1}, e_t, u_{t-1}, Δt, optional ego-state_t)
h_t = original_VaVAM_context(o_t, n_t)
δ_t = gate(z_t) * projection(z_t)
trajectory_t = VaVAM_action_expert(noise, time, h_t, δ_t, n_t)
```

**Integration candidates, in order of complexity:**

1. **Action-feature residual (first choice).** Project `z_t` into the action expert's internal token dimension; inject as a zero-initialized gated residual in its action-token conditioning. Maintain all original attention masks and trajectory decoding. This is a proposed code patch, **not** an upstream configuration feature.
2. **Learned memory tokens (later).** Let the action expert attend to projected AIS token(s) alongside the existing visual context, with careful attention-mask and position semantics.
3. **Broader VaViM representation modifications (not first version).** Only attempt if ablations show the earlier additions are inadequate.

**Compatibility invariant:** With AIS gating set to zero and identical input/noise seeds, the modified policy must reproduce the original VaVAM-B outputs within measured numerical tolerance. This is a non-negotiable regression test.

**Controllability test:** With the current observation and sampling noise fixed, changing a plausible prior observation history should alter the recurrent state and, after training, affect the trajectory. Demonstrate that the policy does not silently ignore memory.

---

## 3. What makes it an approximate information state?

**A frozen DINOv3 embedding is not by itself an AIS.** Nor does a generic recurrent encoder automatically satisfy AIS theory. The learned summary should approximately preserve information about expected rewards and the distribution of future observations conditional on an action.

Use a lightweight recurrent compressor:

\[
  z_{t+1} = f_\eta(z_t, a_t^{\mathrm{executed}}, e_{t+1}, \Delta t),\qquad
  e_t=E_{\mathrm{DINOv3}}(o_t).
\]

Here the update uses **past executed/observed motion**, not a future trajectory target. A candidate reference trajectory emitted earlier is also past information, but should be distinguished from what was physically executed.

### Proposed learning objectives

- **Reward sufficiency:** predict the next reward or reward-component distribution from `(z_t, a_t)`; use squared error for appropriate continuous terms and a classification loss for collision/off-road indicators.
- **Predictive sufficiency:** predict a distribution for the *next* DINOv3 embedding (or a suitable projected embedding) conditioned on `(z_t, a_t)`; start with a Gaussian NLL or feature-space predictive objective that avoids collapse. A pointwise cosine loss alone is only a heuristic.
- **Behavioral supervision:** retain the released VaVAM flow-matching loss for expert future trajectories. Combine it with auxiliary AIS losses through documented loss weights.

A schematic objective is:

\[
\mathcal L = \mathcal L_{\mathrm{flow}} + \lambda_r\mathcal L_{\mathrm{reward}} + \lambda_o\mathcal L_{\mathrm{next\;observation}}.
\]

Start by pretraining the AIS and its auxiliary heads on logged episodes while holding VaVAM frozen. Then fine-tune the adapter plus the original action expert if needed. Measure whether the auxiliary prediction objectives correlate with better decisions, rather than asserting formal AIS guarantees from finite training loss.

### Optional latent mode switching / meta-RL

An experimental branch may add a **learned** small latent context `c_t` or mixture-of-recurrent-states module inferred from historical interactions. Such modes may capture changing traffic dynamics; they are **not** hand-labeled cruise/yield/merge commands. Compare against a parameter-matched plain GRU and check whether learned modes actually influence trajectories.

---

## 4. Data contract and dataset strategy

### Data sources

1. **Pretrained VaVAM-B weights:** Valeo's released checkpoint; verify license terms before use, publication, or redistribution.
2. **AlpaSim closed-loop logs:** sampled RGB camera observations, timestamp, allowed ego state, issued reference trajectory, physically executed egomotion, outcome metrics, episode boundary, scenario ID, and exact driver checkpoint.
3. **Expert demonstrations for SFT:** synchronized camera/trajectory examples from datasets covered by suitable licenses (e.g., nuPlan / nuScenes where available) or approved logged teacher demonstrations. Merely saving VaVAM's own trajectories and retraining on them does not create new expert supervision.
4. **DINOv3 embeddings:** cache frozen visual embeddings with encoder version, preprocessing, frame timestamps, and source video hashes. Recompute on demand for live rollout inference.

### Canonical episode record (schematic—not upstream schema)

```yaml
episode_id: "scene_id/run_id/rollout_id"
checkpoint_id: "vavam-b+ais:<hash>"
timestamps_us: [...]                    # chronological, stable
camera_frames: [...]                     # paths or immutable references
navigation_commands: [...]               # allowed policy inputs
observed_ego_poses: [...]                # policy-visible, if configured
issued_reference_trajectories: [...]     # actions actually chosen
executed_ego_displacements: [...]        # observed subsequent motion
rewards: [...]                            # may be calculated post-rollout
reward_components:
  progress: [...]
  collision_any: [...]
  offroad: [...]
terminal_reason: "..."
model_features_cache: "..."              # optional and versioned
```

**Never leak the post-action reward or future ego trajectory into the input of an earlier policy decision.** Preserve sampling frequencies, coordinate frames, image calibration, route commands, and trajectory horizon when building training examples. The DINO and VaVAM image preprocessing pipelines are distinct and must be validated separately.

Split by scene / route / traffic configuration, not by adjacent frames. Record train/validation/test episode IDs. Repeated rollouts from the same underlying scene must remain in the same split or be evaluated explicitly as in-distribution repeats.

---

## 5. Training strategy: staged, hardware-conscious

### Stage A — Frozen VaVAM-B baseline

- Run the *stock* AlpaSim `vavam-b` driver in a real scene; save `.asl` logs, metrics and MP4 replay.
- Record model latency, model VRAM, simulator/renderer VRAM, successful steps, and rollout throughput.
- Confirm trajectory format, action horizon, time alignment, camera intrinsics, navigation input and policy frequency.
- Use original code as the benchmark oracle for numerical parity when AIS is disabled.

**NVIDIA's tutorial command does not work on `larx` as written.** It renders with NRE, which has no Turing kernels:

```bash
bash data/download_vavam_assets.sh --model VaVAM-B      # assets: works on larx
uv run alpasim_wizard deploy=local topology=1gpu driver=vavam \
  wizard.log_dir="$PWD/outputs/baseline-vavam-b"        # NRE renderer: fails on Turing
```

**`larx` procedure** (same pattern as the verified Alpamayo runs, §8.1):

1. Add OpenCV-pinhole cameras with radial/tangential distortion to the gsplat sensorsim (`src/metadrive_ais/render/`; gsplat 1.5.3 already supports `camera_model="pinhole"` with `radial_coeffs`/`tangential_coeffs`). Validate them: geometric consistency with the f-theta render of the same view, and photometric statistics.
2. Add a host driver config `configs/driver/vavam.yaml` (local asset paths) and an external-renderer preset carrying the VaVAM camera and 2 Hz timing from `vavam_configs.yaml`.
3. Run `scripts/run_alpasim.sh` with the gsplat renderer, as for Alpamayo.
4. Record two baselines: **B0a** (stock preset, `context_length=1`) and **B0b** (native 8-frame context).

Run this only after checking current workloads, Docker/NVIDIA support, filesystem space and required credentials on `larx`.

### Stage B — DINOv3 representation cache and AIS encoder

- Choose DINOv3-S first; compare DINOv3-B only if feature quality justifies cost.
- Freeze DINOv3, record video preprocessing and cache embeddings from **past and current** frames.
- Train a 128–512-dimensional GRU information state as an initial experiment. Treat dimensions as sweep candidates, not confirmed best settings.
- Add reward prediction and action-conditioned next-embedding prediction heads.
- Compare loss curves and held-out prediction scores against frame stacking and generic GRU baselines.

### Stage C — AIS-to-VaVAM adapter

- Create one trainable zero-initialized gated residual connected to the action expert.
- Load the official VaVAM-B weights; freeze VaViM and initially the 38M action expert.
- Verify exact baseline parity at a zero gate. Verify nonzero gradient flow into the adapter and AIS after enabling its gate.
- Train adapter with expert trajectory supervision using the same flow-matching objective as the action head, plus optional AIS auxiliary losses.
- Profile memory with backward passes through the **frozen** expert (frozen weights still do not guarantee free activation memory).

### Stage D — Supervised action fine-tuning

- If adapter-only training underfits, unfreeze the **38M action expert**, but retain frozen VaViM and DINOv3 first.
- Keep VaVAM's continuous trajectory target and flow-matching denoising loss; do not replace it with arbitrary discrete maneuver labels.
- Use small batches, gradient accumulation, activation checkpointing as needed, mixed precision on Turing, gradient clipping and numerical checks.
- Run open-loop imitation metrics and closed-loop AlpaSim evaluations after each candidate checkpoint.

### Stage E — Simulator-reward policy improvement (after supervised AIS/VaVAM works)

**Do not start by running the stock AlpaGym GRPO command with `driver=vavam`.** AlpaGym's current public policy/trainer path is built around Alpamayo 1.5. AlpaSim supports VaVAM independently. The two facts must not be conflated.

A continuous flow-matching action expert samples continuous trajectories, but that does **not** automatically expose the valid `log π_θ(trajectory | history)` needed by standard PPO/GRPO. Pick and test an actual trainable policy distribution before wiring in a distributed trainer.

**Recommended first genuine RL algorithm — stochastic AIS-conditioned latent control:**

```text
allowed observations, past actions
           │
      AIS state z_t
           │
  trainable latent policy q_φ(c_t | z_t)  ← log_prob(c_t) available
           │                   └── sample and record c_t
           ▼
VaVAM-B flow action expert conditioned on c_t + VaViM features
           │
    generated reference trajectory
           ▼
    native AlpaSim closed-loop rollout and simulator rewards
           │
  REINFORCE / PPO or grouped advantage update of q_φ
```

- Initially **freeze VaViM and the flow expert** while learning `q_φ` and, where justified, the AIS/conditioning adapter. A sampled `c_t` may be Gaussian or categorical; this is a *learned latent action*, not a hand-coded driving mode.
- The score-function gradient applies to the sampled **latent** `c_t`, not the final continuous trajectory. Optimize expected **executed** rollout reward under the composite policy, and retain the exact sampled latent actions and log probabilities. If a parameter-independent random seed is used by the frozen flow expert, record it for reproducibility; do not claim its log density is being optimized.
- Measure whether varying `c_t` while holding the rest of the conditioning fixed generates meaningful trajectory variation. Without an effective action channel, RL cannot improve driving behavior.
- For online updates, store `episode_id`, `policy_version`, timestamped observations or reproducible policy inputs, AIS initial/reset state, sampled latent actions, old log probabilities, rewards, terminations and trajectory outputs. Recompute or validate current policy log probabilities with correct handling of recurrent memory and rollout-policy lag.
- Start with REINFORCE + baseline or clipped PPO and compare against supervised fine-tuning and reward-weighted imitation. Group-normalized rollout advantages are an **optional estimator design**, not evidence of having implemented AlpaGym's model-specific GRPO.
- Use reward components such as progress, collisions and off-road events from AlpaSim; validate reward polarity, scenario grouping, time aggregation, degenerate all-equal groups and credit assignment. Track behavior on held-out scenes.

**Lower-risk alternative — reward-weighted behavioral cloning / AWR:** use simulator scores to weight actual sampled demonstrations, with clearly documented assumptions. This can establish an early improvement signal, but **must not be labeled on-policy RL or GRPO**.

**Research-heavy alternative:** policy gradients for stochastic denoising trajectories or an explicitly density-tractable flow policy. This is not required for the first working RL loop.

### Stage F — AlpaGym / Cosmos-RL interoperability (after a correct standalone RL update)

**Current upstream status (checked 22 October 2026):** AlpaGym is a framework for closed-loop simulator rollouts and distributed RL training, but its public reference policy is Alpamayo 1.5. Its `Policy` / `InferenceEngine` / model-bundle pathway, episode artifact transport and existing GRPO trainer are *not* a plug-and-play VaVAM-B trainer. AlpaSim's own `driver=vavam` gRPC integration **is** already supported.

**Engineering tasks and interfaces to inspect in the pinned upstream code:**

1. **Rollout policy**: implement or extend an AlpaGym `Policy` owning per-session AIS memory and producing AlpaSim ego-frame `PolicyOutput` trajectories. The policy resets memory per episode and obeys navigation, camera and action coordinate contracts.
2. **Model bundle and inference**: implement a VaVAM-specific `InferenceModel` / policy-bundle registration (or equivalent for pinned code), and a batching-capable inference path for VaViM, the AIS adapter, the flow action expert and sampled latent policy. The published `sample_trajectories_from_data(...)` path is Alpamayo-specific; do not assume VaVAM implements it.
3. **Trajectory and learning data**: extend rollout artifacts to preserve sampled latent decisions, log probabilities, policy version, AIS replay inputs/state, executed trajectories and simulator rewards. Define serialization, time-step masks and cross-worker reconstruction tests.
4. **Trainer integration**: register a separate **VaVAM latent-policy trainer** with a valid REINFORCE/PPO objective, optimizer, model/optimizer checkpoints, and weight sync. Reusing Cosmos-RL orchestration does not mean reusing the Alpamayo log-probability computation or GRPO trainer unchanged.
5. **Transport and concurrency**: validate local single-process rollout, disk artifact transport and one training update before enabling distributed workers/NCCL, multiple concurrent episodes, and policy weight publication. Guarantee per-session recurrent state isolation.
6. **Fallback**: if AlpaGym's current abstractions impose excessive model-specific integration work, keep the independently working **AlpaSim gRPC driver + a lightweight custom rollout/reward trainer** as the primary experimental loop. Report AlpaGym interoperability as incomplete rather than faking compatibility. The project's AIS/SFT results do not depend on AlpaGym.

**Evidence gates**: (a) a real VaVAM AlpaSim episode with saved action/log-prob/reward records; (b) one nonzero and finite RL update against that episode; (c) deterministic replay/checkpoint restoration; (d) a second rollout using updated weights; (e) only then distributed throughput tests. Report improvements only from held-out, repeatable closed-loop evaluation.

**Hardware schedule**: prefer sequential rollout generation → release inference/simulator memory where possible → policy update → checkpoint → more rollouts. Concurrency is an optimization only after a reliable one-GPU/one-worker version; use measured memory, not an assumed 1/1/2-GPU split.

---

## 6. Ablations — isolate the causal value of AIS

Use identical evaluation suites and approximately matched training data, compute and trainable capacity.

| ID | Policy | What it tests |
|---|---|---|
| B0a | Stock VaVAM-B, AlpaSim preset (`context_length=1`) | Documented AlpaSim baseline |
| B0b | Stock VaVAM-B, native 8-frame context (as trained) | **The memory baseline AIS must beat**: VaViM's own ~3.5 s causal window |
| B1 | VaVAM-B + current-frame DINO adapter, no memory | Extra visual features vs. history |
| B2 | VaVAM-B + frame-stack DINO features | Naive finite memory |
| B3 | VaVAM-B + DINO-GRU without AIS auxiliary losses | Generic learned recurrent memory |
| B4 | VaVAM-B + DINO-GRU trained with AIS losses | AIS objective beyond architecture |
| B5 | VaVAM-B + AIS with learned latent context modes | Optional adaptation / mode switching |
| B6 | VaVAM-B + GRU of VaViM's own features | Is extra DINO backbone truly necessary? |
| Oracle | Privileged simulator state, **diagnostic only** | Upper-reference estimate of observability gap |
| R-A1 / R-A1.5 | Frozen Alpamayo 1 / 1.5 (optional) | External reference only; different sensors (4 cameras) and model scale, so not a matched control |

If B4 beats B0a but not B0b or B3, the correct conclusion is **context helps** (and VaVAM's own context may suffice), not **AIS objectives help**. If B6 matches B4, the extra DINOv3 encoder may be unnecessary. Negative findings should still be saved and discussed.

### Special memory-dependent scenario tests

- Two sequences with similar **current image** but different preceding motion history (e.g., lead-car deceleration vs. acceleration).
- Different traffic reactivity patterns in otherwise comparable scenes.
- Transient occlusion, brief visual ambiguity, stop-and-go traffic and interrupted merges.
- Long-history vs. short-history ablation (e.g., 1, 4, 8, 16 decision steps), controlling observation timestamps.
- State reset / rollout mixing tests to ensure no prior episode contamination.

Construct such cases only from actual supported simulator replays or validated scenario controls. Never claim two images are identical merely because they look roughly similar in a presentation.

---

## 7. Metrics and evidence

### Closed-loop primary metrics

- Collision rate **with denominator defined** (per rollout, per distance, or per time).
- Off-road / drivable-area violation rate.
- Route progress and completion rate.
- Comfort: acceleration, jerk, and heading smoothness if AlpaSim metrics support them.
- Interventions, stalled-rollout and timeout rates.
- Paired performance across held-out scene IDs, with confidence intervals where practical.

### Evaluation statistics (measured in the archived v0 project on `larx`)

- **CATK traffic is stochastic** (top-5 tokens, temperature 1.0). The same deterministic policy flips outcomes between runs, so run ≥ 2 rollouts per scene and never compare single rollouts.
- **Scenes, not rollouts, are the unit.** Use a cluster bootstrap over scenes, and compare policies **paired on the scenes all of them completed**.
- **Power.** Paired per-scene score differences had SD ≈ 0.18–0.26 between behaviourally different policies, so resolving ±0.05 needs **~100 held-out scenes**. A 16-scene split only detects large effects.
- **Simulator aborts.** CATK aborts (e.g. "no map geometry") hit up to ~25% of rollouts on some scene sets. Exclude them from scoring (not score 0), report the count, and compare on matched scenes.
- **Policy-independent failures.** Some scenes fail identically for every policy (e.g. a lane-centre route ~1 m from the human path through parked cars in a narrow street). Report per-scene outcomes so such floors are visible.
- **Renderer limits affect metrics** (§8.1). Pedestrians and riders are never rendered, and ~10% of moving vehicle tracks have no Gaussians, so collisions with invisible actors still count. Report the collision partner and whether it was rendered.
- **Reusable splits.** The frozen, disjoint v0 scene splits (train 64 / val 4 / val_cl 16 / test_cl 16 / eval_big 96; seed 0) are in the git tag `archive/v0-mlp-planner` (`configs/evaluation/splits.yaml`). 152 of those scenes are already downloaded; each scene is ~1.6 GB.

### Training and AIS diagnostics

- Held-out flow-matching loss; trajectory ADE/FDE with correct frame/time alignment.
- Reward prediction quality, event calibration and conditional next-embedding prediction quality.
- AIS ablations vs. recurrent memory without AIS objectives.
- Parameter count, checkpoint sizes, trainable parameter count, peak VRAM, GPU utilization, milliseconds per inference call, rollout wall time, updates per hour.
- For RL only: reward versus **actual simulator interactions**, sampled-latent entropy, KL/ratio diagnostics, stochastic-action diversity, sample efficiency, policy version lag and generalization. Record whether runs use standalone AlpaSim training or the AlpaGym-integrated path.

**Do not optimize a single weighted driving reward and call it a safety result.** Report underlying metrics and qualitative failures separately.

### Per-run artifact contract

Every meaningful run must produce:

```text
outputs/<experiment_id>/
  config.yaml
  git_commit.txt
  environment.txt
  checkpoint_manifest.json
  rollout_policy_version.json       # when training with RL
  gpu_profile.json
  metrics.json
  metrics.parquet                  # if supported
  rollout_logs/                   # native .asl etc.
  videos/                         # actual simulator MP4s
  qualitative_notes.md
```

No fake videos and no screenshots of unrelated driving. For human-readable demos, maintain `outputs/demo_gallery/index.md` with explicit **baseline / AIS / optional RL** checkpoint labels and scene identifiers.

---

## 8. `larx` hardware / systems policy

- Original **Quadro RTX 8000**: Turing / compute capability 7.5; 48 GB VRAM each. Expect to use **FP16**, not BF16-optimized kernels. Validate software support for SDPA or other compatible attention implementations; do not assume FlashAttention-2.
- Start with **one GPU for the VaVAM-B driver / AlpaSim smoke test**, then profile whether the renderer, policy and auxiliary DINO encoding can coexist. AlpaSim microservices can use GPU resources as configured; the diagram is not a GPU allocation guarantee.
- Cache DINOv3 features offline for AIS training; online evaluation must still run the encoder live from allowed observations.
- Prefer frozen VaViM + small trainable adapter (and then 38M action expert) before any broader unfreezing.
- Use `torch.cuda.amp.GradScaler` or the current supported equivalent for FP16 training where appropriate, and log nonfinite gradients/losses. Confirm gradient scaling is compatible with the installed PyTorch version.
- For distributed training: first show correct single-GPU training, then 2- and 4-GPU **DDP** for reproducibility/throughput. Distributed execution is not automatically faster for a small head; report measured speedups and GPU-hours.
- Run long jobs inside persistent `tmux`/scheduler sessions, keep checkpoints on host-mounted persistent storage and support resume. Coordinate GPU ownership with other repos and agents. Never reboot the server, kill others' jobs or overwrite existing checkpoints automatically.
- Datasets and pretrained weights may have research-only, noncommercial, or attribution requirements. Document licenses and do not distribute restricted weights or raw datasets as part of a public portfolio without permission.

### 8.1 Verified `larx` facts (22 October 2026)

**Rendering.**
- NVIDIA's NRE sensorsim has no sm_75 kernels. AlpaSim renders on `larx` only through the in-repo **gsplat 3DGUT sensorsim** (`src/metadrive_ais/render/`, `scripts/serve_gsplat_renderer.sh`), as an external renderer (`configs/alpasim/driver_source/`, `scripts/run_alpasim.sh`).
- It supports **f-theta cameras only** (all four Alpamayo cameras verified). Pinhole cameras, which VaVAM needs, are not yet implemented.
- Fidelity against each scene's recorded photo: median ~18 dB PSNR over 24 scenes. There is no camera response model and the sky is flat, so geometry matches but colour does not. Renders degrade once the ego is ~70–85 m from the recorded path.
- Pedestrians, riders and strollers are never rendered, nor are moving vehicle tracks without dynamic Gaussians (140 of 1,388 in 12 scenes). Parked vehicles are part of the static background.

**Precision and kernels.**
- bf16 is emulated on Turing: Alpamayo-R1-10B takes 87 s per inference in bf16 vs 6.4 s in fp16, with a ~24 GB peak either way.
- FlashAttention 2 is unavailable; use SDPA.
- The VaVAM driver already uses fp16 on x86.
- `patches/alpasim/0003` adds `ALPAMAYO_DTYPE`, SDPA for Alpamayo 1, and `ALPAMAYO_DEVICE_MAP=layer-split` for Alpamayo 2.

**Verified closed loop.** Each completed a real AlpaSim episode with this renderer:

| model | GPUs | fp16 latency per inference |
|---|---|---|
| Alpamayo 1 (10B) | 1 | 3–6 s |
| Alpamayo 1.5 (10B) | 1 | ~5 s |
| Alpamayo 2 Super (36B, layer-split) | 2 | ~13 s |

**GPU budget per AlpaSim stack.** Each stack runs physics + CATK (in containers), a renderer (≤ 15 GB with a 6-scene cache) and a driver. With several stacks, the GPU hosting physics/CATK becomes the bottleneck; `configs/alpasim/topology/single_gpu.yaml` puts a whole stack on one idle GPU. Co-locating two renderers and a stack on one 48 GB card has caused CUDA OOM aborts, so watch `nvidia-smi` before adding a stack.

**AlpaGym.**
- The environment builds with a user-space CUDA 12.8 toolkit (`~/ext/cuda128`, for `grouped-gemm`), and its 469 unit tests pass.
- The local smoke run additionally needs sudo-installed `redis-server`, cuDNN and NCCL headers.
- Its Alpamayo policy bundle hard-codes bf16, which is emulated here, so a VaVAM bundle for M5b should use fp16.

**alpamayo-recipes.** Repository tests pass (32), but the recipes target 8× H100 (SFT), ≥ 5× 80 GB (RL) or Blackwell (FP8/NVFP4). They are out of scope on `larx`.

**Access.** Hugging Face access is granted for the NuRec scenes, DINOv3 (`facebook/dinov3-*`) and `nvidia/Cosmos-Reason2-8B` (needed by Alpamayo 1.5 and AlpaGym).

---

## 9. Milestones, acceptance gates, and stopping criteria

### M0 — Repo and source audit

**Deliver:** dependency and checkpoint inventory; exact local upstream commits; verified working versions; licensing notes; GPU census; existing rollout inventory.  
**Done when:** `docs/SOURCES.md`, `docs/HARDWARE.md` and reproducibility commands are complete. Do not assume current code matches online documentation.  
**Already verified (§8.1):** AlpaSim pinned at `affc2eab` with `patches/alpasim/0001–0003`; gsplat sensorsim; Alpamayo 1/1.5/2 closed loop; AlpaGym environment and unit tests; HF access. Still open: VaVAM assets and license, and the inventory documents.

### M1 — Real VaVAM-B AlpaSim baseline

**Deliver:** OpenCV-pinhole support in the gsplat sensorsim, with a validation report; then a completed closed-loop episode, real MP4, native logs, metrics, reproducible driver configuration and GPU profile, for both B0a (`context_length=1`) and B0b (8 frames).  
**Done when:** a human can watch the actual VaVAM-B replay and trace it to scene/checkpoint/config; video isn't an offline prediction masquerading as a rollout. The B0a/B0b comparison runs on a pre-declared scene set (§7 statistics).

### M2 — Dataset and AIS representation

**Deliver:** validated temporal dataset, DINOv3 feature cache, GRU, reward and next-embedding heads, held-out prediction report.  
**Done when:** no temporal leakage is detected, correct episode resets pass tests, and training runs from a clean checkout.

### M3 — AIS-conditioned VaVAM-B and imitation adaptation

**Deliver:** zero-gate parity, AIS influence/gradient tests, adapter checkpoint and optionally fine-tuned action-expert checkpoint, open-loop plus AlpaSim metrics.  
**Done when:** the *same* pretrained trajectory expert receives AIS-conditioned information and runs real closed-loop episodes with measured outcomes.

### M4 — Research ablations

**Deliver:** B0–B4 as priority, then B5/B6 as justified, paired scenario metrics, prediction diagnostics and qualitative counterexamples.  
**Done when:** the report accurately states whether AIS-specific objectives, generic memory, or extra DINO features explain any differences.

### M5 — Genuine reward-based learning and AlpaGym integration (gated stretch)

**M5a: Standalone AlpaSim RL (primary reward-training target).** Deliver a correctly specified trainable stochastic latent policy, nontrivial trajectory sensitivity to sampled latents, recorded action log probabilities, simulator-derived rewards, at least one reproducible policy update and re-evaluation on held-out scenes. Compare with an equally budgeted fine-tuned non-RL agent. **Done when:** RL updates are mathematically valid for sampled latent actions, checkpoints restore, and the system produces real closed-loop evidence. Reward-weighted imitation must be labeled separately.

**M5b: AlpaGym/Cosmos-RL interoperability (engineering stretch).** Implement VaVAM policy/inference model registration or compatibility layer, rollout data transport and a compatible latent-policy trainer. **Done when:** AlpaGym can execute a VaVAM/AIS episode, score it, perform a verified optimizer step, synchronize updated weights and rerun, with 1/2/4-GPU throughput measured where appropriate. Upstream default Alpamayo training does not count as VaVAM integration.

**Fallback:** if AlpaGym adaptation is infeasible within `larx`'s compute or the pinned code's interface, retain a functioning standalone AlpaSim RL loop and record the precise blocker; do not block AIS/SFT research or fabricate an AlpaGym result.

### M6 — Final demo and portfolio

**Deliver:** side-by-side synchronized rollout MP4s; checkpoint gallery; metrics and failure taxonomy; technical readme; reproducible experiment commands; minimal replay/metrics UI if worth the effort.  
**Done when:** an outside reviewer can understand the objective, inspect real results, reproduce at least a small evaluation and see which components were actually implemented.

**Execution principle:** The *overall* objective is M0–M6. M5a is the reward-training target after supervised validation; M5b is a separate interoperability stretch. Agents should tackle one independently verifiable milestone at a time. Milestones are not complete merely because code exists. Do not silently skip difficult objectives; record feasibility blockers and alternatives explicitly.

---

## 10. Suggested project layout

```text
metadrive-ais/
├── CLAUDE.md                         # concise agent rules and links
├── docs/
│   ├── PROJECT_GUIDELINES.md          # this document
│   ├── ROADMAP.md
│   ├── PROGRESS.md
│   ├── SOURCES.md
│   └── HARDWARE.md
├── configs/
│   ├── baseline/
│   ├── data/
│   ├── ais/
│   ├── sft/
│   ├── rl/
│   └── alpagym/
├── patches/alpasim/                   # small, tested AlpaSim patches (applied by scripts/setup_alpasim.sh)
├── src/metadrive_ais/
│   ├── render/                          # gsplat sensorsim for Turing
│   ├── data/                            # synchronized rollouts/datasets
│   ├── vision/                          # DINO transforms + cache
│   ├── memory/                          # GRU, prediction heads, latent modes
│   ├── models/                          # VaVAM integration, adapters
│   ├── drivers/                         # AlpaSim-compatible custom driver
│   ├── training/                        # flow SFT and genuine latent-policy RL
│   ├── alpagym/                         # optional policy bundle / transport / trainer
│   ├── evaluation/                      # scenes, metrics, video gallery
│   └── utils/
├── tests/
│   ├── test_shapes_and_coordinate_frames.py
│   ├── test_zero_gate_parity.py
│   ├── test_memory_episode_reset.py
│   ├── test_no_future_leakage.py
│   ├── test_adapter_gradients.py
│   ├── test_driver_interface.py
│   ├── test_latent_policy_logprobs.py
│   ├── test_rl_episode_serialization.py
│   └── test_alpagym_interop.py
├── scripts/
├── checkpoints/                        # persistent; ignore large files in git
└── outputs/                            # metrics/logs/videos; not mock assets
```

Track external upstream code as pinned submodules, vendored references with attribution, or separately cloned dependencies; avoid unreviewable edits directly into upstream repos. Keep patch surface small and testable.

---

## 11. Instructions for coding agents

1. Read this guideline and inspect **the actual locally checked-out model code** before designing a new interface. Identify the action-expert forward path, diffusion/flow training loss, video-feature shapes, input preprocessing, output trajectory semantics, and AlpaSim driver boundary.
2. Inspect existing files and experiments; **do not claim the baseline is missing until you have searched the actual outputs**.
3. Do not substitute scaffolding, stubs, fabricated rollouts or static video screenshots for integration tests.
4. Do not silently modify VaVAM's action semantics, trajectory horizon, navigation conditioning, or native controller configuration.
5. Use no privileged simulation state in runtime policy input without marking an explicitly separate oracle experiment.
6. Log checkpoint provenance, licenses and dataset splits. Don't overwrite pretrained weights.
7. Update `docs/PROGRESS.md` at each meaningful achievement: change, test run, observed metrics, artifact paths, blockers and next test.
8. Keep experiments small enough for `larx`. Stop runaway loops and GPU contention; report real resource constraints rather than guessing.
9. Be scientifically precise: **AlpaSim is not AlpaGym; VaVAM has no out-of-box AlpaGym GRPO; action-head SFT is not LLM SFT; AWR is not GRPO; next-feature prediction is an AIS proxy, not a proof of information sufficiency.**
10. Prefer a working, benchmarked, faithful end-to-end system over extra unrelated features.

---

## 12. Immediate execution brief

**Next objective for Claude:**
1. Download VaVAM-B assets and record their license.
2. Add and validate OpenCV-pinhole cameras in the gsplat sensorsim.
3. Run a genuine stock VaVAM-B closed-loop rollout through `scripts/run_alpasim.sh`, both as B0a (1 frame) and B0b (8 frames). Record video, metrics, checkpoint and GPU measurements.
4. Identify the precise tensor interface at which a zero-initialized AIS conditioning adapter can be introduced without altering existing trajectory generation.

Write these discoveries to `docs/PROGRESS.md` and update the roadmap. The *ultimate project scope* remains M0–M6; this is merely the first evidence gate.

**Success condition for the project:** A trained AIS-conditioned VaVAM-B driving policy whose real closed-loop performance can be compared fairly against the unmodified model and recurrent-memory controls, with reproducible runs and compelling AlpaSim videos. The reward-training branch must use a valid sampled latent policy; AlpaGym interoperability is a separate, explicitly validated engineering stretch, never assumed from native simulator support.

---

## 13. Verified upstream references and license checks

- [Valeo.ai — VideoActionModel official repository](https://github.com/valeoai/VideoActionModel): weights and parameter counts; `vam/train.py`, imitation action-learning instructions; model code; license statements. **Code MIT, model weights research-only.**
- [VaViM and VaVAM paper](https://arxiv.org/abs/2502.15672): video features, frozen-backbone action learning, flow matching, trajectory conditioning.
- [NVIDIA AlpaSim tutorial](https://github.com/NVlabs/alpasim/blob/main/docs/TUTORIAL.md): `vavam-b` asset download, native driver, local closed-loop deployment and video outputs.
- [AlpaSim operations](https://github.com/NVlabs/alpasim/blob/main/docs/OPERATIONS.md): custom VaVAM assets, custom driver service image and configuration.
- [AlpaSim video-model notes](https://github.com/NVlabs/alpasim/blob/main/docs/VIDEO_MODEL.md): VaVAM latest-frame context in the documented preset; distinction between image and video renderer deployments.
- [NVIDIA AlpaGym](https://github.com/NVlabs/alpagym): currently Alpamayo 1.5 policy support, AlpaSim simulator plus Cosmos-RL training orchestration.
- [AlpaGym runtime interfaces](https://github.com/NVlabs/alpagym/blob/main/packages/runtime/README.md): per-session `Policy`, shared `InferenceEngine`, released-model-specific `InferenceModel`, rollout artifacts and transfer layers.
- [DINOv3 model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md): supported ViT-S/B and other encoders; review relevant license/access conditions.
- [AIS framework — Subramanian, Sinha, Seraj & Mahajan, JMLR 2022](https://www.jmlr.org/beta/papers/v23/20-1165.html): reward/prediction sufficiency and recursive information-state compression.

- Local checkout, AlpaSim `affc2eab`: `src/wizard/configs/driver/vavam.yaml` and `vavam_configs.yaml` (1-frame preset, 2 Hz, pinhole `extra_cameras`); `src/driver/src/alpasim_driver/models/vam_model.py` (fp16, default 8 frames); `vam/action_expert/video_action_model.py` (`finetuning_timesteps=8`).

**Source verification date:** 22 October 2026. Upstream APIs and configs change; local pinned commits take precedence over assumptions in this document.
