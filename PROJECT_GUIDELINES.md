# MetaDrive-RL: Mode-Conditioned VLM Post-Training in Closed-Loop Autonomous Driving

> **Project guideline · v0.1 · October 21, 2026**  
> **Platform:** NVIDIA AlpaSim / AlpaGym  
> **Compute budget:** 4 × NVIDIA Quadro RTX 8000 (48 GB each, Turing, compute capability 7.5)  
> **Project type:** Independent, portfolio-first applied ML research and engineering  
> **Working title:** **MetaDrive-RL — Distributed RLVR and Approximate Information States for Closed-Loop Vision-Language Driving**

---

## 0. Executive summary

Build a **closed-loop autonomous-driving agent** whose frozen or parameter-efficiently fine-tuned vision-language backbone chooses *high-level driving decisions* (e.g., cruise, yield, brake, merge), and whose **mode-conditioned trajectory planner** turns those decisions into AlpaSim-compatible planned paths. Improve its behavior through a staged post-training pipeline:

**Frozen baseline → SFT → preference optimization (DPO) → simulator-reward RL (GRPO/RLVR) → AIS-aware adaptation.**

The primary outcome is a **convincing, reproducible engineering portfolio**: distributed multimodal fine-tuning, environment rollouts, online RL, deployment, evaluation, and striking side-by-side driving demos. The secondary research question is whether a **task-aware approximate information state (AIS)** improves context adaptation and closed-loop performance over memoryless and generic recurrent policies.

**Principle:** Prioritize a running system and rigorous empirical comparisons over new control-theoretic theorems. A compact, well-executed project is better than an unfinishable foundation-model training program.

### Success looks like

1. A reproducible AlpaSim closed-loop evaluation with a custom driver and replayable video.
2. A policy whose discrete/continuous high-level decisions **actually change** the vehicle's generated trajectory.
3. Working **SFT + preference optimization + GRPO/RLVR** checkpoints, trained with documented dataset provenance and reward definitions.
4. Instrumented multi-GPU training or rollout execution on the available 4-GPU server, with measured throughput and memory usage.
5. A matched-budget AIS ablation and an interactive or prerecorded baseline-versus-trained driving showcase.

### Non-goals for v1

- Training a large VLA foundation model from scratch.
- Formal, unconditional collision-avoidance certification.
- Inventing a novel game-theoretic formulation or proving new MPC convergence theorems.
- Reproducing a full 8× H20/H100-scale paper training recipe on the older GPUs.
- Deploying a real-world safety-critical driving system.

---

## 1. Guiding research and engineering questions

**Engineering question:** Can a small, modular trainable policy leverage a pretrained vision-language representation and AlpaSim's closed-loop, verifiable feedback to outperform its own frozen/SFT baselines in route progress, safety, and adaptation?

**AIS research question:** Holding the driving backbone, planner, scenarios, and RL budget fixed, does an explicitly trained approximate information state outperform generic memory on partially observed, history-dependent driving decisions?

**Stretch question:** When the traffic environment changes (e.g., yielding tendencies or cut-in behavior), can an AIS/context-conditioned policy adapt within an episode faster than a reactive policy?

Avoid presupposing the answer. Negative or mixed results, including representation collapse and poor reward transfer, are still valid findings if accurately measured.

---

## 2. System architecture

```text
AlpaSim (camera frames, ego telemetry, route, rewards)
                          │
                          ▼
       Frozen image/VLM representation backbone
          (small model; compatibility first)
                          │  visual embedding e_t
                          ▼
     Context / information-state encoder (optional)
       ├── current-frame only             [baseline]
       ├── K-frame stack                  [baseline]
       ├── recurrent history encoder      [baseline]
       └── reward/transition-trained AIS  [research]
                          │  z_t
                          ▼
       Trainable high-level decision policy
     mode + target speed/headway + maneuver intent
          (LoRA/head; SFT → DPO → GRPO)
                          │
                          ▼
         Mode-conditioned trajectory planner
  candidate trajectories + feasibility/kinematic checks
                          │  planned ego trajectory
                          ▼
      AlpaSim low-level trajectory controller
                          │
                          ▼
     Physics + renderer + reactive traffic simulator
                          │
                          ├── closed-loop metrics/reward
                          └── replay logs / comparison videos
```

### 2.1 Separation of responsibilities

| Component | Owns | v1 training status |
|---|---|---|
| Visual/VLM encoder | Encode camera observations and, optionally, route text | **Initially frozen**; LoRA only after performance is established |
| Information-state encoder | Summarize history into decision-relevant context | Optional; trainable in AIS experiments |
| High-level policy | Driving mode and compact parameters | **Primary post-training target** |
| Trajectory generator | Convert high-level decisions into valid candidate paths | Start **deterministic**, with fixed feasibility checks |
| AlpaSim controller | Track chosen reference trajectory | Preserve simulator implementation for fair comparisons |
| Simulator + evaluator | Advance closed loop and score outcomes | Fixed across policy comparisons |

**Key design decision:** Do **not** ask the VLM to directly emit low-level steering/throttle. Train decisions with a clear, controllable interface and leave vehicle tracking to the existing controller.

### 2.2 Initial high-level action space

Start with a constrained, interpretable action schema:

```json
{
  "mode": "CRUISE | FOLLOW | YIELD | BRAKE | CHANGE_LANE_LEFT | CHANGE_LANE_RIGHT",
  "speed_target_mps": 9.0,
  "time_headway_s": 2.0,
  "planning_horizon_s": 2.0
}
```

All numbers above are **illustrative placeholders**, not calibrated defaults. Use bounded/discretized parameter bins for initial GRPO (e.g., relative speed choices and a few headway settings) to keep sampling and reward attribution manageable. Validate parameter ranges against the simulator and road context.

A deterministic mapper generates and ranks geometrically/kinematically feasible trajectories. Start with simple lane-following, longitudinal acceleration/deceleration, and lane-change primitives; add more sophisticated planning only if the baseline needs it.

**Hard requirement:** In a fixed scenario, switching modes must produce meaningfully different planned trajectories and measured ego behavior. Demonstrate this before any expensive fine-tuning.

### 2.3 Keep two different kinds of “mode” separate

- **Ego action mode** `m_t`: the policy's maneuver choice (yield, merge, follow, etc.). This is observable and controllable.
- **Latent traffic/context mode** `c_t`: inferred properties of the environment (cooperative, aggressive, uncertain, congested, etc.). This is hidden; it should emerge from observation history rather than be leaked from simulator ground truth.

The distinction matters for the meta-RL/AIS study: history is useful when it changes belief about `c_t`, which then influences decisions about `m_t`.

---

## 3. Post-training curriculum

### Stage A — Frozen model and rule-based control

- Create a valid custom AlpaSim driver adapter: observations in, trajectory out.
- Benchmark a stock/reference driver when practical and a compact frozen VLM + rule-based mode selector.
- Persist observations, candidate choices, selected paths, controller outcomes, and reward components.
- Produce first closed-loop video and a scenario-level evaluation report.

**Output:** a working simulator integration, not merely offline trajectory predictions.

### Stage B — Supervised fine-tuning (SFT)

- Collect or curate examples of `(observation/history, route instruction) → high-level action`.
- Label from an expert/reference planner or carefully audited trajectory outcomes; avoid using the final test scenarios as supervision.
- Train the action head or a LoRA adapter of a small VLM using supervised cross-entropy over structured action tokens.
- Evaluate **closed-loop improvement**, not just action-label accuracy.

**Output:** an SFT checkpoint, training curves, dataset manifest, and rollout videos.

### Stage C — Preference optimization (DPO)

- For matching scenario states, generate candidate action/trajectory alternatives and replay them when the simulator supports comparable initialization.
- Generate pairwise preference labels based on **explicitly defined criteria** (safety, smoothness, progress); separately collect a small genuine **human-preference** set if claiming RLHF.
- Run DPO (or another practical preference objective), compare against SFT under matched scenarios, and measure over-cautiousness.
- Distinguish **simulator-derived preference optimization** from **RLHF with human feedback** in README and resume claims.

**Output:** DPO checkpoint plus a labeled-preference audit and a small preference-review UI/notebook.

### Stage D — RLVR / GRPO using AlpaSim

- Treat AlpaSim outcomes as the objective verifier: progress, collisions, off-road events, comfort, task completion.
- From a matched initial scenario (or reproducible scenario seed), sample a **group** of high-level policy decisions/rollout candidates, obtain terminal or segmented closed-loop rewards, and form group-relative advantages.
- Record the policy log-probabilities of **the actual sampled action tokens**. Preserve the action-to-trajectory mapping used during rollout.
- Train using a GRPO-style clipped objective with an appropriate reference-policy penalty and monitor KL, entropy, reward hacking, and variance.
- Use episode-level rewards initially; add shorter rollout horizons or reward-to-action assignment only when credit assignment requires it.
- Keep an explicit account of simulator interactions, GPU-hours, and evaluation scenes.

**Output:** replayable GRPO training run, distributed-systems metrics, best/last checkpoints, and comparison videos.

> **Terminology:** Here “RLVR” means reinforcement learning using **simulator-verifiable** outcomes. These rewards are objectively measured within the simulator, but are not equivalent to ground-truth real-world driving safety or transfer performance.

### Stage E — AIS / meta-adaptation extension

- Freeze the same visual backbone and high-level action space across memory baselines.
- Train history representations with optional reward- and transition-predictive objectives.
- Compare history encoders with matched downstream training budgets.
- Test policy performance on scenes where the current observation does not reveal relevant recent interactions or other agents' tendencies.

**Output:** an AIS ablation report with prediction losses, reward curves, adaptation plots, and representative matched scenario replays.

---

## 4. Approximate information state (AIS) experiment

### 4.1 Minimal formulation

Let the driving observation history be

\[
h_t=(o_0,a_0,o_1,\ldots,a_{t-1},o_t),
\]

with `o_t` including available camera embeddings, ego telemetry, and route information, but **not** simulator-only privileged state. Learn a compact summary

\[
z_t=\phi_\eta(h_t),\qquad a_t\sim\pi_\theta(\cdot\mid z_t).
\]

Operationally, a recurrent AIS module may take the form

\[
z_{t+1}=f_\eta(z_t,a_t,e_{t+1}),\quad e_{t+1}=E_{\mathrm{frozen}}(o_{t+1}).
\]

An AIS-inspired learning objective can combine decision-relevant predictions:

\[
\mathcal L_{\mathrm{AIS}}=
\lambda_r\,\mathcal L_{\mathrm{reward}}+
\lambda_p\,\mathcal L_{\mathrm{transition}}.
\]

The reward head predicts the conditional reward or return under an action; the transition head predicts a distribution over the **next learned state/observable embedding**, using an explicitly chosen likelihood, divergence, or proper scoring rule. Include regularization or anti-collapse tests as needed.

**Interpretation caution:** Low empirical predictive loss does not prove that the learned representation satisfies a uniform AIS approximation bound. This is an empirical testbed, not an automatic theoretical certificate.

### 4.2 Controlled comparison

| Representation | Uses history? | Additional learning objective? | Role |
|---|---:|---:|---|
| Current-frame features | No | No | Memoryless baseline |
| Frame stacking (`K` recent embeddings) | Yes, fixed window | No | Cheap memory baseline |
| GRU/transformer history encoder | Yes | Task reward / policy training | Generic memory baseline |
| AIS-trained GRU/transformer | Yes | Reward + transition prediction | Main research variant |
| Privileged simulator-state policy | Oracle state | Not comparable deployment input | Diagnostic upper-reference baseline only |

Keep backbone checkpoints, action space, reward definitions, scenario splits, and online interaction budget fixed for the main comparison. Match trainable-parameter counts or report the mismatch explicitly.

### 4.3 High-value scenarios

- **Observation aliasing:** Similar current camera frames, different recent interactions, different appropriate driving decisions.
- **Changing traffic behavior:** Episode-specific or time-varying yielding/cut-in patterns, if AlpaSim scene controls allow reliable manipulation.
- **Temporary occlusion:** Relevant vehicles temporarily leave view; historical context may matter.
- **Adaptation:** Evaluate decisions immediately before and after the agent observes a traffic response.

If exact matched-history interventions are unavailable, document proxy scenario construction and avoid causal claims based solely on visual resemblance.

### 4.4 Analysis and hypotheses

- **H1:** History-aware policies outperform memoryless policies on carefully designed partially observable scenarios.
- **H2:** AIS predictive training improves robustness or sample efficiency relative to generic recurrence, *conditional on actually retaining decision-relevant information*.
- **H3:** Improvements are most visible under context changes, not on uncomplicated lane following.

Measure adaptation lag, context-probe accuracy where labels exist, reward/transition prediction calibration, task completion, collisions, progress, and training sample efficiency. Run seed-level uncertainty estimates; distinguish statistical variability from systematic improvements.

---

## 5. Compute-aware implementation strategy

### 5.1 Treat hardware compatibility as a hard gate

The original **Quadro RTX 8000** is a Turing GPU (48 GB VRAM each; compute capability 7.5). This is **192 GB aggregate**, not one 192 GB device. The GPUs do **not** provide native BF16 support, and common FlashAttention-2 builds require newer GPU architectures.

- Start with **FP16**, PyTorch-compatible/eager or supported SDPA attention, and small input resolutions/context windows.
- Benchmark the exact `transformers`, CUDA, PyTorch, inference runtime, and LoRA stack **before** integrating the simulator with training.
- Avoid assuming that H100/A100/RTX 6000 Ada example commands work without modification.
- Prefer a roughly **3B-class multimodal backbone** with LoRA/QLoRA *only if the chosen quantization stack supports Turing correctly*. An initial vision encoder + small policy head is an acceptable fallback.
- Quantify visual encoding, policy inference, render time, and training step time separately.

### 5.2 Candidate compute configurations — benchmark, don't assume

| Mode | Tentative allocation | Notes |
|---|---|---|
| Environment smoke tests | 1+ GPUs as required | AlpaSim components may have separate requirements |
| Offline SFT/DPO | 2–4 GPUs | Start with LoRA + DDP; use FSDP/ZeRO only if useful and supported |
| Simulator rollout collection | GPU(s) for rendering + inference | May be bottlenecked by rendering, CPU, or gRPC, not by policy |
| Online GRPO | Asynchronous or alternating rollout/train phases | Try concurrent workers only after memory/throughput profiling |
| Final evaluation | Reserve deterministic evaluation configuration | No gradient computation |

A realistic fallback is **collect rollouts → write immutable batches → train LoRA → evaluate → repeat**. This still demonstrates RLVR, distributed rollout orchestration, checkpointing, and data lineage, without forcing real-time joint training and rendering on four older GPUs.

### 5.3 Distributed-systems skills to deliberately demonstrate

- PyTorch `torchrun` and **DDP** for supported multi-GPU training; optional FSDP/ZeRO when justified.
- LoRA checkpointing, resumption, reference policy/versioning, and reproducible batch generation.
- Distributed rollout workers, queue/backpressure handling, timeouts, failure retries, and episode identifiers.
- GPU/CPU memory profiling, tokens/sec, samples/sec, rollouts/hour, simulator wall time, communication overhead.
- Experiment tracking (e.g., MLflow or W&B), structured configuration, seed manifests, and artifacts.

Do **not** invent a speedup claim. Report measured throughput versus 1-GPU or smaller-worker baselines with identical workloads.

---

## 6. Evaluation protocol

### 6.1 Baselines/checkpoints

1. Stock/reference AlpaSim policy, if runnable on the hardware.
2. Frozen small-VLM mode policy + fixed planner.
3. SFT policy.
4. SFT + DPO policy.
5. SFT (or SFT + DPO) + GRPO policy.
6. GRPO policy with memoryless versus recurrent versus AIS context (research arm).
7. Privileged-state oracle **diagnostic**, explicitly segregated from deployable methods.

### 6.2 Metrics

| Category | Required measurements |
|---|---|
| Closed-loop driving | Route progress, route/goal completion, collision counts/rates, off-road events |
| Driving quality | Comfort proxies (acceleration/jerk), lane keeping, unnecessary stops, travel time |
| Safety/progress trade-off | Reward decomposition, collisions versus distance traveled, risk–progress Pareto curve |
| RL training | Episode return, group-relative advantage statistics, KL to reference, entropy, reward variance |
| Generalization | Held-out routes/scenes, reactive-traffic variants, distribution-shift performance |
| Efficiency | Environment interactions, rollouts/hour, GPU-hours, peak VRAM, inference latency, training throughput |
| AIS diagnostics | Reward/transition prediction error, calibration, performance conditioned on history ambiguity |

Prefer rates with meaningful denominators (e.g., events per episode or per distance driven) and include completion/progress so that stopping forever cannot score as “safe.”

### 6.3 Comparison rules

- Fix training/validation/test splits **before** tuning, and reserve a held-out test set.
- Use the **same simulator version, scenario configuration, scoring weights, and low-level controller** across comparisons.
- Prefer matched scenario seeds and repeated rollouts; document nondeterminism and reactive-traffic settings.
- Report mean, uncertainty intervals, and individual failure modes; inspect tail risk, not just mean reward.
- Include **no-history**, **no-DPO**, **no-RLVR**, and **no-AIS-objective** ablations.
- Attribute performance gains conservatively: SFT may be benefiting from expert data; DPO from simulator preferences; GRPO from additional environment interactions.
- Validate that there is no leakage of simulator ground-truth labels into the online policy observation channel.

### 6.4 Reward shaping and reward-hacking tests

AlpaSim reward configuration can incorporate progress, collisions, off-road events, and other terms. Treat any example reward weights as starting references, **not immutable ground truth**. Inspect the active configuration and log all components.

At minimum check for:

- Conservative policy that avoids collisions by refusing to move.
- Fast policy that optimizes progress despite unsafe merges.
- Spurious dependence on simulator-specific privileged information.
- Reward differences caused by rollout duration, reset, or scene sampling rather than policy quality.
- GRPO instability when all sampled actions receive near-identical rewards.

---

## 7. Milestones and acceptance gates

### M0 — Feasibility and compatibility gate

**Tasks:** Clone/pin AlpaSim; inspect its actual driver interface and expected trajectory format; run the minimal environment; verify the CUDA/software stack; benchmark a compact VLM in FP16; estimate storage and scene requirements.

**Exit gate:** At least one local closed-loop episode completes, with logged inputs/outputs and a saved replay. If this fails, investigate a smaller scene/configuration or an offline trajectory-planning prototype before pursuing GPU-heavy training.

### M1 — Custom mode-conditioned driver and first demo

**Tasks:** Implement an action parser, deterministic trajectory candidates, constraint/fallback logic, rollout data schema, and video overlay showing selected mode and path.

**Exit gate:** Same scenario + different selected modes yields different physically meaningful executed paths. At least one reproducible baseline comparison video exists.

### M2 — SFT + evaluation harness

**Tasks:** Collect expert examples, train LoRA or policy head, set up experiment tracking and held-out evaluation, publish training and inference scripts.

**Exit gate:** SFT checkpoint loads through the same driver API and has a documented closed-loop result against the frozen baseline.

### M3 — Preference optimization

**Tasks:** Construct auditable paired preferences, train DPO, test simulator-derived versus human-rated preferences if available, measure the safety/progress trade-off.

**Exit gate:** DPO produces reproducible metrics and behavior comparisons, including at least one case where its learned preference affects a driving decision.

### M4 — Distributed RLVR

**Tasks:** Parallel rollout collection, group-relative reward construction, GRPO policy updates, checkpointing, distributed-training instrumentation, artifact tracking.

**Exit gate:** End-to-end repeated collection → update → evaluation loop runs without manual intervention, with actual sampled log-probabilities, policy versioning, and a documented interaction/compute budget.

### M5 — AIS research extension

**Tasks:** Recurrent encoder, AIS predictive heads, context-shift and aliasing scenarios, controlled ablation and adaptation analysis.

**Exit gate:** Matched-budget AIS comparison with metrics, uncertainty estimates, and at least one interpretable failure/success case illustrated in a replay.

### M6 — Portfolio publication

**Tasks:** Polish README, architecture diagram, demo landing page, benchmark report, reproducible commands, model/data cards (where licensing permits), and a short narrated video.

**Exit gate:** A reviewer can understand the task, training methods, hardware constraints, experimental results, and demos without running the entire simulator.

**Priority rule:** If time is constrained, ship **M0–M4 + polished demos** before investing heavily in AIS or extra mathematical extensions.

---

## 8. Reproducible project organization

```text
metadrive-rl/
├── README.md
├── PROJECT_GUIDELINES.md
├── configs/
│   ├── simulator/
│   ├── model/
│   ├── training/
│   └── evaluation/
├── src/metadrive/
│   ├── alpasim/              # gRPC/driver integration, observations, trajectories
│   ├── perception/           # frozen VLM adapter and feature caching
│   ├── planner/              # mode-to-trajectory mapping, safety/fallback checks
│   ├── policy/               # structured action policy, tokenizer, adapters
│   ├── ais/                  # history encoder, predictive heads
│   ├── data/                 # rollout datasets, preferences, manifests
│   ├── training/             # SFT, DPO, GRPO, distributed orchestration
│   ├── evaluation/           # closed-loop metrics, matched-seed benchmark
│   └── visualization/        # overlays, replay, comparison rendering
├── notebooks/
│   ├── 01_simulator_baseline.ipynb
│   ├── 02_sft_analysis.ipynb
│   ├── 03_preference_analysis.ipynb
│   ├── 04_grpo_diagnostics.ipynb
│   └── 05_ais_ablation.ipynb
├── scripts/
│   ├── smoke_test.sh
│   ├── collect_rollouts.sh
│   ├── train_sft.sh
│   ├── train_dpo.sh
│   ├── train_grpo.sh
│   └── evaluate.sh
├── tests/
├── reports/
└── demos/
```

Keep large scene data, extracted frames, rollout shards, and model weights outside Git; version their manifests/checksums and licensing constraints. Include `.env.example`, pinned environments, config snapshots, and seed lists.

### Minimum rollout record

Each episode should capture:

- `run_id`, `episode_id`, `scenario_id`, simulator/model/checkpoint/config versions, seed.
- Observation timestamps and identifiers (with references to images, not bloated inline blobs).
- Available route/ego observations and policy history window.
- Sampled high-level action, action-token IDs, log-probabilities, planner trajectory, fallback/intervention flags.
- Step/episode rewards with **decomposed reward terms**, terminal condition, evaluation metrics.
- Execution timings and relevant GPU/CPU usage.

This logging discipline is essential for valid GRPO credit assignment, audits, and replay synchronization.

---

## 9. Demo and portfolio plan

### Flagship demo: same scenario, different checkpoints

Show a synchronized comparison of **frozen → SFT → DPO → GRPO** on identical or closely matched scenario initializations:

- Main camera view and/or bird's-eye replay.
- Ego trajectory overlay and selected high-level mode.
- Speed, headway, and important neighboring-agent interactions.
- Explicit reward decomposition and episode outcome.
- Brief *behavioral description* generated or logged by the agent; avoid presenting unvalidated reasoning text as a causal explanation.

### AIS demo: same-looking present, different past

Show two histories leading to similar current visual frames. Compare a memoryless agent and an AIS-conditioned agent to test whether historical interactions change the maneuver choice. Label the scenarios clearly and show the preceding observation window.

### Engineering dashboard

Show:

- Training progression and best-checkpoint selection.
- Episode reward, progress, collision rate, and route completion across stages.
- GPU utilization/VRAM and 1-versus-multi-GPU throughput.
- Evaluation scenario explorer and rollout videos.
- Failure-analysis cards with actual metrics and replay links.

**Do not fabricate illustrative metrics** in final public demos. Separate mockups from measured results.

### Resume outputs (fill with real numbers after benchmarking)

- Built a mode-conditioned VLM autonomous-driving agent in NVIDIA AlpaSim, with a custom gRPC rollout interface and closed-loop trajectory planning.
- Implemented multimodal SFT, preference optimization, and simulator-reward GRPO with distributed rollout collection and parameter-efficient fine-tuning on **4× Quadro RTX 8000**.
- Developed a reproducible closed-loop benchmark and visual comparison dashboard; measured **[X]** improvement in **[metric]** over **[baseline]** on **[N]** held-out scenarios.
- Evaluated AIS-trained recurrent policies against frame-only and generic-memory baselines under partial observability and changed traffic behavior.

Only retain bullet claims for components actually implemented; replace placeholders with reproducible values.

---

## 10. Risks and fallback decisions

| Risk | Detection | Fallback |
|---|---|---|
| AlpaSim rendering/inference unsupported or too slow on Turing | M0 startup failure or unusable rollout throughput | Use smaller simulator config, reference driver, cached visual features for offline experiments, and reduced episode set; document limitations |
| Main Alpamayo backbone too demanding | OOM, kernel incompatibility, unacceptable latency | Small compatible VLM or frozen visual encoder + trainable action head |
| Categorical modes do not affect actual executed behavior | Identical trajectories/controls under different decisions | Fix planner mapping and action bins **before** SFT/RL |
| Online joint training too expensive | GPU contention, idle workers, low tokens/sec | Alternate batched rollouts and LoRA updates; use asynchronous queues |
| Sparse/degenerate simulator reward | GRPO groups have low reward variance | Better episode sampling, curriculum, carefully logged intermediate outcomes |
| Reward hacking / excessive hesitation | Collision reduction accompanies near-zero progress | Multi-metric selection, hold-out scenarios, progress constraints |
| AIS predicts well but does not help decisions | Low predictive error, no closed-loop gain | Report negative result; change history-sensitive scenario design and inspect shortcuts |
| Comparisons are confounded | Unequal pretrained data, compute, seeds, driver | Match budgets/configs or explicitly qualify claims |
| Data/model license complications | Release restrictions in scene data and weights | Publish code, configs, aggregate metrics, and appropriately licensed videos only |

---

## 11. Relevant starting points

**Platform:**

- [NVIDIA AlpaSim](https://github.com/NVlabs/alpasim) — inspect the current design docs, driver protocol, controller behavior, and supported configs in a pinned commit.
- [NVIDIA Cosmos-RL](https://github.com/NVIDIA/Cosmos-RL) — inspect current AlpaGym integration and whether the desired policy/rollout interface is directly supported.

**Research anchors (architectural inspiration, not prerequisites to reproduce fully):**

- [VLM-MPC](https://www.sciencedirect.com/science/article/pii/S0968090X25004917) — high-level language/vision reasoning plus trajectory control.
- [DriveWorld-VLA](https://proceedings.mlr.press/v306/jia26f.html) — latent world modeling and VLA driving.
- [WorldRFT](https://ojs.aaai.org/index.php/AAAI/article/view/38149) — world-model planning and RL fine-tuning.
- [AlphaAdj](https://arxiv.org/abs/2603.21142) — adaptive safety-filter ideas.
- AIS literature relevant to **Tao's information-state / policy-gradient work** — insert the exact paper/preprint citation before making paper-specific theoretical claims.

**Implementation note:** Recheck repository versions, model licenses, simulator availability, supported GPUs, and training interfaces at the start of M0. Links and architecture sketches are starting points, not guarantees that the complete stack runs as written on Turing GPUs.

---

## 12. Immediate next actions

1. **Pin the environment.** Record GPU model, driver, CUDA/PyTorch versions, host RAM/storage, and usable scene data.
2. **Run one AlpaSim episode.** Save a camera replay, raw rewards, and an exact driver input/output trace.
3. **Implement the action-to-trajectory adapter.** Prove that `YIELD` versus `CRUISE` produces distinguishable closed-loop outcomes.
4. **Benchmark two compact model paths.** Compare a frozen vision encoder + policy head with a small LoRA-compatible VLM under actual simulator latency constraints.
5. **Freeze the experimental protocol.** Define held-out scenes, reward metrics, and the baseline evaluation commands before collecting training data.

> **Decision rule:** No distributed GRPO training until the simulator runs, the driver controls the car, rollouts are reproducible enough to compare, and the reward/data logger is correct.

---

## Project identity in one sentence

**MetaDrive-RL is an end-to-end multimodal post-training and distributed RL engineering project, with a focused AIS/meta-adaptation experiment, evaluated through actual closed-loop driving behavior in AlpaSim.**

