# MetaDrive-AIS

Closed-loop driving research with NVIDIA AlpaSim on 4× Quadro RTX 8000
(Turing, sm_75, 48 GB each). The question (`PROJECT_GUIDELINES.md`): does an
approximate-information-state (AIS) memory over frozen DINOv3 features and
executed actions make **VaVAM-B** a better closed-loop driver than its own
8-frame context, and can simulator-reward learning improve it further?

**Results so far: [`reports/summary.md`](reports/summary.md).** In short:
- An adapter on VaVAM's action expert improves open-loop imitation, mostly
  because it supplies the executed motion (speed).
- In closed loop that adapter is worse than stock VaVAM-B on the 48-scene
  eval_big subset (paired score −0.09 [−0.16, −0.02]).
- Memory and the AIS objective add nothing measurable.
- Latent-policy RL (PPO on exact log-probs through AlpaSim) did not improve on
  stock VaVAM within 512 rollouts per run.

Every comparison was declared before running (`docs/PROGRESS.md`); negative
results are kept.

The earlier MLP-over-planner pipeline is preserved under the Git tag
`archive/v0-mlp-planner`.

## What is implemented and verified

| component | status |
|---|---|
| VaVAM-B in AlpaSim, corrected loader | AlpaSim drops VaVAM's muP base shapes (flow velocity 4×); `patches/alpasim/0005` + `VAVAM_MUP_SHAPES_DIR` restore them. Baselines B0a\*/B0b\* (`reports/m1_corrected_baselines.md`) |
| gsplat sensorsim (`src/metadrive_ais/render/`) | Turing replacement for NVIDIA's NRE renderer: f-theta and OpenCV-pinhole cameras, rolling shutter, dynamic actors; pinhole validated against OpenCV (43–49 dB) |
| VaVAM-AIS AlpaSim driver (`drivers/vavam_ais.py`, entry point `vavam_ais`) | per-session memory; zero-gate output bit-identical to stock VaVAM; seeded latent policies with exact log-prob logging |
| Episode datasets (`data/`, `vision/`) | frames, route commands, executed motion and outcomes from AlpaSim logs; DINOv3-S and VQ-token caches |
| Memory / AIS (`memory/`) | frame, stack and GRU encoders; reward and next-embedding heads; scene-held-out evaluation |
| Flow-matching SFT (`training/flow_sft.py`) | through the frozen VaVAM-B action expert, gated adapter, optional latent channel (CVAE, hindsight PCA) |
| Latent-policy RL (`training/latent_rl.py`, `scripts/latent_rl_loop.sh`) | on-policy PPO over AlpaSim rollouts: scene-group baseline, AlpaSim scored-window credit, KL control, held (option) latents; AWR comparator (`training/latent_awr.py`) |
| Evaluation (`evaluation/`) | scene-cluster bootstrap, matched scenes, paired differences, abort accounting; forced-replay open-loop error; speed profiles |
| Alpamayo 1 / 1.5 / 2 Super in AlpaSim | closed loop on Turing: fp16, SDPA, layer-split over 2 GPUs for A2 (`patches/alpasim/0003`) |
| AlpaGym (M5b) | environment builds, 469 unit tests pass; smoke run blocked on system packages (`redis-server`, cuDNN/NCCL headers) |

Demo: `outputs/demo_gallery/` (built by `scripts/make_gallery.py`, index in
[`docs/demo_gallery.md`](docs/demo_gallery.md)). Side-by-side AlpaSim rollout
videos of stock VaVAM-B, the AIS adapter and the RL policy on the 16 val_cl
scenes.

## Reproduce a small evaluation

```bash
scripts/setup_alpasim.sh                 # AlpaSim at the pinned commit + patches
scripts/setup_render_env.sh              # renderer conda env (metadrive-render)
export VAVAM_MUP_SHAPES_DIR=$HOME/ext/alpasim/data/drivers/vavam/mup_shapes
scripts/serve_gsplat_renderer.sh 46110 0 &
SCENES=$(python -c "import yaml; print(','.join(s['scene_id'] for s in \
  yaml.safe_load(open('configs/evaluation/splits.yaml'))['val_cl'][:4]))")
DRIVER_GPU=0 DRIVER_SOURCE=external_renderer ALPASIM_RENDERER_ADDRS='[localhost:46110]' \
  DRIVER_OVERRIDES="inference.context_length=8" \
  scripts/run_alpasim.sh runs/repro/b0b vavam "scenes.scene_ids=[$SCENES]" \
  runtime.simulation_config.n_rollouts=2 trafficsim=catk \
  runtime.simulation_config.force_gt_duration_us=4000000
python -m metadrive_ais.evaluation.summarize runs/repro/b0b=B0b
```

For the AIS driver, use `vavam_ais` with
`DRIVER_OVERRIDES="model.checkpoint_path=<bundle.pt> inference.context_length=8"`.
Bundles come from `python -m metadrive_ais.drivers.vavam_ais <sft best.pt> <out>`.
Exact commands for each experiment are in `docs/PROGRESS.md` and the run
launchers (`runs/**/launch.sh`).

## Layout

- `src/metadrive_ais/`: package (render, data, vision, memory, models, training,
  drivers, evaluation).
- `scripts/`: setup, renderer, AlpaSim runner, RL loop, reports, gallery.
- `configs/`: host driver configs, AlpaSim wizard overlays, frozen scene splits.
- `patches/alpasim/`: changes to the pinned AlpaSim checkout.
- `reports/`: one report per declared experiment; `docs/`: progress log,
  roadmap, sources, hardware.

## Hardware notes

- FlashAttention 2 has no Turing kernels: use SDPA. bf16 is emulated on Turing:
  use fp16 (Alpamayo-R1-10B: ~87 s per inference in bf16, ~6 s in fp16).
- VaVAM's action tokens have norm ~4e3; conditioning residuals need a matched
  scale (~0.2 m of trajectory change per 100 units of residual norm).

## License

Code: Apache-2.0. NuRec scenes, VaVAM and Alpamayo checkpoints are under their
providers' licenses.
