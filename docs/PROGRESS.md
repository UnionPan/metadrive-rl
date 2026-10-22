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
