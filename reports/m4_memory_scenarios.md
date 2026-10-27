# M4: memory-dependent scenario tests (guideline §6)

Declared in `docs/PROGRESS.md` before running. All cases come from actual AlpaSim
rollouts or supported simulator controls. Adapters: flow-SFT checkpoints (seed 0)
`abl_motion_s0`, `abl_frame_s0`, `abl_stack_s0`, `abl_gru16_s0` (B3, 16 epochs),
`b4_gru16_s0` (frozen AIS-trained GRU, 16 epochs).

## Coverage of the §6 list

| §6 case | test here | status |
|---|---|---|
| similar current image, different preceding motion | T2 (ego-speed-history slices), T4 (informative CATK steps) | done, from the ego's own motion history (lead-vehicle history has no supported control) |
| long- vs short-history ablation (1, 4, 8, 16 steps) | T1 (open loop, memory re-run over the last k queries), T3 (closed loop, `VAVAM_AIS_MEMORY_RESET`) | done |
| state reset / rollout mixing | `tests/test_vavam_ais_driver.py` (session isolation, memory reset) | done |
| different traffic reactivity in comparable scenes | — | not built: AlpaSim offers CATK or log replay, not a controlled reactivity knob |
| transient occlusion, stop-and-go, interrupted merges | stop-and-go only as a T2 slice (3 windows) | occlusion and merges not built: they need scene editing or event labels that val/train scenes do not provide |

## T1: history length (open loop, held-out flow loss / ADE m)

| adapter | k = 1 | k = 2 | k = 4 | k = 8 | k = 16 | k = all |
|---|---|---|---|---|---|---|
| gru16 | 0.0559 / 2.79 | 0.0558 / 2.82 | 0.0567 / 2.82 | 0.0562 / 2.80 | 0.0560 / 2.80 | 0.0556 / 2.79 |
| b4_gru16 | 0.0630 / 2.83 | 0.0608 / 2.78 | 0.0604 / 2.77 | 0.0604 / 2.77 | 0.0604 / 2.77 | 0.0604 / 2.77 |

Frame adapter (no history) for reference: 0.038 / 2.36.

## T2: memory-relevant slices (held-out windows, flow loss / ADE m)

Slices from the ego's speed over the last 2 s: stop-and-go (stopped, now > 3 m/s), decelerating / accelerating (> 2 m/s change), steady.

| slice | windows | stock | motion | frame | stack | gru16 | b4_gru16 | GRU − frame flow [95% CI] | GRU − frame ADE |
|---|---|---|---|---|---|---|---|---|---|
| stop-and-go | 3 | 0.0339 | 0.0465 / 4.29 | 0.0271 / 3.19 | 0.0270 / 3.12 | 0.0313 / 3.49 | 0.0232 / 2.80 | +0.0042 (n too small) | +0.30 |
| decelerating | 20 | 0.0267 | 0.0243 / 1.56 | 0.0230 / 1.53 | 0.0226 / 1.47 | 0.0269 / 1.98 | 0.0258 / 1.71 | +0.0038 [-0.0017, +0.0067] | +0.45 [-0.17, +0.76] |
| accelerating | 12 | 0.0509 | 0.0526 / 3.56 | 0.0395 / 3.10 | 0.0394 / 3.12 | 0.0436 / 3.46 | 0.0383 / 3.30 | +0.0041 [-0.0018, +0.0070] | +0.35 [+0.09, +0.49] |
| steady | 277 | 0.0908 | 0.0427 / 2.43 | 0.0398 / 2.38 | 0.0400 / 2.25 | 0.0598 / 2.82 | 0.0631 / 2.83 | +0.0200 [+0.0029, +0.0438] | +0.44 [-0.11, +1.13] |
| all | 312 | 0.0847 | 0.0419 / 2.43 | 0.0386 / 2.36 | 0.0387 / 2.24 | 0.0568 / 2.79 | 0.0594 / 2.77 | +0.0182 [+0.0026, +0.0436] | +0.44 [-0.10, +1.14] |

## T3: history in closed loop (val_cl, M1 protocol; GRU-16 adapter)

| run | scenes | rollouts | aborted | score | pass rate | at-fault collision | offroad | progress | distance (m) |
|---|---|---|---|---|---|---|---|---|---|
| never reset | 16 | 32 | 0 | 0.156 [5.26e-05, 0.344] | 0.188 [0.0312, 0.375] | 0.188 [0.0312, 0.375] | 0.656 [0.438, 0.844] | 0.466 [0.338, 0.601] | 79.3 [61.2, 96.8] |
| reset every 4 (2 s) | 16 | 32 | 0 | 0.262 [0.112, 0.435] | 0.406 [0.219, 0.625] | 0.0625 [0, 0.188] | 0.562 [0.375, 0.75] | 0.523 [0.381, 0.666] | 88.4 [69.5, 106] |
| reset every query | 16 | 32 | 0 | 0.249 [0.0627, 0.451] | 0.344 [0.125, 0.562] | 0.125 [0, 0.312] | 0.625 [0.406, 0.844] | 0.501 [0.363, 0.644] | 86.4 [65.7, 107] |

Paired never reset − reset every query over 16 matched scenes:
- score: -0.093 [-0.262, +0.0553] (5 scenes differ)
- pass rate: -0.156 [-0.344, +0.0312] (5 scenes differ)
- at-fault collision: +0.0625 [-0.0625, +0.219] (3 scenes differ)
- offroad: +0.0312 [-0.125, +0.219] (4 scenes differ)
- progress: -0.0352 [-0.106, +0.0219] (15 scenes differ)
- distance (m): -7.05 [-18, +2.72] (16 scenes differ)
Paired never reset − reset every 4 (2 s) over 16 matched scenes:
- score: -0.105 [-0.199, -0.0261] (6 scenes differ)
- pass rate: -0.219 [-0.375, -0.0625] (6 scenes differ)
- at-fault collision: +0.125 [+0, +0.281] (3 scenes differ)
- offroad: +0.0938 [-0.0625, +0.25] (4 scenes differ)
- progress: -0.0573 [-0.125, -0.00649] (15 scenes differ)
- distance (m): -9.1 [-19.4, -1.62] (16 scenes differ)

## T4: AIS heads on informative CATK steps (speed change > 3 m/s over 2 s)

Held-out: 108 informative steps (collision positives 2, offroad 8), 719 other steps.

| memory | subset | collision AUROC | offroad AUROC | next-embedding NLL |
|---|---|---|---|---|
| frame | informative | 0.34 ± 0.05 | 0.83 ± 0.00 | -0.136 ± 0.003 |
| frame | rest | 0.74 ± 0.11 | 0.72 ± 0.07 | -0.163 ± 0.003 |
| stack | informative | 0.33 ± 0.07 | 0.71 ± 0.03 | -0.121 ± 0.005 |
| stack | rest | 0.58 ± 0.10 | 0.72 ± 0.07 | -0.140 ± 0.006 |
| gru | informative | 0.39 ± 0.03 | 0.79 ± 0.02 | -0.118 ± 0.015 |
| gru | rest | 0.73 ± 0.08 | 0.69 ± 0.07 | -0.154 ± 0.002 |

## Reading

- **No memory benefit where memory should matter.** On windows whose speed
  history changed, the GRU adapter is not better than the frame adapter (flow
  +0.004, CI includes 0; ADE worse when accelerating). B4 uses about 1 s of
  history (k = 1 → 2 helps) but stays far behind frame.
- **The GRU adapter ignores its history open loop** (flat in k), and **in
  closed loop long history hurts**: never resetting vs resetting every 2 s
  scores −0.105 [−0.199, −0.026] (secondary comparison; the declared primary,
  never vs every query, is −0.093 [−0.262, +0.055]).
- **The AIS heads gain nothing from memory on informative steps**: frame ≥ GRU
  on offroad AUROC (0.83 vs 0.79) and next-embedding NLL. Collision AUROC
  rests on 2 positives and is not interpretable.
- **Power is low.** Only 35 of 312 held-out windows and 108 CATK steps are
  memory-relevant, and val/train scenes are short (~16 s of control).
  Occlusion, merge and reactivity cases could not be built from supported
  controls. The negative result holds for the situations these scenes contain.
