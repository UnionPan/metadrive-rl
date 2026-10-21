# MetaDrive-RL

Mode-conditioned policy post-training (SFT → DPO → GRPO/RLVR → AIS) for closed-loop driving in
[NVIDIA AlpaSim](https://github.com/NVlabs/alpasim), on 4× Quadro RTX 8000 (Turing, FP16).

> Work in progress. See [PROJECT_GUIDELINES.md](PROJECT_GUIDELINES.md) for the plan and milestones.

## Design in one paragraph

A frozen visual backbone encodes AlpaSim camera frames; a small trainable policy picks a
high-level action (`CRUISE@v`, `FOLLOW`, `YIELD`, `BRAKE`, `CHANGE_LANE_{LEFT,RIGHT}@v`); a
deterministic planner turns that action into a jerk/acceleration/curvature-limited trajectory along
the route; AlpaSim's own controller tracks it. The policy never emits steering or throttle.

## Status

| Milestone | Status |
|---|---|
| M0 feasibility (AlpaSim on Turing, no Docker) | in progress |
| M1 mode-conditioned driver | planner done |
| M2 SFT | — |
| M3 DPO | — |
| M4 GRPO / RLVR | — |
| M5 AIS | — |
| M6 publication | — |

## Development

```bash
pip install -e ".[dev]"
pytest
```
