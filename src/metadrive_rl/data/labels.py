"""Expert action labels from executed (logged) ego futures.

In force-GT collection runs the ego replays the human drive, so its executed
rig trajectory *is* the expert demonstration. Each decision's label is the
planner candidate whose plan stays closest to that future (average
displacement error over the plan horizon). The full ADE vector is kept so
preference pairs can be built from it later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .asl import RolloutLog


@dataclass
class LabeledDecision:
    decision_id: str
    time_us: int
    ade_m: np.ndarray  # (K,) per-candidate ADE against the executed future
    label: int  # argmin of ``ade_m``
    speed_mps: float
    command: int
    lane_offset_m: float


def expert_labels(log: RolloutLog) -> list[LabeledDecision]:
    """Label every decision whose full plan horizon lies inside the log."""
    labeled = []
    for d in log.decisions:
        if not d["is_decision"]:
            continue
        t = d["candidates_t_us"][1:]  # the first waypoint is the current pose
        if t[-1] > log.rig_t_us[-1]:
            continue
        future = log.rig_xy_at(t)
        ade = np.linalg.norm(d["candidates_xy"][:, 1:] - future[None], axis=-1).mean(-1)
        labeled.append(
            LabeledDecision(
                decision_id=d["decision_id"],
                time_us=int(d["time_us"]),
                ade_m=ade,
                label=int(np.argmin(ade)),
                speed_mps=float(d["speed_mps"]),
                command=int(d["command"]),
                lane_offset_m=float(d["lane_offset_m"]),
            )
        )
    return labeled
