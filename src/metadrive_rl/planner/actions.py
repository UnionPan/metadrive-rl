"""High-level action schema shared by the policy, planner, and data logs.

The policy picks one entry of a small flat categorical action space. Keeping
the space flat (rather than factorised into mode x parameter heads) makes the
log-probability of a sampled action a single, unambiguous number, which is
what GRPO needs for its importance ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Mode(IntEnum):
    CRUISE = 0
    FOLLOW = 1
    YIELD = 2
    BRAKE = 3
    CHANGE_LANE_LEFT = 4
    CHANGE_LANE_RIGHT = 5


# Modes whose longitudinal profile is parameterised by a target-speed bin.
# FOLLOW holds the current flow speed: without a lead-vehicle observation a
# speed bin could only act as a cap, which would alias distinct actions.
SPEED_MODES = (Mode.CRUISE, Mode.CHANGE_LANE_LEFT, Mode.CHANGE_LANE_RIGHT)
SPEED_BINS_MPS = (5.0, 10.0, 15.0)


@dataclass(frozen=True)
class HighLevelAction:
    mode: Mode
    speed_target_mps: float | None = None

    def __post_init__(self) -> None:
        if (self.mode in SPEED_MODES) != (self.speed_target_mps is not None):
            raise ValueError(f"{self.mode.name} speed target mismatch: {self.speed_target_mps}")

    @property
    def index(self) -> int:
        return ACTION_INDEX[self]

    def to_text(self) -> str:
        if self.speed_target_mps is None:
            return self.mode.name
        return f"{self.mode.name}@{self.speed_target_mps:g}"

    @classmethod
    def from_text(cls, text: str) -> HighLevelAction:
        name, _, speed = text.strip().partition("@")
        return cls(Mode[name], float(speed) if speed else None)

    def to_dict(self) -> dict:
        return {"mode": self.mode.name, "speed_target_mps": self.speed_target_mps}


def _build_action_space() -> tuple[HighLevelAction, ...]:
    actions = []
    for mode in Mode:
        if mode in SPEED_MODES:
            actions.extend(HighLevelAction(mode, v) for v in SPEED_BINS_MPS)
        else:
            actions.append(HighLevelAction(mode))
    return tuple(actions)


ACTIONS: tuple[HighLevelAction, ...] = _build_action_space()
ACTION_INDEX: dict[HighLevelAction, int] = {a: i for i, a in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)
