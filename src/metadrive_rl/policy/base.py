"""Policy interface: deployable observation in, high-level action distribution out."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from metadrive_rl.planner import ACTIONS, NUM_ACTIONS, HighLevelAction


@dataclass
class Observation:
    """Everything a deployable policy may condition on at one decision step.

    No simulator-privileged state (other agents' poses, recorded ground truth)
    is allowed here; that keeps the observation channel leak-free by type.
    """

    timestamp_us: int
    speed_mps: float
    accel_mps2: float
    route_xy: np.ndarray | None  # (N, 2) rig frame
    command: int  # alpasim DriveCommand value (LEFT/STRAIGHT/RIGHT/UNKNOWN)
    lane_offset_m: float  # committed lane offset from the planner state
    image: np.ndarray | None = None  # HWC uint8, front camera
    features: np.ndarray | None = None  # frozen-encoder embedding of ``image``
    lead_gap_m: float | None = None  # gap to the lead car estimated from ``image``


@dataclass
class Decision:
    action_index: int
    log_prob: float
    probs: np.ndarray  # (NUM_ACTIONS,)

    @property
    def action(self) -> HighLevelAction:
        return ACTIONS[self.action_index]


class Policy(Protocol):
    version: str

    def distribution(self, obs: Observation) -> np.ndarray:
        """Return action probabilities, shape (NUM_ACTIONS,)."""
        ...


def decide(policy: Policy, obs: Observation, rng: np.random.Generator, greedy: bool) -> Decision:
    probs = np.asarray(policy.distribution(obs), dtype=np.float64)
    if probs.shape != (NUM_ACTIONS,) or not np.isclose(probs.sum(), 1.0):
        raise ValueError(f"{policy.version}: invalid action distribution {probs}")
    index = int(np.argmax(probs)) if greedy else int(rng.choice(NUM_ACTIONS, p=probs))
    return Decision(index, float(np.log(max(probs[index], 1e-12))), probs)


def one_hot(index: int, eps: float = 0.0) -> np.ndarray:
    probs = np.full(NUM_ACTIONS, eps / NUM_ACTIONS)
    probs[index] += 1.0 - eps
    return probs
