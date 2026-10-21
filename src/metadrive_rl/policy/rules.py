"""Non-learned policies: fixed-action probes and the Stage A rule-based baseline."""

from __future__ import annotations

import numpy as np

from metadrive_rl.planner import HighLevelAction, Mode

from .base import Observation, one_hot

# alpasim_driver DriveCommand values; duplicated so this module has no
# simulator dependency.
CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT = 0, 1, 2


class FixedPolicy:
    """Always selects one action. Used to probe mode controllability (M1)."""

    def __init__(self, action: HighLevelAction):
        self.action = action
        self.version = f"fixed:{action.to_text()}"

    def distribution(self, obs: Observation) -> np.ndarray:
        return one_hot(self.action.index)


class RouteRulePolicy:
    """Perception-free baseline: lane keeping with a route-geometry speed choice.

    Slows to the low speed bin for turns, otherwise cruises at the middle bin.
    It is blind to other agents by construction, which makes it the reference
    point for what perception (and post-training) must add.
    """

    version = "rule:route"

    def __init__(self, turn_curvature: float = 0.02, lookahead_m: float = 30.0):
        self.turn_curvature = turn_curvature
        self.lookahead_m = lookahead_m

    def _max_curvature_ahead(self, route_xy: np.ndarray | None) -> float:
        if route_xy is None or len(route_xy) < 3:
            return 0.0
        ahead = route_xy[(route_xy[:, 0] > 0) & (np.hypot(*route_xy.T) < self.lookahead_m)]
        if len(ahead) < 3:
            return 0.0
        d = np.diff(ahead, axis=0)
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        seg = np.maximum(np.linalg.norm(d, axis=1)[1:], 1e-3)
        return float(np.max(np.abs(np.diff(heading)) / seg))

    def distribution(self, obs: Observation) -> np.ndarray:
        turning = obs.command in (CMD_LEFT, CMD_RIGHT) or (
            self._max_curvature_ahead(obs.route_xy) > self.turn_curvature
        )
        speed = 5.0 if turning else 10.0
        return one_hot(HighLevelAction(Mode.CRUISE, speed).index)
