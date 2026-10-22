"""Deployable state features shared by dataset building and online inference.

Only quantities the driver legitimately observes go in: ego speed, the route
command, the committed lane offset, and the route geometry ahead (rig frame).
"""

from __future__ import annotations

import numpy as np

ROUTE_LOOKAHEAD_M = (5.0, 10.0, 20.0, 40.0)
N_COMMANDS = 4  # alpasim DriveCommand: LEFT, STRAIGHT, RIGHT, UNKNOWN
STATE_DIM = 1 + N_COMMANDS + 1 + len(ROUTE_LOOKAHEAD_M)


def route_lateral_profile(route_xy: np.ndarray | None) -> np.ndarray:
    """Route lateral offset (m, left +) at fixed distances ahead along the route."""
    if route_xy is None or len(route_xy) < 2:
        return np.zeros(len(ROUTE_LOOKAHEAD_M))
    pts = route_xy[:, :2]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    # Arc length is measured from the route point nearest the ego.
    s = s - s[int(np.argmin(np.linalg.norm(pts, axis=1)))]
    return np.interp(ROUTE_LOOKAHEAD_M, s, pts[:, 1])


def state_features(speed_mps: float, command: int, lane_offset_m: float,
                   route_xy: np.ndarray | None) -> np.ndarray:
    command_one_hot = np.zeros(N_COMMANDS)
    command_one_hot[int(command)] = 1.0
    return np.concatenate([
        [speed_mps / 10.0],
        command_one_hot,
        [lane_offset_m / 3.5],
        route_lateral_profile(route_xy) / 10.0,
    ]).astype(np.float32)
