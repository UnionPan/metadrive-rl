"""Per-episode sequences for the AIS memory (policy inputs and targets).

Inputs at query ``t`` use only information available then: the frame
embedding ``e_t`` and the *executed* ego motion since the previous query
``u_{t-1}`` (forward/lateral displacement and heading change in the previous
ego frame, the elapsed time, and the implied speed). The action ``a_t`` is the
trajectory the driver issued at ``t`` (rig frame). Targets are next-step
outcomes and the next embedding ``e_{t+1}``; they are never inputs.

Steps without an issued trajectory (e.g. while an 8-frame context fills) keep
their observation but are masked out of the losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

ACTION_POINTS = 6  # VaVAM horizon (waypoints after the current pose)


@dataclass
class Episode:
    episode_id: str
    scene_id: str
    embed: np.ndarray  # (T, D) e_t
    motion: np.ndarray  # (T, 5) u_{t-1}: dx, dy, dyaw, dt_s, speed (zeros at t=0)
    action: np.ndarray  # (T, 12) issued waypoints, flattened; zeros if none
    has_action: np.ndarray  # (T,) bool
    collision_next: np.ndarray  # (T,) outcome at t+1 (NaN at the last step)
    offroad_next: np.ndarray  # (T,)
    progress_gain: np.ndarray  # (T,) progress_{t+1} - progress_t


def _motion(df: pl.DataFrame) -> np.ndarray:
    x, y, yaw = (df[c].to_numpy() for c in ("ego_x", "ego_y", "ego_yaw"))
    t = df["time_us"].to_numpy() / 1e6
    out = np.zeros((len(df), 5), dtype=np.float32)
    for i in range(1, len(df)):
        c, s = np.cos(yaw[i - 1]), np.sin(yaw[i - 1])
        dx, dy = x[i] - x[i - 1], y[i] - y[i - 1]
        dt = max(t[i] - t[i - 1], 1e-3)
        dyaw = (yaw[i] - yaw[i - 1] + np.pi) % (2 * np.pi) - np.pi
        out[i] = (c * dx + s * dy, -s * dx + c * dy, dyaw, dt, np.hypot(dx, dy) / dt)
    return out


def _action(issued) -> tuple[np.ndarray, bool]:
    pts = (
        np.asarray(issued, dtype=np.float32).reshape(-1, 2)
        if issued is not None
        else np.zeros((0, 2))
    )
    if len(pts) < ACTION_POINTS + 1:
        return np.zeros(2 * ACTION_POINTS, dtype=np.float32), False
    return pts[1 : ACTION_POINTS + 1].reshape(-1), True  # drop the current pose


def _next(g: pl.DataFrame, col: str) -> np.ndarray:
    """Value at the following query (NaN after the last one)."""
    v = g[col].cast(pl.Float64).to_numpy()
    return np.append(v[1:], np.nan).astype(np.float32)


def episodes_from_table(steps: pl.DataFrame, embeddings: np.ndarray) -> list[Episode]:
    """Split an episode table (``metadrive_ais.data.episodes``) into sequences."""
    steps = steps.with_row_index("_row")
    out = []
    for (episode_id,), g in steps.sort("episode_id", "step").group_by(
        "episode_id", maintain_order=True
    ):
        rows = g["_row"].to_numpy()
        actions = [_action(a) for a in g["issued_xy"].to_list()]

        progress = g["outcome_progress"].cast(pl.Float64).to_numpy()
        out.append(
            Episode(
                episode_id=episode_id,
                scene_id=g["scene_id"][0],
                embed=embeddings[rows].astype(np.float32),
                motion=_motion(g),
                action=np.stack([a for a, _ in actions]),
                has_action=np.array([ok for _, ok in actions]),
                collision_next=_next(g, "outcome_collision_any"),
                offroad_next=_next(g, "outcome_offroad"),
                progress_gain=(np.append(progress[1:], np.nan) - progress).astype(np.float32),
            )
        )
    return out


def collate(episodes: list[Episode], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Pad episodes to a batch [B, T, ...] with a validity mask."""
    T = max(len(e.embed) for e in episodes)

    def pad(arrays, fill=0.0):
        first = arrays[0]
        out = np.full((len(arrays), T, *first.shape[1:]), fill, dtype=np.float32)
        for i, a in enumerate(arrays):
            out[i, : len(a)] = a
        return torch.from_numpy(out).to(device)

    valid = pad([np.ones(len(e.embed), dtype=np.float32) for e in episodes]).bool()
    return {
        "embed": pad([e.embed for e in episodes]),
        "motion": pad([e.motion for e in episodes]),
        "action": pad([e.action for e in episodes]),
        "has_action": pad([e.has_action.astype(np.float32) for e in episodes]).bool() & valid,
        "collision_next": pad([e.collision_next for e in episodes], np.nan),
        "offroad_next": pad([e.offroad_next for e in episodes], np.nan),
        "progress_gain": pad([e.progress_gain for e in episodes], np.nan),
        "valid": valid,
    }
