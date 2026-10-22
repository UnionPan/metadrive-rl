"""Deterministic mode-conditioned trajectory planner.

Converts a :class:`HighLevelAction` into a timed ego trajectory in the rig
frame (x forward, y left). Lateral geometry follows the route polyline,
optionally shifted by a committed lane offset; the longitudinal profile is a
jerk- and acceleration-limited speed controller whose target depends on the
mode. A curvature speed cap keeps lateral acceleration bounded.

The planner sees only deployable inputs (ego speed/acceleration, the route,
and its own committed lane offset), never simulator ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actions import HighLevelAction, Mode


@dataclass(frozen=True)
class PlannerConfig:
    horizon_s: float = 4.0
    dt_s: float = 0.1
    lane_width_m: float = 3.5
    max_lane_offsets: int = 1
    # Distance over which a lateral offset change is blended in, as time at
    # current speed, with a floor for low speeds.
    lateral_transition_s: float = 3.0
    min_lateral_transition_m: float = 12.0
    speed_gain: float = 1.0  # 1/s, P-gain from speed error to desired accel
    max_jerk: float = 4.0
    max_lat_accel: float = 2.5
    max_speed_mps: float = 20.0
    path_step_m: float = 0.5
    # (target speed, max accel, max decel) overrides per non-speed mode.
    yield_speed_mps: float = 1.0
    yield_decel: float = 2.0
    brake_decel: float = 4.0
    cruise_accel: float = 1.5
    cruise_decel: float = 2.0
    follow_decel: float = 1.0


@dataclass
class PlannedTrajectory:
    action: HighLevelAction
    xy: np.ndarray  # (T, 2) positions at t = dt..T*dt
    heading: np.ndarray  # (T,)
    speed: np.ndarray  # (T,)
    accel: np.ndarray  # (T,)
    lane_offset_target_m: float
    dt_s: float
    flags: dict[str, bool] = field(default_factory=dict)

    @property
    def lateral_accel_max(self) -> float:
        yaw_rate = np.diff(np.unwrap(self.heading)) / self.dt_s
        return float(np.max(np.abs(yaw_rate * self.speed[1:]), initial=0.0))


class RoutePath:
    """Arc-length parameterised reference polyline in the rig frame."""

    HEAD_EXTENSION_M = 20.0
    TAIL_EXTENSION_M = 200.0

    def __init__(self, waypoints_xy: np.ndarray | None, step_m: float):
        pts = self._clean(waypoints_xy)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(int(s[-1] / step_m) + 1, 2)
        s_uniform = np.linspace(0.0, s[-1], n)
        xy = np.stack([np.interp(s_uniform, s, pts[:, i]) for i in range(2)], axis=1)
        # Light smoothing removes kinks between sparse route waypoints.
        if n >= 5:
            kernel = np.ones(5) / 5
            padded = np.pad(xy, ((2, 2), (0, 0)), mode="edge")
            xy[2:-2] = np.stack(
                [np.convolve(padded[:, i], kernel, "valid") for i in range(2)], axis=1
            )[2:-2]
        self.s = s_uniform
        self.xy = xy
        d = np.gradient(xy, s_uniform, axis=0)
        self.heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self.curvature = np.gradient(self.heading, s_uniform)

    @staticmethod
    def _clean(waypoints_xy: np.ndarray | None) -> np.ndarray:
        if waypoints_xy is None or len(waypoints_xy) < 2:
            return np.array([[-10.0, 0.0], [200.0, 0.0]])
        pts = np.asarray(waypoints_xy, dtype=float)[:, :2]
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-3])
        pts = pts[keep]
        if len(pts) < 2:
            return np.array([[-10.0, 0.0], [200.0, 0.0]])
        # Extend both ends so projections and long horizons stay on the path.
        head_dir = pts[1] - pts[0]
        tail_dir = pts[-1] - pts[-2]
        head = pts[0] - RoutePath.HEAD_EXTENSION_M * head_dir / np.linalg.norm(head_dir)
        tail = pts[-1] + RoutePath.TAIL_EXTENSION_M * tail_dir / np.linalg.norm(tail_dir)
        return np.vstack([head, pts, tail])

    def project(self, point: np.ndarray) -> tuple[float, float]:
        """Return (arc length, signed left offset) of ``point`` on the path."""
        i = int(np.argmin(np.linalg.norm(self.xy - point, axis=1)))
        tangent = np.array([np.cos(self.heading[i]), np.sin(self.heading[i])])
        rel = point - self.xy[i]
        along = float(rel @ tangent)
        lateral = float(tangent[0] * rel[1] - tangent[1] * rel[0])
        return self.s[i] + along, lateral

    def covers(self, s: float) -> bool:
        """Whether arc length ``s`` lies on the given waypoints, not an extension."""
        return self.HEAD_EXTENSION_M <= s <= self.s[-1] - self.TAIL_EXTENSION_M

    def sample(self, s: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.clip(s, self.s[0], self.s[-1])
        x = np.interp(s, self.s, self.xy[:, 0])
        y = np.interp(s, self.s, self.xy[:, 1])
        h = np.interp(s, self.s, self.heading)
        normal = np.stack([-np.sin(h), np.cos(h)], axis=1)
        return np.stack([x, y], axis=1) + normal * offset[:, None], h


def _smoothstep5(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10 - 15 * u + 6 * u**2)


class ModeConditionedPlanner:
    def __init__(self, config: PlannerConfig | None = None):
        self.cfg = config or PlannerConfig()

    @property
    def num_steps(self) -> int:
        return int(round(self.cfg.horizon_s / self.cfg.dt_s))

    def next_lane_offset(self, action: HighLevelAction, committed_offset_m: float) -> float:
        """Lane offset (relative to the route) the ego commits to after ``action``."""
        cfg = self.cfg
        limit = cfg.max_lane_offsets * cfg.lane_width_m
        if action.mode == Mode.CHANGE_LANE_LEFT:
            return min(committed_offset_m + cfg.lane_width_m, limit)
        if action.mode == Mode.CHANGE_LANE_RIGHT:
            return max(committed_offset_m - cfg.lane_width_m, -limit)
        return committed_offset_m

    def _longitudinal_limits(
        self, action: HighLevelAction, v0: float
    ) -> tuple[float, float, float]:
        cfg = self.cfg
        match action.mode:
            case Mode.CRUISE | Mode.CHANGE_LANE_LEFT | Mode.CHANGE_LANE_RIGHT:
                return action.speed_target_mps, cfg.cruise_accel, cfg.cruise_decel
            case Mode.FOLLOW:
                # Hold the current flow speed; never accelerate past it.
                return v0, 0.0, cfg.follow_decel
            case Mode.YIELD:
                return min(v0, cfg.yield_speed_mps), 0.0, cfg.yield_decel
            case Mode.BRAKE:
                return 0.0, 0.0, cfg.brake_decel
        raise ValueError(action)

    def plan(
        self,
        action: HighLevelAction,
        speed_mps: float,
        accel_mps2: float,
        route_xy: np.ndarray | None,
        committed_offset_m: float = 0.0,
        transition_elapsed_m: float = 0.0,
    ) -> PlannedTrajectory:
        """Plan ``action`` from the current ego state.

        ``transition_elapsed_m`` is the distance driven since the decision that
        set the current lane-offset target. The lateral blend resumes its
        profile from there, so replanning every cycle completes a lane change
        in one transition length instead of restarting the blend each call.
        """
        cfg = self.cfg
        v0 = float(np.clip(speed_mps, 0.0, cfg.max_speed_mps))
        a0 = float(accel_mps2)
        path = RoutePath(route_xy, cfg.path_step_m)
        s_ego, d_ego = path.project(np.zeros(2))

        d_target = self.next_lane_offset(action, committed_offset_m)
        v_target, a_max, d_max = self._longitudinal_limits(action, v0)
        flags = {"speed_capped": False, "fallback": False}

        # Curvature speed limit, propagated backwards so the profile starts
        # slowing before a bend rather than inside it.
        kappa = np.abs(path.curvature)
        v_curve = np.sqrt(cfg.max_lat_accel / np.maximum(kappa, 1e-6))
        ds = np.diff(path.s, prepend=path.s[0])
        for i in range(len(v_curve) - 2, -1, -1):
            v_curve[i] = min(v_curve[i], np.sqrt(v_curve[i + 1] ** 2 + 2 * d_max * ds[i + 1]))

        # Integrate the longitudinal profile on a fine grid.
        sub = 5
        h = cfg.dt_s / sub
        n = self.num_steps * sub
        v, a, s = v0, float(np.clip(a0, -d_max, max(a_max, 0.0))), 0.0
        vs, as_, ss = np.empty(n), np.empty(n), np.empty(n)
        for k in range(n):
            v_cap = float(np.interp(s_ego + s, path.s, v_curve))
            v_goal = min(v_target, v_cap, cfg.max_speed_mps)
            if v_cap < v_target and v_cap < v:
                flags["speed_capped"] = True
            if action.mode == Mode.BRAKE:
                a_des = -d_max if v > 0 else 0.0
            else:
                a_des = float(np.clip(cfg.speed_gain * (v_goal - v), -d_max, a_max))
            # A hard brake must not be delayed by the jerk limit.
            jerk = cfg.max_jerk if action.mode != Mode.BRAKE else 4 * cfg.max_jerk
            a += float(np.clip(a_des - a, -jerk * h, jerk * h))
            v = max(v + a * h, 0.0)
            if v == 0.0:
                a = max(a, 0.0)
            s += v * h
            vs[k], as_[k], ss[k] = v, a, s

        idx = np.arange(sub - 1, n, sub)
        speed, accel, s_rel = vs[idx], as_[idx], ss[idx]

        # Depends only on the action, so it stays fixed while a decision is held.
        transition_m = max(cfg.lateral_transition_s * v_target, cfg.min_lateral_transition_m)
        done = _smoothstep5(np.array(transition_elapsed_m / transition_m))
        if done >= 1.0 - 1e-6:
            offset = np.full_like(s_rel, d_target)
        else:
            ahead = _smoothstep5((transition_elapsed_m + s_rel) / transition_m)
            offset = d_ego + (d_target - d_ego) * (ahead - done) / (1.0 - done)
        xy, heading = path.sample(s_ego + s_rel, offset)

        if not np.all(np.isfinite(xy)):
            flags["fallback"] = True
            xy = np.zeros_like(xy)
            heading = np.zeros_like(heading)
            speed = np.zeros_like(speed)
            accel = np.zeros_like(accel)

        return PlannedTrajectory(action, xy, heading, speed, accel, d_target, cfg.dt_s, flags)

    def plan_all(
        self,
        actions: list[HighLevelAction] | tuple[HighLevelAction, ...],
        speed_mps: float,
        accel_mps2: float,
        route_xy: np.ndarray | None,
        committed_offset_m: float = 0.0,
        transition_elapsed_m: float = 0.0,
    ) -> list[PlannedTrajectory]:
        return [
            self.plan(a, speed_mps, accel_mps2, route_xy, committed_offset_m, transition_elapsed_m)
            for a in actions
        ]
