from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("alpasim_grpc")

from metadrive_rl.data.asl import RolloutLog  # noqa: E402
from metadrive_rl.data.labels import expert_labels  # noqa: E402


def _log(rig_speed_mps: float, duration_s: float = 10.0) -> RolloutLog:
    t = np.arange(0, int(duration_s * 1e6) + 1, 100_000, dtype=np.int64)
    rig = np.stack([rig_speed_mps * t / 1e6, np.zeros(len(t))], 1)
    horizon = np.arange(0, 41) * 100_000
    decisions = []
    for k, t0 in enumerate(t[::5]):
        ts = t0 + horizon
        x0 = rig_speed_mps * t0 / 1e6
        # Candidates: constant speeds 0, 5, 10 m/s from the current position.
        cands = np.stack([np.stack([x0 + v * horizon / 1e6, np.zeros(41)], 1) for v in (0, 5, 10)])
        decisions.append(
            {
                "is_decision": True,
                "decision_id": f"d{k}",
                "time_us": int(t0),
                "candidates_xy": cands,
                "candidates_t_us": ts,
                "speed_mps": rig_speed_mps,
                "command": 1,
                "lane_offset_m": 0.0,
            }
        )
    return RolloutLog(Path("x"), "scene", "uuid", t, rig, np.zeros(len(t)), t, rig, decisions)


def test_label_is_the_candidate_matching_the_logged_future():
    labeled = expert_labels(_log(rig_speed_mps=5.0))
    assert labeled and all(d.label == 1 for d in labeled)
    assert all(d.ade_m[1] == pytest.approx(0.0, abs=1e-9) for d in labeled)


def test_decisions_without_a_full_future_are_dropped():
    log = _log(rig_speed_mps=5.0, duration_s=10.0)
    labeled = expert_labels(log)
    # Plans span 4 s, so decisions after t = 6 s have no complete future.
    assert max(d.time_us for d in labeled) <= 6_000_000
    assert len(labeled) == 13


def test_gap_at_uses_latest_frame_not_after_query():
    from metadrive_rl.data.lead_gap import gap_at

    ts = np.array([0, 100_000, 200_000])
    gaps = np.array([30.0, 20.0, 10.0])
    assert gap_at(ts, gaps, 150_000) == 20.0
    assert gap_at(ts, gaps, 200_000) == 10.0
    assert gap_at(ts, gaps, -5) == 30.0


def test_gap_metrics_separate_misses_from_false_alarms():
    from metadrive_rl.evaluation.gap_error import gap_metrics

    true = np.array([5.0, 5.0, 50.0, 50.0])
    pred = np.array([6.0, 40.0, 50.0, 8.0])
    m = gap_metrics(pred, true)
    assert m["recall_lead_within_10m"] == 0.5
    assert m["dangerous_miss_rate"] == 0.5
    assert m["false_alarm_rate"] == 0.5


def test_visible_tracks_keep_only_vehicles_the_renderer_draws():
    from metadrive_rl.data.lead_gap import visible_tracks

    still = [[0.0, 0.0, 0.0, 0, 0, 0, 1]] * 2
    moving = [[0.0, 0.0, 0.0, 0, 0, 0, 1], [30.0, 0.0, 0.0, 0, 0, 0, 1]]
    tracks = {
        "tracks_id": ["car", "parked", "ghost", "walker"],
        "tracks_label_class": ["automobile", "automobile", "automobile", "person"],
        "tracks_timestamps_us": [[0, 10], [0, 10], [5, 20], [0, 10]],
        "tracks_poses": [moving, still, moving, moving],
    }
    # "car" has dynamic Gaussians over its span; "ghost" moves but has none.
    assert visible_tracks(tracks, {(0, 10)}) == {"car", "parked"}


def test_path_gap_follows_the_driven_curve():
    from metadrive_rl.data.lead_gap import NO_LEAD_M, future_path, path_gap

    # Ego drives a left quarter circle of radius 30 m.
    th = np.linspace(0, np.pi / 2, 60)
    path = future_path(np.stack([30 * np.sin(th), 30 * (1 - np.cos(th))], 1), np.pi / 2)
    on_curve = ("car", 30 * np.sin(0.6), 30 * (1 - np.cos(0.6)))  # 18 m along the arc
    straight_ahead = ("parked", 18.0, 0.0)  # in a straight corridor, off the curve
    half = {"car": 2.0, "parked": 2.0}
    assert abs(path_gap(path, [on_curve], half) - (30 * 0.6 - 2.6 - 2.0)) < 0.6
    assert path_gap(path, [straight_ahead], half) == NO_LEAD_M
