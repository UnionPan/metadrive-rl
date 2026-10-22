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
        decisions.append({"is_decision": True, "decision_id": f"d{k}", "time_us": int(t0),
                          "candidates_xy": cands, "candidates_t_us": ts, "speed_mps": rig_speed_mps,
                          "command": 1, "lane_offset_m": 0.0})
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
