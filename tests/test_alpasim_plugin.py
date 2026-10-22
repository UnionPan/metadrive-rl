import json

import numpy as np
import pytest

base = pytest.importorskip("alpasim_driver.models.base")

from metadrive_rl.alpasim.model import MetaDriveModel  # noqa: E402
from metadrive_rl.policy.spec import PolicySpec  # noqa: E402


class _Vec:
    def __init__(self, x, y):
        self.x, self.y, self.z = x, y, 0.0


class _Route:
    def __init__(self, pts):
        self.waypoints = [_Vec(x, y) for x, y in pts]


def _inputs(poses, seed, t_us, speed=8.0):
    # Like the servicer, hand over a fresh list holding the same pose objects.
    poses = sorted(poses, key=lambda p: p[0])
    # The servicer passes plain tuples, not CameraFrame instances.
    frame = (t_us, np.zeros((64, 96, 3), np.uint8))
    return base.PredictionInput(
        camera_images={"cam": [frame]},
        command=base.DriveCommand.STRAIGHT,
        speed=speed,
        acceleration=0.0,
        ego_pose_history=poses,
        inference_seed=seed,
        previous_plan=None,
        route=_Route([(x, 0.0) for x in range(0, 100, 5)]),
    )


def _model(ref, **kw):
    spec = PolicySpec.parse(ref)
    for k, v in kw.items():
        setattr(spec, k, v)
    return MetaDriveModel(spec, ["cam"], 10, device=None)


def _record(pred):
    return json.loads(pred.reasoning_text)["metadrive"]


def test_fixed_policy_selects_its_candidate():
    model = _model("fixed:YIELD")
    pred = model.predict(_inputs([(0, object())], 100, 0))
    rec = _record(pred)
    assert rec["action"] == "YIELD" and rec["is_decision"]
    assert pred.candidate_positions.shape == (12, 40, 3)
    assert pred.selected_index == rec["action_index"]
    assert rec["log_prob"] == pytest.approx(0.0)


def test_decisions_are_held_per_session():
    model = _model("fixed:CHANGE_LANE_LEFT@10", decision_period_s=0.5)
    a, b = [(0, object())], [(0, object())]
    recs_a = [_record(model.predict(_inputs(a, 10 + k, k * 100_000))) for k in range(6)]
    rec_b = _record(model.predict(_inputs(b, 500, 0)))
    assert [r["is_decision"] for r in recs_a] == [True, False, False, False, False, True]
    assert rec_b["is_decision"] and rec_b["step"] == 0
    # A held lane change commits exactly one lane, not one per replanning call.
    assert recs_a[4]["lane_offset_m"] == pytest.approx(3.5)


def test_session_reset_on_seed_discontinuity():
    model = _model("rule")
    poses = [(0, object())]
    model.predict(_inputs(poses, 10, 0))
    rec = _record(model.predict(_inputs(poses, 999, 100_000)))
    assert rec["step"] == 0


def test_lane_change_progress_survives_repeated_decisions():
    """Re-deciding the same lane change must not restart the lateral blend."""
    model = _model("fixed:CHANGE_LANE_LEFT@10", decision_period_s=0.5)
    poses = [(0, object())]
    recs = [_record(model.predict(_inputs(poses, 10 + k, k * 100_000, speed=8.0)))
            for k in range(12)]
    assert recs[5]["is_decision"] and recs[10]["is_decision"]
    assert recs[11]["lateral_elapsed_m"] == pytest.approx(8.0 * 1.1)
    assert all(r["lane_offset_m"] == pytest.approx(3.5) for r in recs)


def _shifted_route_inputs(poses, seed, t_us, lateral_shift):
    p = _inputs(poses, seed, t_us)
    p.route.waypoints = [_Vec(w.x, w.y + lateral_shift) for w in p.route.waypoints]
    return p


def test_committed_lane_survives_route_reanchoring():
    """AlpaSim's MAP route can re-anchor to the ego's current lane mid-episode.

    After a left lane change the ego sits ~3.5 m left of the route; when the
    route jumps under the ego, the committed offset must follow so the ego
    keeps its absolute lane instead of changing lanes again.
    """
    model = _model("fixed:CHANGE_LANE_LEFT@10", decision_period_s=0.5)
    poses = [(0, object())]
    # Ego already moved left: the route lies 3.5 m to the ego's right.
    for k in range(3):
        rec = _record(model.predict(_shifted_route_inputs(poses, 10 + k, k * 100_000, -3.5)))
    assert rec["lane_offset_m"] == pytest.approx(3.5)
    # Route re-anchors onto the ego's lane.
    rec = _record(model.predict(_shifted_route_inputs(poses, 13, 300_000, 0.0)))
    assert rec["lane_offset_m"] == pytest.approx(0.0, abs=0.05)


def test_decision_images_are_saved(tmp_path):
    model = _model("rule", image_dir=str(tmp_path), decision_period_s=0.5)
    poses = [(0, object())]
    recs = [_record(model.predict(_inputs(poses, 10 + k, k * 100_000))) for k in range(6)]
    saved = sorted(p.stem for p in tmp_path.glob("*.jpg"))
    assert saved == sorted({r["decision_id"] for r in recs})
    assert len(saved) == 2
