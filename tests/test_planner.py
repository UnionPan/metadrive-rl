import itertools

import numpy as np
import pytest

from metadrive_rl.planner import (
    ACTIONS,
    NUM_ACTIONS,
    HighLevelAction,
    Mode,
    ModeConditionedPlanner,
    PlannerConfig,
)

STRAIGHT_ROUTE = np.stack([np.linspace(0, 120, 25), np.zeros(25)], axis=1)
theta = np.linspace(0, np.pi / 2, 40)
LEFT_CURVE_ROUTE = np.stack([30 * np.sin(theta), 30 * (1 - np.cos(theta))], axis=1)


@pytest.fixture
def planner():
    return ModeConditionedPlanner()


def test_action_space_roundtrip():
    assert NUM_ACTIONS == len(set(ACTIONS)) == 12
    for i, a in enumerate(ACTIONS):
        assert a.index == i
        assert HighLevelAction.from_text(a.to_text()) == a


def test_speed_target_validation():
    with pytest.raises(ValueError):
        HighLevelAction(Mode.BRAKE, 5.0)
    with pytest.raises(ValueError):
        HighLevelAction(Mode.CRUISE)


def test_modes_produce_distinct_trajectories(planner):
    """Hard requirement from the guide: switching modes changes the plan."""
    plans = planner.plan_all(ACTIONS, speed_mps=8.0, accel_mps2=0.0, route_xy=STRAIGHT_ROUTE)
    for p, q in itertools.combinations(plans, 2):
        gap = np.max(np.linalg.norm(p.xy - q.xy, axis=1))
        assert gap > 0.3, (p.action.to_text(), q.action.to_text(), gap)


def test_longitudinal_ordering(planner):
    v0 = 8.0
    final = {
        m: planner.plan(
            HighLevelAction(m, 10.0 if m == Mode.CRUISE else None), v0, 0.0, STRAIGHT_ROUTE
        )
        for m in (Mode.CRUISE, Mode.FOLLOW, Mode.YIELD, Mode.BRAKE)
    }
    dist = {m: p.xy[-1, 0] for m, p in final.items()}
    assert dist[Mode.CRUISE] > dist[Mode.FOLLOW] > dist[Mode.YIELD] > dist[Mode.BRAKE]
    assert final[Mode.BRAKE].speed[-1] == pytest.approx(0.0, abs=1e-6)
    assert final[Mode.FOLLOW].speed.max() <= v0 + 1e-6


def test_lane_change_shifts_laterally(planner):
    left = planner.plan(HighLevelAction(Mode.CHANGE_LANE_LEFT, 10.0), 10.0, 0.0, STRAIGHT_ROUTE)
    right = planner.plan(HighLevelAction(Mode.CHANGE_LANE_RIGHT, 10.0), 10.0, 0.0, STRAIGHT_ROUTE)
    assert left.xy[-1, 1] > 2.5 and right.xy[-1, 1] < -2.5
    assert left.lane_offset_target_m == pytest.approx(3.5)
    # Lane offset is clamped to one lane each side.
    assert planner.next_lane_offset(left.action, 3.5) == pytest.approx(3.5)


def test_committed_offset_is_kept(planner):
    plan = planner.plan(HighLevelAction(Mode.CRUISE, 10.0), 10.0, 0.0, STRAIGHT_ROUTE, 3.5)
    assert plan.xy[-1, 1] == pytest.approx(3.5, abs=0.05)


def test_limits_respected(planner):
    # Start well before the bend so the speed cap has room to act.
    approach = np.vstack(
        [np.stack([np.linspace(-60, -1, 30), np.zeros(30)], axis=1), LEFT_CURVE_ROUTE + [40.0, 0.0]]
    )
    for route in (STRAIGHT_ROUTE, approach, None):
        for plan in planner.plan_all(ACTIONS, 12.0, 0.0, route):
            assert np.all(np.isfinite(plan.xy))
            assert np.all(plan.speed >= 0)
            assert np.max(np.abs(plan.accel)) <= 4.0 + 1e-6
            assert plan.lateral_accel_max < 4.0, plan.action.to_text()


def test_curvature_caps_speed(planner):
    plan = planner.plan(HighLevelAction(Mode.CRUISE, 15.0), 15.0, 0.0, LEFT_CURVE_ROUTE)
    assert plan.flags["speed_capped"]
    assert plan.speed[-1] < 15.0


def test_follows_curved_route(planner):
    plan = planner.plan(HighLevelAction(Mode.CRUISE, 5.0), 5.0, 0.0, LEFT_CURVE_ROUTE)
    assert plan.xy[-1, 1] > 1.0  # bends left with the route
    assert plan.heading[-1] > 0.2


def test_lane_change_completes_under_replanning(planner):
    """Replanning every step must not stretch the lane change out indefinitely.

    Simulates the closed loop on a straight route: each step the ego moves to
    the plan's first waypoint and replans from its new lateral offset, passing
    the distance travelled since the decision.
    """
    action = HighLevelAction(Mode.CHANGE_LANE_LEFT, 10.0)
    x, y, travelled = 0.0, 0.0, 0.0
    while x < 60.0:
        route = STRAIGHT_ROUTE - [x, y]
        plan = planner.plan(
            action, 10.0, 0.0, route, committed_offset_m=0.0, transition_elapsed_m=travelled
        )
        dx, dy = plan.xy[0]
        x, y, travelled = x + dx, y + dy, travelled + float(np.hypot(dx, dy))
    assert y == pytest.approx(3.5, abs=0.1)


def test_nonfinite_route_waypoints_are_ignored(planner):
    route = STRAIGHT_ROUTE.copy()
    route[15:] = np.nan  # AlpaSim can send routes with trailing NaN waypoints
    plan = planner.plan(HighLevelAction(Mode.CRUISE, 10.0), 10.0, 0.0, route)
    assert np.all(np.isfinite(plan.xy)) and not plan.flags["fallback"]
    assert plan.xy[-1, 0] > 30.0


@pytest.mark.parametrize("v0,gap", [(5.0, 8.0), (10.0, 20.0), (15.0, 40.0)])
def test_governor_stops_behind_stationary_lead(planner, v0, gap):
    for action in ACTIONS:
        plan = planner.plan(action, v0, 0.0, STRAIGHT_ROUTE, lead_gap_m=gap)
        travelled = np.linalg.norm(plan.xy[-1])
        assert travelled < gap, (action.to_text(), travelled)
        if action.mode != Mode.BRAKE:
            assert plan.flags["gap_capped"]


def test_governor_inactive_without_a_near_lead(planner):
    for action in ACTIONS:
        free = planner.plan(action, 12.0, 0.0, STRAIGHT_ROUTE)
        far = planner.plan(action, 12.0, 0.0, STRAIGHT_ROUTE, lead_gap_m=1000.0)
        np.testing.assert_allclose(far.xy, free.xy)
        assert not far.flags["gap_capped"]


def test_governor_follows_moving_lead_at_headway(planner):
    """Replanning every step with the true gap converges to s0 + T * v_lead."""
    cfg = planner.cfg
    v_lead, gap, v, a = 8.0, 30.0, 15.0, 0.0
    min_gap = gap
    for _ in range(300):
        plan = planner.plan(
            HighLevelAction(Mode.CRUISE, 15.0), v, a, STRAIGHT_ROUTE, lead_gap_m=gap
        )
        v_next, a = float(plan.speed[0]), float(plan.accel[0])
        gap += (v_lead - (v + v_next) / 2) * cfg.dt_s
        v = v_next
        min_gap = min(min_gap, gap)
    assert min_gap > 0.5 * cfg.standstill_gap_m
    assert abs(gap - (cfg.standstill_gap_m + cfg.time_headway_s * v_lead)) < 2.0
    assert abs(v - v_lead) < 0.5


def test_governor_ignores_leads_beyond_its_range():
    near = ModeConditionedPlanner(PlannerConfig(governor_range_m=25.0))
    cruise = HighLevelAction(Mode.CRUISE, 15.0)
    assert not near.plan(cruise, 15.0, 0.0, STRAIGHT_ROUTE, lead_gap_m=30.0).flags["gap_capped"]
    assert near.plan(cruise, 15.0, 0.0, STRAIGHT_ROUTE, lead_gap_m=20.0).flags["gap_capped"]
