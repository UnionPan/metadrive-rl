import polars as pl
import pytest

pytest.importorskip("alpasim_grpc")

from metadrive_rl.data.branch_prefs import _label_to_action, preference_pairs  # noqa: E402
from metadrive_rl.planner import HighLevelAction  # noqa: E402

IDX = {a: HighLevelAction.from_text(a).index for a in ("CRUISE@10", "FOLLOW", "BRAKE", "YIELD")}


def test_labels_round_trip():
    assert _label_to_action("CHANGE_LANE_LEFT_10") == "CHANGE_LANE_LEFT@10"
    assert _label_to_action("FOLLOW") == "FOLLOW"


def test_safety_then_progress():
    out = pl.DataFrame([
        {"scene_id": "s", "action": "CRUISE@10", "passed": False, "progress": 0.9},
        {"scene_id": "s", "action": "FOLLOW", "passed": True, "progress": 0.5},
        {"scene_id": "s", "action": "YIELD", "passed": True, "progress": 0.49},  # within margin
        {"scene_id": "s", "action": "BRAKE", "passed": True, "progress": 0.1},
    ])
    pairs = {(r["winner"], r["loser"], r["reason"]) for r in preference_pairs(out, 0.02).to_dicts()}
    assert (IDX["FOLLOW"], IDX["CRUISE@10"], "safety") in pairs  # fast but at fault loses
    assert (IDX["FOLLOW"], IDX["BRAKE"], "progress") in pairs  # stopping is not "safe enough"
    assert not any({w, lo} == {IDX["FOLLOW"], IDX["YIELD"]} for w, lo, _ in pairs)
    assert not any(lo == IDX["FOLLOW"] and w == IDX["CRUISE@10"] for w, lo, _ in pairs)
