import json

import numpy as np

from metadrive_ais.evaluation.summarize import bootstrap_ci, paired, rollouts


def _write(run, rows):
    f = run / "sim" / "aggregate" / "results-summary.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"rollouts": rows}))


def _rollout(scene, score, collision=0.0):
    metrics = {
        "collision_at_fault": collision,
        "offroad": 0.0,
        "progress": score,
        "dist_traveled_m": 10.0 * score,
    }
    return {"clipgt_id": scene, "score": score, "passed": collision == 0.0, "metrics": metrics}


def test_aborted_rollouts_are_excluded_and_counted(tmp_path):
    aborted = {"clipgt_id": "s2", "score": 0.0, "passed": False, "metrics": {"error": "catk"}}
    _write(tmp_path, [_rollout("s1", 0.5), aborted])
    rows, n_aborted = rollouts(tmp_path)
    assert [r["scene"] for r in rows] == ["s1"] and n_aborted == 1


def test_paired_difference_uses_matched_scenes(tmp_path):
    _write(tmp_path / "a", [_rollout("s1", 1.0), _rollout("s2", 0.5), _rollout("s3", 0.2)])
    _write(tmp_path / "b", [_rollout("s1", 0.5), _rollout("s2", 0.5)])
    text = paired(rollouts(tmp_path / "a")[0], rollouts(tmp_path / "b")[0], "a − b")
    assert "over 2 matched scenes" in text
    assert "- score: +0.25" in text and "(1 scenes differ)" in text


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci(np.array([0.0, 1.0, 0.5, 0.25]))
    assert lo < 0.4375 < hi
