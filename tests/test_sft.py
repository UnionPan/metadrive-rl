import json
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from metadrive_rl.planner import NUM_ACTIONS
from metadrive_rl.policy.features import STATE_DIM, route_lateral_profile, state_features
from metadrive_rl.policy.neural import load_neural_policy


def test_route_profile_reads_lateral_offsets_ahead():
    xs = np.linspace(-5, 60, 66)
    route = np.stack([xs, 0.05 * xs], 1)  # gentle left drift
    profile = route_lateral_profile(route)
    assert profile == pytest.approx(0.05 * np.array([5, 10, 20, 40]) / np.sqrt(1 + 0.05**2), rel=1e-2)
    assert route_lateral_profile(None).tolist() == [0.0] * 4
    assert state_features(5.0, 1, 0.0, route).shape == (STATE_DIM,)


def _write_split(data, split, n, rng, feature_dim=16):
    # The best action is decided by the sign of image feature 0.
    feats = rng.normal(size=(n, feature_dim)).astype(np.float32)
    ade = np.full((n, NUM_ACTIONS), 3.0)
    best = np.where(feats[:, 0] > 0, 2, 7)
    ade[np.arange(n), best] = 0.1
    pl.DataFrame({
        "decision_id": [f"d{i}" for i in range(n)],
        "image": ["unused"] * n,
        "state": [state_features(5.0, 1, 0.0, None).tolist()] * n,
        "ade_m": ade.tolist(),
        "label": best,
    }).write_parquet(data / f"{split}.parquet")
    np.save(data / f"{split}.fake.npy", feats)


def test_sft_learns_feature_dependent_action(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("train", 256), ("val", 64)):
        _write_split(data, split, n, rng)
    out = tmp_path / "run"
    subprocess.run([sys.executable, "-m", "metadrive_rl.training.sft", "--data", str(data),
                    "--encoder", "fake", "--out", str(out), "--epochs", "60"], check=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["best"]["val"]["top1"] > 0.9
    assert summary["best"]["val"]["regret_mean_m"] < 0.3
    policy = load_neural_policy(str(out / "best.pt"))
    assert policy.encoder_name == "fake"


def test_dpo_pairs_prefer_lowest_ade():
    import torch

    from metadrive_rl.training.dpo import preference_pairs

    ade = torch.tensor([[0.2, 0.5, 3.0, 5.0] + [9.0] * (NUM_ACTIONS - 4)])
    pairs = preference_pairs(ade, margin=1.0, max_per_decision=2)
    assert pairs.tolist() == [[0, 0, 2], [0, 0, 3]]  # hardest qualifying negatives first


def test_dpo_runs_from_sft_checkpoint(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("train", 256), ("val", 64)):
        _write_split(data, split, n, rng)
    subprocess.run([sys.executable, "-m", "metadrive_rl.training.sft", "--data", str(data),
                    "--encoder", "fake", "--out", str(tmp_path / "sft"), "--epochs", "20"], check=True)
    subprocess.run([sys.executable, "-m", "metadrive_rl.training.dpo", "--data", str(data),
                    "--sft", str(tmp_path / "sft" / "best.pt"), "--out", str(tmp_path / "dpo"),
                    "--epochs", "10"], check=True)
    summary = json.loads((tmp_path / "dpo" / "summary.json").read_text())
    assert summary["pairs"] > 0
    assert summary["best"]["val"]["regret_mean_m"] <= summary["sft_val"]["regret_mean_m"]
    assert load_neural_policy(str(tmp_path / "dpo" / "best.pt")).encoder_name == "fake"


def test_route_profile_ignores_nonfinite_waypoints():
    # Pattern seen in AlpaSim routes: 20 waypoints ~4.2 m apart, last 5 NaN.
    xs = np.arange(20) * 4.2
    route = np.stack([xs, 0.02 * xs], 1)
    route[15:] = np.nan
    profile = route_lateral_profile(route)
    assert np.all(np.isfinite(profile))
    assert profile == pytest.approx(0.02 * np.array([5, 10, 20, 40]), rel=1e-2)
