import numpy as np
import polars as pl
import torch

from metadrive_rl.planner import NUM_ACTIONS
from metadrive_rl.policy.features import STATE_DIM
from metadrive_rl.policy.neural import PolicyHead, save_checkpoint
from metadrive_rl.training.grpo import load_head, update


def test_update_moves_probability_toward_positive_advantage(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, dim = 400, 16
    head = PolicyHead(dim)
    save_checkpoint(tmp_path / "init.pt", head, "fake", "init", {})
    feats = rng.normal(size=(n, dim)).astype(np.float32)
    state = np.zeros((n, STATE_DIM), np.float32)
    with torch.no_grad():
        logp = torch.log_softmax(head(torch.tensor(feats), torch.tensor(state)), -1).numpy()
    actions = rng.integers(0, NUM_ACTIONS, n)
    advantage = np.where(actions == 3, 1.0, -0.2)  # action 3 led to better outcomes
    batch = tmp_path / "batch"
    batch.mkdir()
    pl.DataFrame({"rollout_uuid": ["r"] * n, "time_us": np.arange(n),
                  "action": actions, "behaviour_logp": logp[np.arange(n), actions],
                  "advantage": advantage, "state": state.tolist()}).write_parquet(
        batch / "decisions.parquet")
    np.save(batch / "features.npy", feats)

    result = update(batch, tmp_path / "init.pt", tmp_path / "init.pt", tmp_path / "new.pt",
                    clip=0.2, kl_coef=0.01, epochs=10, lr=1e-3, batch_size=128, version="it1")
    new, ckpt = load_head(tmp_path / "new.pt", torch.device("cpu"))
    with torch.no_grad():
        p_old = np.exp(logp)[:, 3].mean()
        p_new = torch.softmax(new.cpu()(torch.tensor(feats), torch.tensor(state)), -1)[:, 3].mean()
    assert float(p_new) > p_old + 0.05
    assert 0.0 < result["final"]["kl_to_ref"] < 1.0
    assert ckpt["version"] == "it1"
