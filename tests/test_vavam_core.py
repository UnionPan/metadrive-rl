"""metadrive_ais.models.vavam_core: AlpaSim-equivalent VaVAM preprocessing without its driver."""

from pathlib import Path

import numpy as np
import pytest

from metadrive_ais.models.vavam_core import COMMANDS, route_command

FRAMES = sorted(Path("data/episodes/m1/frames").glob("*/*.jpg"))[:6]


def test_route_command_follows_alpasims_rule():
    nan = float("nan")
    straight = np.array([[5.0 * i, 0.0] for i in range(1, 6)] + [[nan, nan]] * 15)
    left = np.array([[5.0, 0.0], [12.0, 1.0], [21.0, 4.0]] + [[nan, nan]] * 17)
    right = np.array([[22.0, -3.5]] + [[nan, nan]] * 19)
    short = np.array([[5.0, 6.0]] + [[nan, nan]] * 19)  # never reaches the 20 m lookahead
    assert route_command(straight) == COMMANDS["STRAIGHT"]
    assert route_command(left) == COMMANDS["LEFT"]
    assert route_command(right) == COMMANDS["RIGHT"]
    assert route_command(short) == COMMANDS["STRAIGHT"]


def test_tokens_match_alpasims_vam_driver():
    """First-call tokens of fresh instances (TorchScript re-profiles on its first
    call per batch size, so later calls can differ by ~0.3% of tokens)."""
    torch = pytest.importorskip("torch")
    vam_model = pytest.importorskip("alpasim_driver.models.vam_model")
    from PIL import Image

    from metadrive_ais.models.vavam_core import ASSETS, TOKENIZER, VaVAMTokenizer

    if not (torch.cuda.is_available() and TOKENIZER.exists() and len(FRAMES) == 6):
        pytest.skip("needs GPU, VaVAM assets and M1 frames")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    frames = [np.asarray(Image.open(f).convert("RGB")) for f in FRAMES]
    ours = VaVAMTokenizer("cuda")(frames)
    driver = vam_model.VAMModel(
        str(ASSETS / "VAM_width_1024_pretrained_139k.pt"),
        str(TOKENIZER),
        torch.device("cuda"),
        ["camera_front_wide_120fov"],
        context_length=8,
    )
    assert torch.equal(ours.cpu(), driver._tokenize_frames(frames).cpu())
