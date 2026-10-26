import math

import numpy as np
import polars as pl
import pytest

pytest.importorskip("alpasim_driver")
from metadrive_ais.training.flow_sft import HORIZON, human_futures  # noqa: E402


def test_human_futures_in_the_rig_frame():
    n = 10
    yaw = math.pi / 2  # driving along +y
    g = pl.DataFrame(
        {"ego_x": [0.0] * n, "ego_y": [2.0 * i for i in range(n)], "ego_yaw": [yaw] * n}
    )
    fut, ok = human_futures(g)
    assert ok.tolist() == [True] * (n - HORIZON) + [False] * HORIZON
    # Straight ahead in the rig frame: x = 2, 4, ..., 12 m, y = 0.
    np.testing.assert_allclose(fut[0], np.stack([2.0 * np.arange(1, 7), np.zeros(6)], 1), atol=1e-5)
