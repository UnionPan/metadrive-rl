import polars as pl

from metadrive_rl.evaluation.metrics import coverage_exit_us


def _write(tmp_path, values):
    n = len(values)
    pl.DataFrame(
        {
            "name": ["dist_to_gt_trajectory"] * n,
            "timestamps_us": [k * 100_000 for k in range(n)],
            "values": values,
            "valid": [True] * n,
        }
    ).write_parquet(tmp_path / "metrics.parquet")


def test_coverage_exit_is_first_timestamp_beyond_threshold(tmp_path):
    _write(tmp_path, [0.1, 0.5, 3.9, 4.0, 2.0, 9.0])
    assert coverage_exit_us(tmp_path) == 300_000  # leaving once counts, even if it returns


def test_no_exit_while_within_corridor(tmp_path):
    _write(tmp_path, [0.1, 3.5, 3.9])
    assert coverage_exit_us(tmp_path) is None
