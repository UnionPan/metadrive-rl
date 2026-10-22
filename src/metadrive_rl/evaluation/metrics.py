"""Per-rollout scalar metrics from AlpaSim's ``metrics.parquet`` files.

AlpaSim's evaluator writes one row per (metric, timestamp) with the metric's
own time aggregation (``max``, ``min``, ``last``, ``mean``). We reduce each
metric over the valid timestamps with that aggregation, giving one scalar per
metric per rollout, as AlpaSim's aggregation does.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

_AGG = {
    "max": pl.col("values").max(),
    "min": pl.col("values").min(),
    "mean": pl.col("values").mean(),
    "last": pl.col("values").sort_by("timestamps_us").last(),
}


def rollout_metrics(run_dir: str | Path) -> pl.DataFrame:
    """One row per rollout, one column per metric."""
    files = sorted(Path(run_dir).glob("**/rollouts/**/metrics.parquet"))
    if not files:
        raise FileNotFoundError(f"no metrics.parquet under {run_dir}")
    df = pl.concat([pl.read_parquet(f) for f in files]).filter(pl.col("valid"))
    parts = []
    for agg, expr in _AGG.items():
        part = (
            df.filter(pl.col("time_aggregation") == agg)
            .group_by("clipgt_id", "rollout_id", "name")
            .agg(expr.alias("value"))
        )
        parts.append(part)
    long = pl.concat(parts)
    unknown = set(df["time_aggregation"].unique()) - set(_AGG)
    if unknown:
        raise ValueError(f"unhandled time aggregations: {unknown}")
    return long.pivot(on="name", index=["clipgt_id", "rollout_id"], values="value").sort(
        "clipgt_id", "rollout_id"
    )
