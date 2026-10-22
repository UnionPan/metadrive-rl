"""Runtime policy specification, loaded from YAML or a short inline form."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from metadrive_rl.planner import HighLevelAction

from .base import Policy
from .rules import FixedPolicy, RouteRulePolicy


@dataclass
class PolicySpec:
    policy: dict[str, Any]
    encoder: dict[str, Any] | None = None
    # A decision is held for this long; the plan is still recomputed at every
    # driver call so the held action stays closed-loop.
    decision_period_s: float = 0.5
    greedy: bool = True
    # When set, per-decision encoder features are written here for training.
    feature_dir: str | None = None
    # When set, the camera image seen at each decision is saved here as
    # <decision_id>.jpg, so datasets can be re-encoded with other backbones.
    image_dir: str | None = None
    # Save the image at every driver call (named <decision_id>_<step>.jpg)
    # instead of only at decisions; for dense perception datasets.
    image_every_step: bool = False
    # Gap-head checkpoint (metadrive_rl.perception.lead_gap). When set, the
    # lead gap is estimated from the camera at every driver call and drives
    # the planner's speed governor for all candidates.
    gap_head: str | None = None
    # The governor uses the minimum of the last this-many estimates: a missed
    # lead must then persist across consecutive frames to go unnoticed.
    gap_filter_steps: int = 1
    planner: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, ref: str) -> PolicySpec:
        """``ref`` is a YAML path, ``rule``, or ``fixed:<ACTION>`` (e.g. fixed:YIELD)."""
        if ref == "rule":
            return cls(policy={"type": "rule"})
        if ref.startswith("fixed:"):
            return cls(policy={"type": "fixed", "action": ref.removeprefix("fixed:")})
        return cls(**yaml.safe_load(Path(ref).read_text()))


def build_policy(cfg: dict[str, Any]) -> Policy:
    match cfg["type"]:
        case "fixed":
            return FixedPolicy(HighLevelAction.from_text(cfg["action"]))
        case "rule":
            return RouteRulePolicy()
        case "neural":
            from .neural import load_neural_policy

            return load_neural_policy(cfg["checkpoint"], cfg.get("temperature", 1.0))
    raise ValueError(f"unknown policy type {cfg['type']!r}")
