"""AlpaSim driver plugin: high-level policy + mode-conditioned planner.

Registered under the ``alpasim.models`` entry point as ``metadrive`` so the
stock AlpaSim driver server handles gRPC, image decoding, and route updates.
``model.checkpoint_path`` in the driver config names a policy spec (see
:class:`metadrive_rl.policy.spec.PolicySpec`).

Every drive response carries all planner candidates (``sampled_trajectories``
in the ASL log) and a JSON decision record in ``reasoning_text``, so a rollout
log alone is enough to reconstruct which action was sampled, with which
log-probability, under which policy version.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)

from metadrive_rl.planner import ACTIONS, ModeConditionedPlanner, PlannerConfig
from metadrive_rl.planner.trajectory import RoutePath
from metadrive_rl.policy.base import Decision, Observation, decide
from metadrive_rl.policy.spec import PolicySpec, build_policy

logger = logging.getLogger(__name__)

RECORD_KEY = "metadrive"
REANCHOR_JUMP_M = 1.5


@dataclass
class _SessionState:
    next_inference_seed: int
    rng: np.random.Generator
    # Committed lane offset after the current decision, and the offset it was
    # planned from; held lane changes replan from the latter so they do not stack.
    lane_offset_m: float = 0.0
    base_offset_m: float = 0.0
    decision: Decision | None = None
    decision_id: str | None = None
    decision_time_us: int = 0
    # Distance driven since the lane-offset target last changed, integrated
    # from ego speed; the planner resumes its lateral blend from here.
    travelled_since_target_m: float = 0.0
    last_obs_time_us: int | None = None
    # Ego's lateral offset from the route at the previous call.
    route_offset_m: float | None = None
    step: int = 0
    history: list = field(default_factory=list)


class _SessionTable:
    """Per-session state keyed by the session's first ego pose object.

    ``PredictionInput`` carries no session id. The servicer rebuilds its pose
    list (``sorted``) on every egomotion update, so the list itself is not a
    stable key, but the pose objects in it persist and the oldest one stays
    first. ``inference_seed`` increases by one per call within a session; the
    seed check guards against object-id reuse after a session ends.
    """

    def __init__(self, max_sessions: int = 512):
        self._states: dict[int, _SessionState] = {}
        self._max = max_sessions

    def get(self, pred_input: PredictionInput) -> _SessionState:
        key = id(pred_input.ego_pose_history[0])
        state = self._states.get(key)
        if state is None or state.next_inference_seed != pred_input.inference_seed:
            if len(self._states) >= self._max:
                self._states.pop(next(iter(self._states)))
            state = _SessionState(
                next_inference_seed=pred_input.inference_seed,
                rng=np.random.default_rng(pred_input.inference_seed),
            )
            self._states[key] = state
        state.next_inference_seed += 1
        return state


def _route_xy(route) -> np.ndarray | None:
    if route is None or len(route.waypoints) < 2:
        return None
    return np.array([[w.x, w.y] for w in route.waypoints], dtype=float)


class MetaDriveModel(BaseTrajectoryModel):
    def __init__(
        self,
        spec: PolicySpec,
        camera_ids: list[str],
        output_frequency_hz: int,
        device: torch.device,
    ):
        self.spec = spec
        self._camera_ids = camera_ids
        self._output_frequency_hz = output_frequency_hz
        self.policy = build_policy(spec.policy)
        self.planner = ModeConditionedPlanner(
            PlannerConfig(dt_s=1.0 / output_frequency_hz, **spec.planner)
        )
        self.encoder = None
        if spec.encoder is not None:
            from metadrive_rl.perception import build_encoder

            self.encoder = build_encoder(spec.encoder, device)
        self.feature_dir = Path(spec.feature_dir) if spec.feature_dir else None
        if self.feature_dir is not None:
            self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.sessions = _SessionTable()
        logger.info("MetaDrive policy %s, encoder %s", self.policy.version, spec.encoder)

    @classmethod
    def from_config(cls, model_cfg, device, camera_ids, context_length, output_frequency_hz):
        return cls(PolicySpec.parse(model_cfg.checkpoint_path), camera_ids, output_frequency_hz,
                   device)

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return 1

    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz

    def _encode_command(self, command: DriveCommand) -> int:
        return int(command)

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        return self.predict_batch([prediction_input])[0]

    def predict_batch(self, prediction_inputs: list[PredictionInput]) -> list[ModelPrediction]:
        states = [self.sessions.get(p) for p in prediction_inputs]
        for p, s in zip(prediction_inputs, states, strict=True):
            self._follow_route_reanchoring(s, _route_xy(p.route))
        observations = [self._observe(p, s) for p, s in zip(prediction_inputs, states, strict=True)]
        due = [i for i, s in enumerate(states) if self._decision_due(s, observations[i])]
        if self.encoder is not None and due:
            feats = self.encoder.encode([observations[i].image for i in due])
            for i, f in zip(due, feats, strict=True):
                observations[i].features = f
        return [
            self._step(observations[i], states[i], i in due) for i in range(len(states))
        ]

    def _follow_route_reanchoring(self, state: _SessionState, route_xy: np.ndarray | None) -> None:
        """Keep the committed lane when the route jumps laterally under the ego.

        AlpaSim's MAP route generator re-anchors the route to the lane the ego
        occupies, e.g. after a lane change. The ego cannot move laterally by
        more than ``REANCHOR_JUMP_M`` between calls, so a larger change in its
        offset from the route means the route moved; shift the committed
        offsets by the same amount to keep the ego's absolute lane.
        """
        if route_xy is None:
            state.route_offset_m = None
            return
        path = RoutePath(route_xy, self.planner.cfg.path_step_m)
        s, offset = path.project(np.zeros(2))
        if not path.covers(s):
            # Ego is past the route's end (or before its start): the offset to
            # the straight extension says nothing about re-anchoring.
            state.route_offset_m = None
            return
        if state.route_offset_m is not None:
            jump = offset - state.route_offset_m
            if abs(jump) > REANCHOR_JUMP_M:
                state.lane_offset_m += jump
                state.base_offset_m += jump
        state.route_offset_m = offset

    def _observe(self, p: PredictionInput, state: _SessionState) -> Observation:
        # The servicer passes plain (timestamp_us, image) tuples.
        timestamp_us, image = p.camera_images[self._camera_ids[0]][-1]
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        return Observation(
            timestamp_us=int(timestamp_us),
            speed_mps=float(p.speed),
            accel_mps2=float(p.acceleration),
            route_xy=_route_xy(p.route),
            command=int(p.command),
            lane_offset_m=state.lane_offset_m,
            image=image,
        )

    def _decision_due(self, state: _SessionState, obs: Observation) -> bool:
        if state.decision is None:
            return True
        return obs.timestamp_us - state.decision_time_us >= self.spec.decision_period_s * 1e6 - 1

    def _step(self, obs: Observation, state: _SessionState, is_decision: bool) -> ModelPrediction:
        if state.last_obs_time_us is not None:
            dt_s = max(obs.timestamp_us - state.last_obs_time_us, 0) / 1e6
            state.travelled_since_target_m += obs.speed_mps * dt_s
        state.last_obs_time_us = obs.timestamp_us
        if is_decision:
            state.decision = decide(self.policy, obs, state.rng, self.spec.greedy)
            state.decision_id = uuid.uuid4().hex
            state.decision_time_us = obs.timestamp_us
            target = self.planner.next_lane_offset(state.decision.action, state.lane_offset_m)
            if target != state.lane_offset_m:
                state.travelled_since_target_m = 0.0
            state.base_offset_m, state.lane_offset_m = state.lane_offset_m, target
            if self.feature_dir is not None and obs.features is not None:
                np.save(self.feature_dir / f"{state.decision_id}.npy", obs.features)
        decision = state.decision
        assert decision is not None

        plans = self.planner.plan_all(
            ACTIONS, obs.speed_mps, obs.accel_mps2, obs.route_xy, state.base_offset_m,
            state.travelled_since_target_m,
        )
        chosen = plans[decision.action_index]

        record = {
            RECORD_KEY: {
                "policy_version": self.policy.version,
                "decision_id": state.decision_id,
                "is_decision": is_decision,
                "step": state.step,
                "action": decision.action.to_text(),
                "action_index": decision.action_index,
                "log_prob": decision.log_prob,
                "probs": np.round(decision.probs, 5).tolist() if is_decision else None,
                "speed_mps": obs.speed_mps,
                "command": obs.command,
                "lane_offset_m": state.lane_offset_m,
                "lateral_elapsed_m": round(state.travelled_since_target_m, 3),
                "flags": chosen.flags,
            }
        }
        state.step += 1

        positions = np.zeros((len(plans), self.planner.num_steps, 3))
        rotations = np.zeros((len(plans), self.planner.num_steps, 3, 3))
        for k, plan in enumerate(plans):
            single = ModelPrediction.from_planar(plan.xy, plan.heading)
            positions[k] = single.candidate_positions[0]
            rotations[k] = single.candidate_rotations[0]
        return ModelPrediction(
            candidate_positions=positions,
            candidate_rotations=rotations,
            selected_index=decision.action_index,
            reasoning_text=json.dumps(record),
        )
