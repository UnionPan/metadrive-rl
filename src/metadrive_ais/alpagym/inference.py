"""Rollout-side VaVAM-AIS inference model for AlpaGym (``InferenceModel`` protocol).

AlpaGym's generic session policy (``AlpamayoPolicy``, configured for one front
camera, 8 frames at 2 Hz and two ego poses 0.5 s apart) hands each query as a
``ModelInput``. Per query this model:

1. tokenizes the 8 frames exactly as AlpaSim's VaVAM driver does,
2. embeds the newest frame with the frozen DINOv3 encoder and computes the
   executed motion u_{t-1} from the two poses (as the training data),
3. computes the AIS state z_t with the bundle's memory. Only stateless memories
   (``frame``, ``motion``) are supported, because AlpaGym queries carry no
   per-session state,
4. samples c_t ~ q_phi(c | z_t) with the query's seed and records log q,
5. runs VaVAM-B with ``adapter(z_t) + W_c c_t`` added to its action tokens.

The synced model (``get_model``) is the prior only, under the same parameter
names as the trainer's (``prior.*``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from alpagym_runtime.inference.types import BatchedModelInput, BatchedModelOutput, ModelInput
from alpagym_runtime.replay import _REPLAY_SCHEMA_VERSION, ActionSelection, PolicyReplayData
from torch import nn

from metadrive_ais.memory.ais import MemoryEncoder
from metadrive_ais.models.latent_policy import LatentChannel
from metadrive_ais.models.vavam_conditioning import GatedAdapter, attach_conditioning
from metadrive_ais.models.vavam_core import VaVAMTokenizer, route_command
from metadrive_ais.models.vavam_loading import load_vavam
from metadrive_ais.vision.encoders import FrozenEncoder

HORIZON = 6
PAYLOAD_SCHEMA = "vavam_ais_latent"


class SyncedPrior(nn.Module):
    """The parameters Cosmos-RL syncs from trainer to rollout: ``prior.*``."""

    def __init__(self, prior: nn.Module):
        super().__init__()
        self.prior = prior


def motion_from_history(xyz: torch.Tensor, rot: torch.Tensor, dt_s: float) -> torch.Tensor:
    """u_{t-1} = (dx, dy in the previous ego frame, dyaw, dt, speed) from the rig-frame
    history ``[B, 1, N, 3]`` / ``[B, 1, N, 3, 3]`` whose last two rows are the
    previous and current query poses (the current one at the origin)."""
    p = xyz[:, 0, -2].float()  # previous pose, in the current rig frame
    r = rot[:, 0, -2].float()
    d = torch.einsum("bji,bj->bi", r, -p)  # R_prev^T (0 - p)
    yaw_prev = torch.atan2(r[:, 1, 0], r[:, 0, 0])
    dyaw = torch.remainder(-yaw_prev + math.pi, 2 * math.pi) - math.pi
    dt = torch.full_like(dyaw, dt_s)
    speed = torch.linalg.vector_norm(d[:, :2], dim=-1) / dt
    return torch.stack([d[:, 0], d[:, 1], dyaw, dt, speed], -1)


def headings(xy: torch.Tensor) -> torch.Tensor:
    """Yaw along the path [B, T] (origin prepended); holds the last yaw on zero steps."""
    pts = torch.cat([torch.zeros_like(xy[:, :1]), xy], 1)
    d = pts[:, 1:] - pts[:, :-1]
    yaw = torch.atan2(d[..., 1], d[..., 0])
    moving = torch.linalg.vector_norm(d, dim=-1) > 1e-3
    for t in range(yaw.shape[1]):
        prev = yaw[:, t - 1] if t else torch.zeros_like(yaw[:, 0])
        yaw[:, t] = torch.where(moving[:, t], yaw[:, t], prev)
    return yaw


class VaVAMAISInferenceModel:
    def __init__(self, checkpoint_dir: str | Path, device: torch.device, bundle_config: dict):
        cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text())
        bundle = torch.load(cfg["bundle_path"], map_location="cpu", weights_only=True)
        self.device = torch.device(device)
        self.dt_s = float(bundle_config.get("step_dt_s", 0.5))
        self.vam = load_vavam(
            bundle["vavam_checkpoint"], self.device, mup_base_shapes=bundle["vavam_mup_base_shapes"]
        ).requires_grad_(False)
        self.conditioner = attach_conditioning(self.vam)
        self.tokenizer = VaVAMTokenizer(self.device)
        self.features = FrozenEncoder(bundle["feature_encoder"], self.device)
        self.memory = MemoryEncoder(**bundle["memory"]["config"]).to(self.device).eval()
        if self.memory.kind not in ("frame", "motion"):
            raise ValueError(
                f"AlpaGym queries are stateless; memory {self.memory.kind!r} needs state"
            )
        self.memory.load_state_dict(bundle["memory"]["state_dict"])
        a = bundle["adapter"]
        self.adapter = GatedAdapter(a["state_dim"], a["token_dim"]).to(self.device).eval()
        self.adapter.load_state_dict(a["state_dict"])
        self.latent = LatentChannel(**bundle["latent"]["config"]).to(self.device).eval()
        self.latent.load_state_dict(bundle["latent"]["state_dict"])
        from safetensors.torch import load_file

        prior = load_file(str(Path(checkpoint_dir) / "model.safetensors"))
        self.policy = SyncedPrior(self.latent.prior)
        self.policy.load_state_dict(prior, strict=True)

    def get_model(self) -> nn.Module:
        return self.policy

    def set_model(self, model: nn.Module) -> None:
        self.policy = model
        self.latent.prior = model.prior

    @torch.no_grad()
    def sample_trajectories_from_data(
        self, model_input: BatchedModelInput, sampling, return_trace_for_rl: bool = False
    ) -> BatchedModelOutput:
        del sampling, return_trace_for_rl
        frames = model_input.camera_frames  # [B, T, 3, H, W] uint8
        b, t = frames.shape[:2]
        hwc = frames.permute(0, 1, 3, 4, 2).cpu().numpy()
        tokens = self.tokenizer([f for clip in hwc for f in clip]).reshape(b, t, 18, 32)
        embed = torch.from_numpy(self.features.encode([clip[-1] for clip in hwc])).to(self.device)
        motion = motion_from_history(
            model_input.ego_history_xyz, model_input.ego_history_rot, self.dt_s
        ).to(self.device)
        command = torch.tensor(
            [route_command(r) for r in model_input.route_xy.float().cpu().numpy()],
            device=self.device,
        ).unsqueeze(1)
        z = self.memory(embed[:, None], motion[:, None])[:, -1]
        dist = self.latent.prior(z)
        if model_input.seed is not None:
            gens = [torch.Generator().manual_seed(int(s)) for s in model_input.seed.reshape(-1)]
            noise = torch.stack([torch.randn(dist.mean.shape[-1], generator=g) for g in gens])
        else:
            noise = torch.randn(dist.mean.shape)
        c = dist.mean + dist.stddev * noise.to(self.device)
        logp = dist.log_prob(c).sum(-1)
        self.conditioner.delta = (self.adapter(z) + self.latent.decoder(c))[:, None, :]
        try:
            with torch.autocast(self.device.type, dtype=torch.float16):
                traj = self.vam(tokens, command, torch.float16)
        finally:
            self.conditioner.delta = None
        xy = traj.float().reshape(b, HORIZON, 2)
        yaw = headings(xy)
        cos, sin, zero, one = yaw.cos(), yaw.sin(), torch.zeros_like(yaw), torch.ones_like(yaw)
        rot = torch.stack(
            [
                torch.stack([cos, -sin, zero], -1),
                torch.stack([sin, cos, zero], -1),
                torch.stack([zero, zero, one], -1),
            ],
            -2,
        )  # [B, T, 3, 3]
        xyz = torch.cat([xy, torch.zeros_like(xy[..., :1])], -1)
        return BatchedModelOutput(
            pred_xyz=xyz[:, None, None],  # [B, sets=1, samples=1, T, 3]
            pred_rot=rot[:, None, None],
            logprob=logp.view(b, 1, 1),
            extra={"z": z.float().cpu(), "c": c.float().cpu()},
        )

    def build_policy_replay_data(
        self, model_input: ModelInput, model_output, action_selection: ActionSelection
    ) -> PolicyReplayData:
        del model_input
        return PolicyReplayData(
            replay_schema_version=_REPLAY_SCHEMA_VERSION,
            payload_schema=PAYLOAD_SCHEMA,
            payload_schema_version=1,
            model_family="vavam_ais",
            action_selection=action_selection,
            old_logprob=model_output.logprob[action_selection.set_ix, action_selection.sample_ix]
            .reshape(())
            .float()
            .cpu(),
            payload={"z": model_output.extra["z"], "c": model_output.extra["c"]},
        )


def build_model_inputs(replay: PolicyReplayData) -> tuple[dict, torch.Tensor]:
    """Trainer-side: replay payload -> forward kwargs (z, c) and the rollout log q."""
    if replay.payload_schema != PAYLOAD_SCHEMA:
        raise ValueError(f"not a VaVAM-AIS latent replay: {replay.payload_schema!r}")
    payload = replay.payload
    inputs = {
        "z": torch.as_tensor(np.asarray(payload["z"]), dtype=torch.float32),
        "c": torch.as_tensor(np.asarray(payload["c"]), dtype=torch.float32),
    }
    return inputs, torch.as_tensor(replay.old_logprob, dtype=torch.float32).reshape(())
