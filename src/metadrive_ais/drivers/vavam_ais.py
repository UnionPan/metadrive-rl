"""AlpaSim driver: VaVAM-B whose action expert is conditioned on an AIS memory.

Registered under ``alpasim.models`` as ``vavam_ais``. ``model.checkpoint_path``
names an AIS bundle (:func:`save_bundle`) that records the VaVAM-B checkpoint,
the frozen feature encoder, the memory encoder and the gated adapter;
``model.tokenizer_path`` is VaVAM's VQ tokenizer, as for the stock ``vam``
driver.

Per query and session: encode the newest frame with the frozen encoder (e_t),
compute the executed ego motion since the session's previous query (u_{t-1},
as in :mod:`metadrive_ais.memory.sequences`), advance the memory (z_t), set
the adapter's residual on VaVAM's action tokens, and run VaVAM's stock
inference. Needs the session id (``patches/alpasim/0004``).

Bundles with a latent action channel (M5a, :mod:`metadrive_ais.models.latent_policy`)
also sample ``c_t ~ q_phi(c | z_t)`` per query and add ``W_c c_t``. The draw is
seeded by session and query index. ``VAVAM_AIS_LATENT=mean`` uses the prior mean
instead. ``VAVAM_AIS_LATENT_HOLD=k`` (default 1) draws a new latent only every k
queries and holds it in between (temporally persistent exploration; the draws
are an option policy with decisions at queries 0, k, 2k, ...). With
``VAVAM_AIS_LATENT_LOG=<file>``, every query appends one JSON line (session,
time, z_t, c_t, log q, seed, decision flag, hold, policy version) for the RL update.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from alpasim_driver.models.base import ModelPrediction, PredictionInput
from alpasim_driver.models.vam_model import VAMModel

from metadrive_ais.memory.ais import MemoryEncoder
from metadrive_ais.models.latent_policy import LatentChannel
from metadrive_ais.models.vavam_conditioning import GatedAdapter, attach_conditioning
from metadrive_ais.vision.encoders import FrozenEncoder

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = "metadrive-ais/vavam-ais-bundle@1"
MAX_SESSIONS = 256


def save_bundle(
    path,
    vavam_checkpoint: str,
    feature_encoder: str,
    memory: MemoryEncoder,
    adapter: GatedAdapter,
    latent: LatentChannel | None = None,
    policy_version: int = 0,
) -> None:
    torch.save(
        {
            **(
                {"latent": {"config": latent.config, "state_dict": latent.state_dict()}}
                if latent is not None
                else {}
            ),
            "policy_version": policy_version,
            "format": BUNDLE_FORMAT,
            "vavam_checkpoint": str(vavam_checkpoint),
            # Trained against VaVAM with its muP base shapes (patches/alpasim/0005).
            "vavam_mup_base_shapes": True,
            "feature_encoder": feature_encoder,
            "memory": {"config": memory.config, "state_dict": memory.state_dict()},
            "adapter": {
                "state_dim": adapter.proj.in_features,
                "token_dim": adapter.proj.out_features,
                "state_dict": adapter.state_dict(),
            },
        },
        path,
    )


def _pose(p) -> tuple[float, float, float, float]:
    q = p.pose.quat
    yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
    return float(p.pose.vec.x), float(p.pose.vec.y), yaw, p.timestamp_us / 1e6


def executed_motion(prev, cur) -> list[float]:
    """u_{t-1}: dx, dy (previous ego frame), dyaw, dt, speed — zeros without history."""
    if prev is None:
        return [0.0] * 5
    (x0, y0, yaw0, t0), (x1, y1, yaw1, t1) = prev, cur
    c, s = np.cos(yaw0), np.sin(yaw0)
    dx, dy = x1 - x0, y1 - y0
    dt = max(t1 - t0, 1e-3)
    dyaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
    return [c * dx + s * dy, -s * dx + c * dy, dyaw, dt, float(np.hypot(dx, dy) / dt)]


def latent_seed(session_uuid: str, query: int) -> int:
    """Seed of the latent draw at one query: reproducible, distinct per session and query."""
    return int(hashlib.sha256(f"{session_uuid}/{query}".encode()).hexdigest()[:15], 16)


class VaVAMAISModel(VAMModel):
    def __init__(
        self,
        bundle: dict[str, Any],
        tokenizer_path: str,
        device,
        camera_ids,
        context_length: int = 8,
    ):
        if bundle.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"not a VaVAM-AIS bundle: {bundle.get('format')!r}")
        if bundle.get("vavam_mup_base_shapes") and not os.environ.get("VAVAM_MUP_SHAPES_DIR"):
            raise RuntimeError(
                "this bundle was trained against VaVAM with muP base shapes; set "
                "VAVAM_MUP_SHAPES_DIR (patches/alpasim/0005) so the driver loads it the same way"
            )
        super().__init__(
            checkpoint_path=bundle["vavam_checkpoint"],
            tokenizer_path=tokenizer_path,
            device=device,
            camera_ids=camera_ids,
            context_length=context_length,
        )
        self._conditioner = attach_conditioning(self._vam)
        self._features = FrozenEncoder(bundle["feature_encoder"], device)
        self._memory = MemoryEncoder(**bundle["memory"]["config"]).to(device).eval()
        self._memory.load_state_dict(bundle["memory"]["state_dict"])
        a = bundle["adapter"]
        self._adapter = GatedAdapter(a["state_dim"], a["token_dim"]).to(device).eval()
        self._adapter.load_state_dict(a["state_dict"])
        self._latent = None
        if "latent" in bundle:
            self._latent = LatentChannel(**bundle["latent"]["config"]).to(device).eval()
            self._latent.load_state_dict(bundle["latent"]["state_dict"])
        self._latent_mode = os.environ.get("VAVAM_AIS_LATENT", "sample")
        if self._latent_mode not in ("sample", "mean"):
            raise ValueError(f"VAVAM_AIS_LATENT={self._latent_mode!r}: expected sample or mean")
        self._latent_hold = int(os.environ.get("VAVAM_AIS_LATENT_HOLD", "1"))
        self._latent_log = os.environ.get("VAVAM_AIS_LATENT_LOG")
        self._policy_version = int(bundle.get("policy_version", 0))
        logger.info(
            "VaVAM-AIS: memory %s, adapter gate %.4f, latent %s, mode %s, hold %d, "
            "policy version %d, VAVAM_MUP_SHAPES_DIR %s",
            bundle["memory"]["config"]["kind"],
            float(self._adapter.gate),
            bundle["latent"]["config"] if "latent" in bundle else None,
            self._latent_mode,
            self._latent_hold,
            self._policy_version,
            os.environ.get("VAVAM_MUP_SHAPES_DIR"),
        )
        self._sessions: OrderedDict[str, dict] = OrderedDict()

    @classmethod
    def from_config(cls, model_cfg, device, camera_ids, context_length, output_frequency_hz):
        if model_cfg.tokenizer_path is None:
            raise ValueError("VaVAM-AIS requires tokenizer_path")
        bundle = torch.load(model_cfg.checkpoint_path, map_location="cpu", weights_only=True)
        return cls(bundle, model_cfg.tokenizer_path, device, camera_ids, context_length or 8)

    def _session(self, uuid: str | None) -> dict:
        if uuid is None:
            raise ValueError("VaVAM-AIS needs PredictionInput.session_uuid (patches/alpasim/0004)")
        if uuid not in self._sessions:
            while len(self._sessions) >= MAX_SESSIONS:
                self._sessions.popitem(last=False)
            self._sessions[uuid] = {
                "state": self._memory.initial_state(1, self._device)[0],
                "pose": None,
                "queries": 0,
            }
        self._sessions.move_to_end(uuid)
        return self._sessions[uuid]

    @torch.no_grad()
    def _residuals(self, inputs: list[PredictionInput]) -> torch.Tensor:
        cam = self._camera_ids[0]
        frames = []
        for inp in inputs:
            image = inp.camera_images[cam][-1][1]
            frames.append(
                image.cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
            )
        embed = torch.from_numpy(self._features.encode(frames)).to(self._device)
        sessions = [self._session(inp.session_uuid) for inp in inputs]
        poses = [_pose(inp.ego_pose_history[-1]) for inp in inputs]
        motion = torch.tensor(
            [executed_motion(s["pose"], p) for s, p in zip(sessions, poses, strict=True)],
            dtype=torch.float32,
            device=self._device,
        )
        state = torch.stack([s["state"] for s in sessions])
        z, new_state = self._memory.step(state, embed, motion)
        for s, st, p in zip(sessions, new_state, poses, strict=True):
            s["state"], s["pose"] = st, p
        residual = self._adapter(z)  # [B, token_dim]
        if self._latent is not None:
            residual = residual + self._latent_residual(inputs, sessions, z)
        for s in sessions:
            s["queries"] += 1
        return residual

    def _latent_residual(self, inputs, sessions, z: torch.Tensor) -> torch.Tensor:
        dist = self._latent.prior(z)
        seeds = [
            latent_seed(inp.session_uuid, s["queries"])
            for inp, s in zip(inputs, sessions, strict=True)
        ]
        decision = [s["queries"] % self._latent_hold == 0 for s in sessions]
        if self._latent_mode == "mean":
            c = dist.mean
        else:
            noise = torch.stack(
                [
                    torch.randn(dist.mean.shape[-1], generator=torch.Generator().manual_seed(seed))
                    for seed in seeds
                ]
            ).to(self._device)
            c = dist.mean + dist.stddev * noise
            for i, s in enumerate(sessions):  # between decisions, keep the last draw
                if decision[i]:
                    s["latent"] = c[i]
                else:
                    c[i] = s["latent"]
        logp = dist.log_prob(c).sum(-1)
        if self._latent_log:
            with open(self._latent_log, "a") as f:
                for i, inp in enumerate(inputs):
                    record = {
                        "session": inp.session_uuid,
                        "query": sessions[i]["queries"],
                        "time_us": int(inp.ego_pose_history[-1].timestamp_us),
                        "z": z[i].tolist(),
                        "c": c[i].tolist(),
                        "logp": float(logp[i]),
                        "seed": seeds[i],
                        "decision": decision[i],
                        "hold": self._latent_hold,
                        "mode": self._latent_mode,
                        "policy_version": self._policy_version,
                    }
                    f.write(json.dumps(record) + "\n")
        return self._latent.decoder(c)

    def predict_batch(self, prediction_inputs: list[PredictionInput]) -> list[ModelPrediction]:
        if not prediction_inputs:
            return []
        self._conditioner.delta = self._residuals(prediction_inputs)[:, None, :]
        try:
            return super().predict_batch(prediction_inputs)
        finally:
            self._conditioner.delta = None


def bundle_from_sft(sft_checkpoint, out, vavam_checkpoint) -> None:
    """Deployable bundle from a ``metadrive_ais.training.flow_sft`` checkpoint."""
    ckpt = torch.load(sft_checkpoint, map_location="cpu", weights_only=True)
    memory = MemoryEncoder(**ckpt["memory"]["config"])
    memory.load_state_dict(ckpt["memory"]["state_dict"])
    adapter = GatedAdapter(memory.state_dim)
    adapter.load_state_dict(ckpt["adapter"])
    latent = None
    if "latent" in ckpt:
        latent = LatentChannel(**ckpt["latent"]["config"])
        latent.load_state_dict(ckpt["latent"]["state_dict"])
    save_bundle(out, vavam_checkpoint, ckpt["feature_encoder"], memory, adapter, latent)


def bundle_with_rl_latent(
    sft_checkpoint, out, vavam_checkpoint, latent_dim: int, scale: float, seed: int = 0
) -> None:
    """Stock VaVAM plus an RL-only latent channel, from a flow-SFT checkpoint.

    The SFT memory supplies z_t (features only); its adapter gate is set to zero,
    so the mean policy is stock VaVAM-B. The latent decoder is a fixed random
    orthonormal map scaled by ``scale``; the prior starts at N(0, I).
    """
    ckpt = torch.load(sft_checkpoint, map_location="cpu", weights_only=True)
    memory = MemoryEncoder(**ckpt["memory"]["config"])
    memory.load_state_dict(ckpt["memory"]["state_dict"])
    adapter = GatedAdapter(memory.state_dim)
    adapter.load_state_dict(ckpt["adapter"])
    with torch.no_grad():
        adapter.gate.zero_()
    torch.manual_seed(seed)
    latent = LatentChannel(memory.state_dim, latent_dim, decoder_scale=scale, decoder_fixed=True)
    save_bundle(out, vavam_checkpoint, ckpt["feature_encoder"], memory, adapter, latent)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Convert a flow-SFT checkpoint to a VaVAM-AIS bundle."
    )
    parser.add_argument("sft_checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--rl-latent",
        nargs=2,
        metavar=("DIM", "SCALE"),
        help="stock VaVAM + fixed random latent channel for RL (bundle_with_rl_latent)",
    )
    parser.add_argument(
        "--vavam-checkpoint",
        type=Path,
        default=Path.home() / "ext/alpasim/data/drivers/vavam/VAM_width_1024_pretrained_139k.pt",
    )
    args = parser.parse_args()
    if args.rl_latent:
        dim, scale = int(args.rl_latent[0]), float(args.rl_latent[1])
        bundle_with_rl_latent(args.sft_checkpoint, args.out, args.vavam_checkpoint, dim, scale)
    else:
        bundle_from_sft(args.sft_checkpoint, args.out, args.vavam_checkpoint)
    print(f"bundle written to {args.out}")
