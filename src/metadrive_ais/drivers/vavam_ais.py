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
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from alpasim_driver.models.base import ModelPrediction, PredictionInput
from alpasim_driver.models.vam_model import VAMModel

from metadrive_ais.memory.ais import MemoryEncoder
from metadrive_ais.models.vavam_conditioning import GatedAdapter, attach_conditioning
from metadrive_ais.vision.encoders import FrozenEncoder

BUNDLE_FORMAT = "metadrive-ais/vavam-ais-bundle@1"
MAX_SESSIONS = 256


def save_bundle(
    path, vavam_checkpoint: str, feature_encoder: str, memory: MemoryEncoder, adapter: GatedAdapter
) -> None:
    torch.save(
        {
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
        return self._adapter(z)  # [B, token_dim]

    def predict_batch(self, prediction_inputs: list[PredictionInput]) -> list[ModelPrediction]:
        if not prediction_inputs:
            return []
        self._conditioner.delta = self._residuals(prediction_inputs)[:, None, :]
        try:
            return super().predict_batch(prediction_inputs)
        finally:
            self._conditioner.delta = None
