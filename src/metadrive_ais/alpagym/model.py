"""Trainer-side model of the VaVAM-AIS latent policy for AlpaGym / Cosmos-RL.

The trainable policy is the latent prior q_phi(c | z_t)
(:class:`metadrive_ais.models.latent_policy.LatentPrior`). Rollouts record the
AIS state z_t and the sampled latent c_t (VaVAM, the memory, the adapter and the
latent decoder are frozen), so the trainer scores ``log q_phi(c_t | z_t)`` without
running VaVAM. The checkpoint is a Hugging Face style directory: ``config.json``
(``model_type: vavam_ais_latent``, the frozen VaVAM-AIS bundle path) and
``model.safetensors`` (the prior), written by :func:`export_checkpoint`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from metadrive_ais.models.latent_policy import LatentPrior

MODEL_TYPE = "vavam_ais_latent"


class VaVAMLatentConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(self, bundle_path: str = "", state_dim: int = 256, latent_dim: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.bundle_path = bundle_path
        self.state_dim = state_dim
        self.latent_dim = latent_dim


def latent_log_probs(prior: torch.nn.Module, z, c, teacher: torch.nn.Module | None = None) -> dict:
    """``{"log_probs": log q(c | z) [B], "kl_div": KL(q || q_teacher) [B] or None}``."""
    dist = prior(z.float())
    out = {"log_probs": dist.log_prob(c.float()).sum(-1), "kl_div": None}
    if teacher is not None:
        out["kl_div"] = torch.distributions.kl_divergence(dist, teacher(z.float())).sum(-1)
    return out


class VaVAMLatentPolicy(PreTrainedModel):
    """Hugging Face model class for the latent prior (AutoModel registration)."""

    config_class = VaVAMLatentConfig

    def __init__(self, config: VaVAMLatentConfig):
        super().__init__(config)
        self.prior = LatentPrior(config.state_dim, config.latent_dim)

    def forward(self, z, c, teacher_model=None, **kwargs):
        del kwargs
        return latent_log_probs(self.prior, z, c, getattr(teacher_model, "prior", None))


def register_hf() -> None:
    AutoConfig.register(MODEL_TYPE, VaVAMLatentConfig, exist_ok=True)
    AutoModel.register(VaVAMLatentConfig, VaVAMLatentPolicy, exist_ok=True)


def export_checkpoint(bundle_path: str | Path, out_dir: str | Path) -> Path:
    """Hugging Face style checkpoint of a VaVAM-AIS bundle's latent prior."""
    from safetensors.torch import save_file

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    latent = bundle["latent"]  # any latent channel: only the prior is trained
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = VaVAMLatentConfig(
        bundle_path=str(Path(bundle_path).resolve()),
        state_dim=latent["config"]["state_dim"],
        latent_dim=latent["config"]["latent_dim"],
        architectures=["VaVAMLatentPolicy"],
    )
    config.save_pretrained(out)
    prior = {
        f"prior.{k[len('prior.') :]}": v.contiguous()
        for k, v in latent["state_dict"].items()
        if k.startswith("prior.")
    }
    save_file(prior, str(out / "model.safetensors"))
    return out


def register_cosmos() -> None:
    """Register the Cosmos-RL model wrapper and weight mapper (AlpaGym envs only)."""
    from cosmos_rl.policy.model.base import BaseModel, ModelRegistry, WeightMapper

    register_hf()
    if MODEL_TYPE in ModelRegistry._MODEL_REGISTRY:
        return

    class VaVAMLatentWeightMapper(WeightMapper):
        """Policy and rollout share parameter names (``prior.*``)."""

        def rollout_split_local_key_n_param_to_hf_key_n_param(self, param_name, param):
            return [(self.rollout_map_local_key_to_hf_key(param_name), param)]

    class VaVAMLatentCosmos(BaseModel):
        """Cosmos-RL ``BaseModel`` for the latent prior: tiny, never sharded."""

        def __init__(self, hf_config):
            super().__init__(hf_config)
            self.hf_config = hf_config
            self.prior = LatentPrior(hf_config.state_dim, hf_config.latent_dim)

        @staticmethod
        def supported_model_types():
            return [MODEL_TYPE]

        def forward(self, z, c, teacher_model=None, **kwargs):
            del kwargs  # e.g. return_log_prob, injected by the AlpaGym packer
            return latent_log_probs(self.prior, z, c, getattr(teacher_model, "prior", None))

        @property
        def parallelize_fn(self):
            def parallelize(model, parallel_dims, config, pp_loss_fn=None):
                if parallel_dims.dp_shard_enabled:
                    raise NotImplementedError(
                        "the latent prior is not sharded; use dp_shard_size 1"
                    )
                if parallel_dims.dp_replicate_enabled:
                    from torch.distributed._composable.replicate import replicate

                    replicate(model, device_mesh=parallel_dims.mesh)
                return None, None

            return parallelize, self

        def apply_pipeline_split(self, pp_rank, pp_size):
            raise NotImplementedError("pipeline parallelism is not supported")

        def get_position_ids(self, **kwargs):
            z = kwargs["z"]
            ids = torch.zeros(z.shape[0], 1, dtype=torch.long, device=z.device)
            return ids, ids, 1

        def load_hf_weights(
            self, model_name_or_path, parallel_dims=None, device=None, revision=None
        ):
            from safetensors.torch import load_file

            state = load_file(os.path.join(model_name_or_path, "model.safetensors"))
            self.load_state_dict(state, strict=True)
            if device is not None:
                self.to(device)

        def separate_model_parts(self):
            return [self]

        def post_to_empty_hook(self, cosmos_config):
            del cosmos_config

        @classmethod
        def from_pretrained(
            cls, hf_config, model_name_or_path=None, max_position_embeddings=None, **kwargs
        ):
            return cls(hf_config)

        @classmethod
        def get_nparams_and_flops(cls, seq_len):
            return 0, 0

    ModelRegistry.register(VaVAMLatentWeightMapper)(VaVAMLatentCosmos)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export a VaVAM-AIS bundle's prior for AlpaGym.")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = export_checkpoint(args.bundle, args.out)
    print(json.dumps({"checkpoint": str(out)}))


if __name__ == "__main__":
    main()
