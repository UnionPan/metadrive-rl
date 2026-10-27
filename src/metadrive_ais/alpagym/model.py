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
    # Cosmos-RL's safetensors export reads a generation config from the base checkpoint.
    from transformers import GenerationConfig

    GenerationConfig().save_pretrained(out)
    prior = {
        f"prior.{k[len('prior.') :]}": v.contiguous()
        for k, v in latent["state_dict"].items()
        if k.startswith("prior.")
    }
    save_file(prior, str(out / "model.safetensors"))
    return out


class NullTokenizer:
    """Cosmos-RL's controller loads a tokenizer for every RL run; the latent policy
    has no text, so it gets this stub (registered as a custom tokenizer loader)."""

    pad_token_id = eos_token_id = 0
    pad_token = eos_token = ""

    def __len__(self) -> int:
        return 1

    def decode(self, *args, **kwargs) -> str:
        return ""

    def batch_decode(self, sequences, *args, **kwargs) -> list[str]:
        return ["" for _ in sequences]

    def save_pretrained(self, save_directory, *args, **kwargs) -> tuple[str, ...]:
        return ()  # nothing to save (called with Cosmos-RL checkpoints)


def is_latent_checkpoint(model_name_or_path: str) -> bool:
    config = Path(model_name_or_path) / "config.json"
    return config.is_file() and json.loads(config.read_text()).get("model_type") == MODEL_TYPE


def bundle_from_export(export_dir: str | Path, out: str | Path, policy_version: int) -> Path:
    """VaVAM-AIS bundle (AlpaSim driver) from a Cosmos-RL safetensors export of the prior."""
    from safetensors.torch import load_file

    export_dir = Path(export_dir)
    config = json.loads((export_dir / "config.json").read_text())
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(f"{export_dir} is not a {MODEL_TYPE} export")
    prior = {}
    for shard in sorted(export_dir.glob("*.safetensors")):
        prior.update(load_file(str(shard)))
    bundle = torch.load(config["bundle_path"], map_location="cpu", weights_only=True)
    state = bundle["latent"]["state_dict"]
    expected = {k for k in state if k.startswith("prior.")}
    if set(prior) != expected:
        raise KeyError(f"export keys {sorted(prior)} != bundle prior keys {sorted(expected)}")
    for key, value in prior.items():
        state[key] = value.to(state[key].dtype)
    bundle["policy_version"] = int(policy_version)
    bundle["fit"] = f"alpagym export {export_dir}"
    torch.save(bundle, out)
    return Path(out)


def register_cosmos() -> None:
    """Register the Cosmos-RL model wrapper, weight mapper and tokenizer stub (AlpaGym only)."""
    from cosmos_rl.policy.model.base import BaseModel, ModelRegistry, WeightMapper
    from cosmos_rl.utils import util as cosmos_util

    register_hf()
    if MODEL_TYPE in ModelRegistry._MODEL_REGISTRY:
        return
    cosmos_util.register_tokenizer_loader(is_latent_checkpoint, lambda path: NullTokenizer())

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
            """FSDP2 over the DP mesh, as Cosmos-RL expects, in fp32: the rollout
            scores log q in fp32 and the PPO ratio must start at exactly 1."""

            def parallelize(model, parallel_dims, config, pp_loss_fn=None):
                from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

                if parallel_dims.dp_shard_enabled:
                    names = ("dp_shard_cp",)
                    if parallel_dims.dp_replicate_enabled:
                        names = ("dp_replicate", "dp_shard_cp")
                    mesh = parallel_dims.mesh[names]
                    if mesh.size() != 1:
                        raise NotImplementedError("the latent prior is tiny: use one policy GPU")
                    policy = MixedPrecisionPolicy(
                        param_dtype=torch.float32, reduce_dtype=torch.float32
                    )
                    fully_shard(model, mesh=mesh, mp_policy=policy)
                elif parallel_dims.dp_replicate_enabled:
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
            from torch.distributed.tensor import DTensor

            state = load_file(os.path.join(model_name_or_path, "model.safetensors"))
            params = dict(self.named_parameters())
            if set(params) != set(state):
                raise KeyError(f"checkpoint keys {sorted(state)} != model keys {sorted(params)}")
            for name, param in params.items():
                # FSDP2 on a one-rank mesh: the local shard is the full tensor.
                target = param.to_local() if isinstance(param, DTensor) else param.data
                with torch.no_grad():
                    target.copy_(state[name].to(device=target.device, dtype=target.dtype))
            if device is not None and not any(isinstance(p, DTensor) for p in params.values()):
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

    parser = argparse.ArgumentParser(
        description="Export a VaVAM-AIS bundle's prior for AlpaGym, or (--from-export) turn an "
        "AlpaGym/Cosmos-RL safetensors export back into a VaVAM-AIS bundle."
    )
    parser.add_argument("source", type=Path, help="bundle.pt, or an export dir with --from-export")
    parser.add_argument("out", type=Path)
    parser.add_argument("--from-export", action="store_true")
    parser.add_argument("--policy-version", type=int, default=0)
    args = parser.parse_args()
    if args.from_export:
        out = bundle_from_export(args.source, args.out, args.policy_version)
        print(json.dumps({"bundle": str(out)}))
    else:
        out = export_checkpoint(args.source, args.out)
        print(json.dumps({"checkpoint": str(out)}))


if __name__ == "__main__":
    main()
