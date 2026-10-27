"""``alpagym.policy_bundles`` entry point for the VaVAM-AIS latent policy."""

from __future__ import annotations

from typing import Any


def install_runtime_bridge() -> None:
    """Register the Hugging Face model type and the Cosmos-RL wrapper."""
    from metadrive_ais.alpagym.model import register_cosmos

    register_cosmos()


def setup_tokenizer(config: Any) -> None:
    del config
    install_runtime_bridge()
    return None


def build_model_inputs(run_config: Any):
    del run_config
    from metadrive_ais.alpagym.inference import build_model_inputs as from_replay

    return from_replay


def build_data_packer(run_config: Any, cosmos_role: str | None) -> Any:
    from alpagym_runtime.cosmos.packer import build_alpagym_data_packer

    install_runtime_bridge()
    return build_alpagym_data_packer(
        run_config=run_config,
        cosmos_role=cosmos_role,
        build_model_inputs=build_model_inputs(run_config),
    )


def load_inference_model(run_config: Any, device: Any, dtype: Any) -> Any:
    del dtype  # VaVAM runs under fp16 autocast; the prior in fp32
    from metadrive_ais.alpagym.inference import VaVAMAISInferenceModel

    model = run_config.policy.model
    return VaVAMAISInferenceModel(model.path, device, dict(model.bundle_config))


def get_bundle():
    from alpagym_runtime.policies.registry import PolicyBundle

    return PolicyBundle(
        setup_tokenizer=setup_tokenizer,
        build_data_packer=build_data_packer,
        install_runtime_bridge=install_runtime_bridge,
        load_inference_model=load_inference_model,
        build_model_inputs=build_model_inputs,
    )
