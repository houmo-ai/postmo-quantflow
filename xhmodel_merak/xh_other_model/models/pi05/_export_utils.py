import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from onnxsim import simplify


class Siglip(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, pixel_values):
        # Pi0.5 consumes unscaled projected vision features. Transformers 5.3
        # divided get_image_features() by sqrt(hidden_size), while 5.16 does
        # not. Export the two native modules directly to keep this contract
        # independent of that changing convenience API.
        hidden = self.model.vision_tower(pixel_values).last_hidden_state
        return self.model.multi_modal_projector(hidden)


class TimeMLPWrapper(nn.Module):
    def __init__(self, time_mlp_in, time_mlp_out):
        super().__init__()
        self.time_mlp_in = time_mlp_in
        self.time_mlp_out = time_mlp_out

    def forward(self, time_emb):
        from torch.nn import functional as F

        x = self.time_mlp_in(time_emb)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        return F.silu(x)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pi05_policy(model_path: str, device: str = "cpu", *, config=None):
    from lerobot.policies.pi05 import PI05Policy
    from safetensors.torch import load_file

    if config is None:
        config = _load_local_pi05_config_compat(model_path)
    if config is None:
        from lerobot.configs.policies import PreTrainedConfig

        config = PreTrainedConfig.from_pretrained(model_path)
    # LeRobot's from_pretrained catches even strict load_state_dict failures
    # and returns a partially initialized policy. Deployment must fail closed.
    config.device = device
    policy = PI05Policy(config)
    weights = load_file(str(Path(model_path) / "model.safetensors"))
    weights = _normalize_pi05_checkpoint_keys(
        weights, policy.state_dict(),
        tied_embeddings=policy.model.paligemma_with_expert.paligemma.config.tie_word_embeddings,
    )
    policy.load_state_dict(weights, strict=True)
    policy.to(device)
    policy.config.device = device
    from ._lerobot_compat import adapt_pi05_reference

    return adapt_pi05_reference(policy.eval())


def _normalize_pi05_checkpoint_keys(weights, expected, *, tied_embeddings=False):
    """Accept the canonical LeRobot SigLIP nesting before/after Transformers 5.16.

    Only rename an exact legacy prefix when the destination exists in the
    instantiated model. Duplicate aliases, unknown keys and shape errors must
    not silently discard checkpoint tensors.
    """
    old = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model."
    new = "model.paligemma_with_expert.paligemma.model.vision_tower."
    normalized = {}
    for name, value in weights.items():
        target = name
        if name.startswith(old) and name not in expected:
            candidate = new + name[len(old):]
            if candidate in expected:
                target = candidate
        if target in normalized:
            raise ValueError(f"Duplicate Pi0.5 checkpoint destination: {target}")
        normalized[target] = value
    head = "model.paligemma_with_expert.paligemma.lm_head.weight"
    embedding = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    # Safetensors omits aliases of a tied tensor. Only restore the declared
    # sharing relationship; never invent an embedding for an untied model.
    if tied_embeddings and embedding in expected and embedding not in normalized and head in normalized:
        normalized[embedding] = normalized[head]
    return normalized


def _load_local_pi05_config_compat(model_path: str):
    config_file = Path(model_path) / "config.json"
    if not config_file.is_file():
        return None

    raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    if raw_config.get("relative_exclude_joints", []) is not None:
        return None

    import draccus
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    raw_config.pop("type", None)
    raw_config["relative_exclude_joints"] = []
    return draccus.decode(PI05Config, raw_config)


def export_onnx_and_simplify(
    model: nn.Module,
    inputs: torch.Tensor | Sequence[Any],
    onnx_file: Path,
    simplified_onnx_file: Path,
    input_names: list[str],
    output_names: list[str],
    test_input_shapes: dict[str, list[int]],
    *,
    opset_version: int = 17,
    verbose: bool = False,
) -> Path:
    import onnx

    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    simplified_onnx_file.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        str(onnx_file),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        verbose=verbose,
    )
    onnx_model = onnx.load(str(onnx_file))
    model_simplified, check = simplify(onnx_model, test_input_shapes=test_input_shapes)
    if not check:
        raise RuntimeError(f"ONNX simplification failed for {onnx_file}")
    onnx.save(model_simplified, str(simplified_onnx_file))
    return simplified_onnx_file
