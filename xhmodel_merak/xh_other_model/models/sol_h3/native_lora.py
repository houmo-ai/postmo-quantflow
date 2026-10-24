# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Keep native H3 Turbo LoRA branches separate from compensated base weights."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


_PAIR = re.compile(r"^(.*)\.lora_([AB])(?:\.default)?\.weight$")
_TARGET = re.compile(
    r"^(?P<stack>transformer_blocks|token_refiner\.refiner_blocks)\.(?P<index>\d+)\."
    r"(?P<projection>attn\.to_[qkv]|attn\.to_out\.0|ff\.net\.0\.proj|ff\.net\.2)$"
)


@dataclass(frozen=True)
class NativeLoRAWeightPair:
    target: str
    a_key: str
    b_key: str
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]

    @property
    def component(self):
        return "text" if self.target.startswith("token_refiner.") else "dit"


@dataclass(frozen=True)
class NativeLoRASpec:
    path: Path
    name: str
    rank: int
    alpha: float
    pairs: tuple[NativeLoRAWeightPair, ...]
    checkpoint_sha256: str

    @property
    def scale(self):
        return self.alpha / self.rank

    def to_dict(self):
        return dict(
            name=self.name,
            format="native_h3_keep_lora",
            checkpoint_sha256=self.checkpoint_sha256,
            rank=self.rank,
            alpha=self.alpha,
            scale=self.scale,
            target_count=len(self.pairs),
            component_target_counts={
                component: sum(pair.component == component for pair in self.pairs) for component in ("text", "dit")
            },
        )


def inspect_native_lora(path, *, name=None):
    """Inspect native H3 safetensors or a PEFT directory without loading tensors.

    Turbo's shipped adapters contain only attention and SwiGLU projections in
    DiT and the text refiner. Unknown targets (including AdaLN) are rejected:
    folding such adapters would require a separately validated conditioning path.
    """
    from safetensors import safe_open

    path = Path(path).expanduser().resolve()
    config = None
    if path.is_dir():
        config = json.loads((path / "adapter_config.json").read_text())
        if not isinstance(config, dict) or str(config.get("peft_type", "")).upper() != "LORA":
            raise ValueError("native H3 adapters require a PEFT LORA configuration")
        unsupported = (
            "alpha_pattern",
            "rank_pattern",
            "modules_to_save",
            "layer_replication",
            "use_dora",
            "use_qalora",
            "use_rslora",
            "lora_bias",
            "fan_in_fan_out",
            "target_parameters",
        )
        if any(config.get(key) for key in unsupported) or config.get("bias", "none") != "none":
            raise ValueError("native H3 keep-LoRA does not support adapter bias or advanced PEFT options")
        adapter_name, weights = path.name, path / "adapter_model.safetensors"
    else:
        adapter_name, weights = path.stem, path
    adapter_name = adapter_name if name is None else name
    if not isinstance(adapter_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", adapter_name):
        raise ValueError("native H3 adapter name must be a safe nonempty path component")
    pairs = {}
    with safe_open(weights, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        try:
            alpha = float(config["lora_alpha"] if config is not None else metadata["alpha"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("native H3 adapter requires an explicit positive alpha") from error
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("native H3 adapter alpha must be positive and finite")
        for key in checkpoint.keys():
            match = _PAIR.fullmatch(key)
            if match is None:
                raise ValueError(f"unsupported native H3 adapter tensor: {key}")
            target, side = match.groups()
            target = target.removeprefix("base_model.model.")
            if _TARGET.fullmatch(target) is None:
                raise ValueError(f"unsupported native H3 LoRA target: {target}")
            entry = pairs.setdefault(target, {})
            if side in entry:
                raise ValueError(f"duplicate native H3 LoRA {side}: {target}")
            entry[side] = (key, tuple(checkpoint.get_slice(key).get_shape()))
    specs, ranks = [], set()
    for target, pair in sorted(pairs.items()):
        if set(pair) != {"A", "B"}:
            raise ValueError(f"incomplete native H3 LoRA A/B pair: {target}")
        (a_key, a_shape), (b_key, b_shape) = pair["A"], pair["B"]
        if len(a_shape) != 2 or len(b_shape) != 2 or min(*a_shape, *b_shape) <= 0 or a_shape[0] != b_shape[1]:
            raise ValueError(f"invalid native H3 LoRA A/B shapes: {target}: {a_shape}, {b_shape}")
        ranks.add(a_shape[0])
        specs.append(NativeLoRAWeightPair(target, a_key, b_key, a_shape, b_shape))
    if len(ranks) != 1:
        raise ValueError("native H3 adapters require nonempty A/B pairs with a single rank")
    rank = ranks.pop()
    if config is not None and (type(config.get("r")) is not int or config["r"] != rank):
        raise ValueError("native H3 adapter rank disagrees with its PEFT configuration")
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return NativeLoRASpec(weights, adapter_name, rank, alpha, tuple(specs), digest.hexdigest())


def bind_native_lora(graph, adapter, *, component, policy):
    """Bind one complete component after H3 graph construction and before PTQ.

    A stays unchanged. B receives exactly the same input/output compensation as
    the already-scaled base Linear. No source or graph base weight is merged.
    Every matching target is preflighted before any graph module is replaced.
    The caller exports both components and records their bound target counts.
    """
    import torch
    from safetensors import safe_open
    from torch import nn

    from xhmodel_merak.xh_llm.lora_layer import LoRAStaticLayer

    from .conditioning import validate_fp16_constants
    from .precision import ATTENTION_VALUE_DIVISOR

    if component not in ("text", "dit"):
        raise ValueError("native H3 LoRA component must be text or dit")
    validate_fp16_constants(torch.tensor(adapter.scale), name="native H3 LoRA scale")
    selected = [pair for pair in adapter.pairs if pair.component == component]
    bindings = []
    for pair in selected:
        match = _TARGET.fullmatch(pair.target)
        projection = match["projection"]
        if projection.startswith("attn."):
            suffix = {"attn.to_q": "q", "attn.to_k": "k", "attn.to_v": "v", "attn.to_out.0": "out"}[projection]
            target = f"blocks.{match['index']}.attention.{suffix}"
            divisor = policy.text_residual_scale if component == "text" else policy.attention_output_scale
            multiplier = (
                1 / ATTENTION_VALUE_DIVISOR
                if suffix == "v"
                else ATTENTION_VALUE_DIVISOR / divisor
                if suffix == "out"
                else 1.0
            )
        else:
            suffix = "projection" if projection == "ff.net.0.proj" else "output"
            target = f"blocks.{match['index']}.ffn.{suffix}"
            divisor = policy.text_residual_scale if component == "text" else policy.ffn_output_scale
            multiplier = policy.ffn_product_scale / divisor if suffix == "output" else 1.0
        try:
            linear = graph.get_submodule(target)
        except AttributeError as error:
            raise ValueError(f"native H3 LoRA target is absent from {component} graph: {pair.target}") from error
        if (
            not isinstance(linear, nn.Linear)
            or pair.a_shape[1] != linear.in_features
            or pair.b_shape[0] != linear.out_features
        ):
            raise ValueError(f"native H3 LoRA target shape/type mismatch: {pair.target}")
        if projection == "ff.net.0.proj" and linear.out_features % 2:
            raise ValueError("native H3 SwiGLU LoRA requires equal linear/gate halves")
        bindings.append((pair, target, linear, multiplier))
    replacements = []
    with safe_open(adapter.path, framework="pt", device="cpu") as checkpoint:
        for pair, target, linear, multiplier in bindings:
            a, b = (checkpoint.get_tensor(key).float() for key in (pair.a_key, pair.b_key))
            b = b * multiplier
            if pair.target.endswith("ff.net.0.proj"):
                b[: linear.out_features // 2] /= policy.ffn_product_scale
            validate_fp16_constants(a, name=f"{pair.target} LoRA A")
            validate_fp16_constants(b, name=f"{pair.target} compensated LoRA B")
            if linear.weight.dtype != torch.float32 or linear.weight.device.type != "cpu":
                raise ValueError("native H3 keep-LoRA must be bound to CPU FP32 graphs before PTQ")
            replacements.append((target, LoRAStaticLayer(linear, a, b, scaling=adapter.alpha)))
    for target, replacement in replacements:
        graph.set_submodule(target, replacement.requires_grad_(False))
    return dict(component=component, target_count=len(bindings), targets=[pair.target for pair in selected])
