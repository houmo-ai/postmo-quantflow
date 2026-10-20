from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open


_PACKED_SUFFIXES = ("qweight", "qzeros", "scales", "g_idx")
_EXPERT_LINEAR_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)$"
)


@dataclass(frozen=True)
class AutoRoundImportStats:
    packed_modules: int
    w4_modules: int
    w8_modules: int
    skipped_modules: int
    source_group_size: int


def build_safetensors_weight_map(model_dir: str | Path) -> dict[str, Path]:
    model_dir = Path(model_dir)
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        data = json.loads(index_file.read_text(encoding="utf-8"))
        return {name: model_dir / relative_path for name, relative_path in data.get("weight_map", {}).items()}

    model_file = model_dir / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"No safetensors checkpoint or index found under {model_dir}")
    with safe_open(model_file, framework="pt", device="cpu") as handle:
        return {name: model_file for name in handle.keys()}


def load_checkpoint_tensor(
    model_dir: str | Path,
    tensor_name: str,
    weight_map: dict[str, Path] | None = None,
) -> torch.Tensor | None:
    weight_map = weight_map or build_safetensors_weight_map(model_dir)
    tensor_file = weight_map.get(tensor_name)
    if tensor_file is None:
        return None
    with safe_open(tensor_file, framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name)


def infer_gptq_bits(qweight: torch.Tensor, in_features: int) -> int:
    matches = [bits for bits in (4, 8) if qweight.shape[0] * (32 // bits) == in_features]
    if len(matches) != 1:
        raise ValueError(
            "Cannot infer GPTQ packed bit width: "
            f"qweight.shape={tuple(qweight.shape)}, in_features={in_features}, matches={matches}"
        )
    return matches[0]


def unpack_gptq_v2_quant_weight(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    *,
    bits: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Unpack GPTQ-v2 integer codes as signed ``[out_features, in_features]`` values.

    GPTQ-v2 stores zero points directly.  In particular, this routine must not
    apply the legacy AutoGPTQ ``qzeros + 1`` convention.
    """

    if bits not in (4, 8):
        raise ValueError(f"Only 4-bit and 8-bit GPTQ-v2 weights are supported, got bits={bits}")
    pack_factor = 32 // bits
    if qweight.shape[0] * pack_factor != in_features:
        raise ValueError(
            f"qweight shape {tuple(qweight.shape)} does not cover in_features={in_features} at {bits} bits"
        )
    if qweight.shape[1] != out_features:
        raise ValueError(
            f"qweight shape {tuple(qweight.shape)} does not match out_features={out_features}"
        )
    if g_idx.numel() != in_features:
        raise ValueError(f"g_idx has {g_idx.numel()} elements, expected {in_features}")

    mask = (1 << bits) - 1
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    packed_zeros = qzeros.to(torch.int32)
    unpacked_zeros = torch.empty(
        qzeros.shape[0],
        out_features,
        dtype=torch.int16,
        device=qzeros.device,
    )
    for lane, shift in enumerate(shifts):
        unpacked_zeros[:, lane::pack_factor] = ((packed_zeros >> shift) & mask).to(torch.int16)

    group_indices = g_idx.to(torch.long)
    if group_indices.min() < 0 or group_indices.max() >= unpacked_zeros.shape[0]:
        raise ValueError(
            f"g_idx range [{int(group_indices.min())}, {int(group_indices.max())}] exceeds "
            f"available zero-point groups={unpacked_zeros.shape[0]}"
        )

    packed_weight = qweight.to(torch.int32)
    quant_weight = torch.empty(out_features, in_features, dtype=torch.int8, device=qweight.device)
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    for lane, shift in enumerate(shifts):
        input_indices = torch.arange(lane, in_features, pack_factor, device=qweight.device)
        unsigned = ((packed_weight >> shift) & mask).to(torch.int16)
        signed = unsigned - unpacked_zeros[group_indices[input_indices]]
        observed_min = int(signed.min())
        observed_max = int(signed.max())
        if observed_min < qmin or observed_max > qmax:
            raise ValueError(
                f"GPTQ-v2 signed codes exceed {bits}-bit range [{qmin}, {qmax}]: "
                f"observed [{observed_min}, {observed_max}]"
            )
        quant_weight[:, input_indices] = signed.transpose(0, 1).to(torch.int8)
    return quant_weight.contiguous()


def convert_autoround_to_xh_ssfp(
    quant_weight: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert AutoRound GPTQ-v2 groups into XH2 SSFP integer codes and positive scales.

    GPTQ-v2 permits negative group scales, while XH2 SSFP requires non-negative
    scales.  Build the target representation directly from the AutoRound codes
    and scales; never use or overwrite the module's floating ``weight``.
    """
    if quant_weight.ndim != 2:
        raise ValueError(f"quant_weight must be rank 2, got shape={tuple(quant_weight.shape)}")
    out_features, in_features = quant_weight.shape
    groups = (in_features + group_size - 1) // group_size
    if tuple(scales.shape) != (groups, out_features):
        raise ValueError(
            f"AutoRound scales shape {tuple(scales.shape)} does not match expected {(groups, out_features)}"
        )
    expected_g_idx = torch.arange(in_features, dtype=torch.long, device=g_idx.device) // group_size
    if not torch.equal(g_idx.to(torch.long), expected_g_idx):
        raise ValueError(
            f"AutoRound g_idx must describe contiguous groups of size {group_size} for XH2 SSFP conversion"
        )

    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    if not torch.isfinite(scales).all():
        raise ValueError("AutoRound scales must be finite for XH2 SSFP conversion")
    group_signs = torch.sign(scales).to(torch.int8)
    group_signs[group_signs == 0] = 1
    input_signs = group_signs[g_idx.to(torch.long)].transpose(0, 1)
    signed_codes = quant_weight.to(torch.int16) * input_signs.to(torch.int16)
    if (signed_codes < qmin).any() or (signed_codes > qmax).any():
        raise ValueError(
            f"AutoRound negative scale sign absorption exceeds W{bits} range [{qmin}, {qmax}]; "
            "refusing to saturate checkpoint integer codes"
        )
    target_codes = signed_codes.to(torch.int8)
    target_scales = scales.abs().contiguous()
    return target_codes.contiguous(), target_scales


def _validate_ssfp_scale_recovery(weight, codes, scales, group_size, name):
    """Reject producer payloads the SDK's float/code SSFP interface cannot retain.

    SDK SSFP uses FP16 group scales. Ratio recovery may lose the source scale
    after a native FP16/BF16 weight cast, even when float and packed agree.
    Validate this conversion explicitly instead of silently requantizing it.
    """
    groups, outputs = scales.shape
    padded = groups * group_size
    if padded != codes.shape[1]:
        weight = torch.nn.functional.pad(weight, (0, padded - codes.shape[1]))
        codes = torch.nn.functional.pad(codes, (0, padded - codes.shape[1]))
    grouped_codes = codes.to(weight.device).reshape(outputs, groups, group_size)
    grouped_weight = weight.reshape(outputs, groups, group_size)
    nonzero = grouped_codes != 0
    ratios = (grouped_weight / grouped_codes).half().abs()
    ratios = torch.where(nonzero, ratios, 0)
    recovered = ratios.amax(dim=-1)
    expected = scales.to(device=weight.device, dtype=torch.float16).T
    active = nonzero.any(dim=-1)
    if not torch.equal(recovered[active], expected[active]) or not torch.equal(scales.half().to(scales.dtype), scales):
        raise ValueError(
            f"AutoRound {name} scales cannot be recovered exactly through the SDK SSFP float/code interface; "
            "use a checkpoint with representable FP16 scales instead of silently changing its quantization"
        )
    if (expected[active] <= 0).any():
        raise ValueError(f"AutoRound {name} nonzero codes require positive SSFP scales")
    effective_scale = torch.where(recovered == 0, 1, recovered).unsqueeze(-1)
    recovered_codes = (grouped_weight / effective_scale).round()
    if not torch.equal(recovered_codes, grouped_codes.to(recovered_codes.dtype)):
        raise ValueError(f"AutoRound {name} integer codes cannot be recovered exactly by SDK SSFP conversion")


def _has_checkpoint_lm_head(weight_map):
    packed_names = [f"lm_head.{suffix}" for suffix in _PACKED_SUFFIXES]
    packed_present = [name in weight_map for name in packed_names]
    if any(packed_present) and not all(packed_present):
        missing = [name for name, present in zip(packed_names, packed_present, strict=True) if not present]
        raise ValueError(f"Incomplete packed lm_head: missing {missing}")
    return "lm_head.weight" in weight_map or all(packed_present)


def _checkpoint_name_candidates(checkpoint_name: str) -> tuple[str, ...]:
    """Map composite-checkpoint names onto text-only CausalLM module paths."""
    names = [checkpoint_name]
    for prefix in ("model.language_model.", "language_model.", "model."):
        if checkpoint_name.startswith(prefix):
            names.append(checkpoint_name[len(prefix) :])
    names.extend(f"model.{name}" for name in tuple(names) if not name.startswith("model."))
    return tuple(dict.fromkeys(names))


def _resolve_checkpoint_module(model: nn.Module, checkpoint_name: str) -> nn.Module | None:
    for candidate in _checkpoint_name_candidates(checkpoint_name):
        try:
            return model.get_submodule(candidate)
        except AttributeError:
            continue
    return None


def _resolve_checkpoint_tensor_target(
    model: nn.Module,
    checkpoint_name: str,
) -> torch.Tensor | None:
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for candidate in _checkpoint_name_candidates(checkpoint_name):
        target = parameters.get(candidate)
        if target is None:
            target = buffers.get(candidate)
        if target is not None:
            return target
    return None


def import_composite_text_weights(
    model: nn.Module,
    model_dir: str | Path,
    weight_map: dict[str, Path] | None = None,
) -> tuple[int, int]:
    """Load non-packed text tensors from a composite VLM checkpoint into a CausalLM."""
    weight_map = weight_map or build_safetensors_weight_map(model_dir)
    packed_suffixes = tuple(f".{suffix}" for suffix in _PACKED_SUFFIXES)
    names_by_file: dict[Path, list[str]] = {}
    for name, tensor_file in weight_map.items():
        if not name.endswith(packed_suffixes):
            names_by_file.setdefault(tensor_file, []).append(name)

    loaded = 0
    skipped = 0
    for tensor_file, names in names_by_file.items():
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            for name in names:
                target = _resolve_checkpoint_tensor_target(model, name)
                if target is None:
                    skipped += 1
                    continue
                source = handle.get_tensor(name)
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        f"Composite checkpoint tensor {name} has shape {tuple(source.shape)}, "
                        f"expected {tuple(target.shape)}"
                    )
                with torch.no_grad():
                    target.copy_(source.to(device=target.device, dtype=target.dtype))
                loaded += 1
    return loaded, skipped


def _resolve_packed_expert_target(
    model: nn.Module,
    checkpoint_name: str,
) -> tuple[nn.Module, int, str, int, int] | None:
    normalized_name = next(
        (name for name in _checkpoint_name_candidates(checkpoint_name) if _EXPERT_LINEAR_RE.fullmatch(name)),
        None,
    )
    if normalized_name is None:
        return None
    match = _EXPERT_LINEAR_RE.fullmatch(normalized_name)
    if match is None:
        return None
    layer_idx = int(match.group("layer"))
    expert_idx = int(match.group("expert"))
    projection = match.group("projection")
    experts = _resolve_checkpoint_module(model, f"layers.{layer_idx}.mlp.experts")
    if experts is None or not hasattr(experts, "gate_up_proj") or not hasattr(experts, "down_proj"):
        return None

    gate_up_proj = experts.gate_up_proj
    down_proj = experts.down_proj
    num_experts = int(gate_up_proj.shape[0])
    intermediate_size = int(gate_up_proj.shape[1] // 2)
    hidden_size = int(gate_up_proj.shape[2])
    if expert_idx >= num_experts:
        raise IndexError(f"Expert index {expert_idx} exceeds num_experts={num_experts} for {checkpoint_name}")
    if projection == "down_proj":
        out_features, in_features = int(down_proj.shape[1]), int(down_proj.shape[2])
    else:
        out_features, in_features = intermediate_size, hidden_size
    return experts, expert_idx, projection, out_features, in_features


def _store_packed_expert_quant_weight(
    experts: nn.Module,
    expert_idx: int,
    projection: str,
    quant_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> None:
    buffer_name = f"{projection}_quant_weight"
    target = getattr(experts, buffer_name, None)
    num_experts = int(experts.gate_up_proj.shape[0])
    expected_shape = (num_experts, *quant_weight.shape)
    if target is None:
        target = torch.empty(expected_shape, dtype=torch.int8, device=quant_weight.device)
        if buffer_name in experts._buffers:
            experts._buffers[buffer_name] = target
        else:
            experts.register_buffer(buffer_name, target, persistent=False)
    elif tuple(target.shape) != expected_shape:
        raise ValueError(
            f"Packed expert quant buffer {buffer_name} has shape {tuple(target.shape)}, expected {expected_shape}"
        )
    target[expert_idx].copy_(quant_weight)
    scale_buffer_name = f"{projection}_quant_scale"
    scale_target = getattr(experts, scale_buffer_name, None)
    expected_scale_shape = (num_experts, *quant_scale.shape)
    if scale_target is None:
        scale_target = torch.empty(expected_scale_shape, dtype=quant_scale.dtype, device=quant_scale.device)
        if scale_buffer_name in experts._buffers:
            experts._buffers[scale_buffer_name] = scale_target
        else:
            experts.register_buffer(scale_buffer_name, scale_target, persistent=False)
    elif tuple(scale_target.shape) != expected_scale_shape:
        raise ValueError(
            f"Packed expert scale buffer {scale_buffer_name} has shape {tuple(scale_target.shape)}, "
            f"expected {expected_scale_shape}"
        )
    scale_target[expert_idx].copy_(quant_scale)
    expert_bits = getattr(experts, "_autoround_expert_bits", None)
    if expert_bits is None:
        experts._autoround_expert_bits = bits
    elif expert_bits != bits:
        raise ValueError(f"Mixed routed-expert bit widths are unsupported: saw {expert_bits} and {bits}")
    experts._autoround_group_size = group_size
    experts._xhquant_weight_origin = "autoround-gptq-v2"


def import_autoround_gptq_v2_weights(model: nn.Module, model_dir: str | Path) -> AutoRoundImportStats:
    """Attach packed AutoRound integer codes to their corresponding HF Linear modules."""

    from xhquant.api import get_xhquant_logger

    model_dir = Path(model_dir)
    logger = get_xhquant_logger()
    config_data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    quant_config: dict[str, Any] = config_data.get("quantization_config", {})
    checkpoint_format = str(quant_config.get("checkpoint_format", "")).lower()
    if checkpoint_format != "gptq_v2":
        raise ValueError(f"Expected AutoRound checkpoint_format='gptq_v2', got {checkpoint_format!r}")
    source_group_size = int(quant_config.get("group_size", 0))
    if source_group_size <= 0:
        raise ValueError(f"Invalid AutoRound source group_size={source_group_size}")

    weight_map = build_safetensors_weight_map(model_dir)
    loaded_plain, skipped_plain = import_composite_text_weights(model, model_dir, weight_map)
    logger.info(
        "Imported composite checkpoint non-packed tensors: loaded=%d, skipped_non_text=%d",
        loaded_plain,
        skipped_plain,
    )
    packed_bases = sorted(name[: -len(".qweight")] for name in weight_map if name.endswith(".qweight"))
    bases_by_file: dict[Path, list[str]] = {}
    for base_name in packed_bases:
        bases_by_file.setdefault(weight_map[f"{base_name}.qweight"], []).append(base_name)
    packed_modules = 0
    w4_modules = 0
    w8_modules = 0
    skipped_modules = 0

    for shard_file, shard_bases in bases_by_file.items():
        logger.info(
            "Importing %d AutoRound packed modules from shard %s",
            len(shard_bases),
            shard_file.name,
        )
        with safe_open(shard_file, framework="pt", device="cpu") as handle:
            for base_name in shard_bases:
                required_names = {suffix: f"{base_name}.{suffix}" for suffix in _PACKED_SUFFIXES}
                missing = [name for name in required_names.values() if name not in weight_map]
                if missing:
                    raise KeyError(f"Incomplete AutoRound packed module {base_name}: missing {missing}")
                module = _resolve_checkpoint_module(model, base_name)
                expert_target = (
                    None if isinstance(module, nn.Linear) else _resolve_packed_expert_target(model, base_name)
                )
                if not isinstance(module, nn.Linear) and expert_target is None:
                    skipped_modules += 1
                    continue
                if isinstance(module, nn.Linear):
                    out_features, in_features = module.out_features, module.in_features
                else:
                    assert expert_target is not None
                    _, _, _, out_features, in_features = expert_target

                tensors = {}
                for suffix, tensor_name in required_names.items():
                    if weight_map[tensor_name] == shard_file:
                        tensors[suffix] = handle.get_tensor(tensor_name)
                    else:
                        tensor = load_checkpoint_tensor(model_dir, tensor_name, weight_map)
                        if tensor is None:
                            raise RuntimeError(f"Failed to load cross-shard tensor {tensor_name}")
                        tensors[suffix] = tensor
                qweight = tensors["qweight"]
                qzeros = tensors["qzeros"]
                scales = tensors["scales"]
                g_idx = tensors["g_idx"]
                bits = infer_gptq_bits(qweight, in_features)
                if scales.shape[0] != qzeros.shape[0] or scales.shape[1] != out_features:
                    raise ValueError(
                        f"AutoRound scale/zero shape mismatch for {base_name}: "
                        f"scales={tuple(scales.shape)}, qzeros={tuple(qzeros.shape)}, out_features={out_features}"
                    )
                quant_weight = unpack_gptq_v2_quant_weight(
                    qweight,
                    qzeros,
                    g_idx,
                    bits=bits,
                    out_features=out_features,
                    in_features=in_features,
                )
                quant_weight, scales = convert_autoround_to_xh_ssfp(
                    quant_weight,
                    scales,
                    g_idx,
                    bits=bits,
                    group_size=source_group_size,
                )
                if isinstance(module, nn.Linear):
                    target_weight = module.weight
                else:
                    assert expert_target is not None
                    experts, expert_idx, projection, _, _ = expert_target
                    if projection == "down_proj":
                        target_weight = experts.down_proj[expert_idx]
                    else:
                        half = experts.gate_up_proj.shape[1] // 2
                        offset = 0 if projection == "gate_proj" else half
                        target_weight = experts.gate_up_proj[expert_idx, offset:offset + half]
                # The current SDK consumes external codes together with float
                # weight, deriving SSFP scales from their ratio. It does not
                # consume the auxiliary quant_scale buffer. Keep the native
                # weight tied to the producer payload, or fail before export.
                floating_weight = (
                    quant_weight.float() * scales.float()[g_idx.long()].transpose(0, 1)
                ).to(device=target_weight.device, dtype=target_weight.dtype)
                if f"{base_name}.weight" in weight_map:
                    if not torch.equal(target_weight, floating_weight):
                        raise ValueError(
                            f"AutoRound {base_name} has inconsistent floating and packed weights; "
                            "the SDK must not derive a different scale from a second weight representation"
                        )
                else:
                    with torch.no_grad():
                        target_weight.copy_(floating_weight)
                _validate_ssfp_scale_recovery(
                    target_weight, quant_weight, scales, source_group_size, base_name
                )
                if isinstance(module, nn.Linear):
                    if "quant_weight" in module._buffers:
                        module._buffers["quant_weight"] = quant_weight
                    else:
                        module.register_buffer("quant_weight", quant_weight, persistent=False)
                    if "quant_scale" in module._buffers:
                        module._buffers["quant_scale"] = scales.contiguous()
                    else:
                        module.register_buffer("quant_scale", scales.contiguous(), persistent=False)
                    module._autoround_bits = bits
                    module.logical_bits = bits
                    module._autoround_group_size = source_group_size
                    module._xhquant_weight_origin = "autoround-gptq-v2"
                else:
                    assert expert_target is not None
                    experts, expert_idx, projection, _, _ = expert_target
                    _store_packed_expert_quant_weight(
                        experts,
                        expert_idx,
                        projection,
                        quant_weight,
                        scales.contiguous(),
                        bits=bits,
                        group_size=source_group_size,
                    )
                packed_modules += 1
                w4_modules += int(bits == 4)
                w8_modules += int(bits == 8)

    if packed_modules == 0:
        raise RuntimeError(
            "AutoRound GPTQ-v2 checkpoint contained packed tensors, but none were attached to HF nn.Linear modules"
        )
    return AutoRoundImportStats(
        packed_modules=packed_modules,
        w4_modules=w4_modules,
        w8_modules=w8_modules,
        skipped_modules=skipped_modules,
        source_group_size=source_group_size,
    )


def restore_untied_lm_head(
    model: nn.Module,
    checkpoint_dir: str | Path,
    lm_head_source: str | Path | None,
) -> bool:
    """Restore an independent output head when an AutoRound checkpoint omitted it."""

    checkpoint_map = build_safetensors_weight_map(checkpoint_dir)
    if _has_checkpoint_lm_head(checkpoint_map):
        return False

    config = getattr(model, "config", None)
    if bool(getattr(config, "tie_word_embeddings", False)):
        return False
    if lm_head_source is None:
        raise RuntimeError(
            "The quantized checkpoint omits lm_head.weight while tie_word_embeddings=false. "
            "Set export.model.lm_head_source to the unquantized base checkpoint; refusing to export a randomly "
            "initialized output head."
        )

    source_map = build_safetensors_weight_map(lm_head_source)
    source_weight = load_checkpoint_tensor(lm_head_source, "lm_head.weight", source_map)
    if source_weight is None:
        raise FileNotFoundError(f"lm_head.weight was not found in explicit lm_head_source={lm_head_source}")
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, nn.Linear):
        raise TypeError(f"Expected model.lm_head to be nn.Linear, got {type(lm_head)}")
    if tuple(source_weight.shape) != tuple(lm_head.weight.shape):
        raise ValueError(
            f"lm_head shape mismatch: source={tuple(source_weight.shape)}, model={tuple(lm_head.weight.shape)}"
        )
    with torch.no_grad():
        lm_head.weight.copy_(source_weight.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype))
    lm_head._xhquant_weight_origin = "explicit-base-lm-head"
    return True


@contextmanager
def checkpoint_overlay_with_untied_lm_head(
    checkpoint_dir: str | Path,
    lm_head_source: str | Path | None,
):
    """Yield a symlink-only checkpoint view with ``lm_head.weight`` indexed at load time."""

    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_map = build_safetensors_weight_map(checkpoint_dir)
    if _has_checkpoint_lm_head(checkpoint_map):
        yield checkpoint_dir
        return
    config_data = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    if bool(config_data.get("tie_word_embeddings", False)):
        yield checkpoint_dir
        return
    if lm_head_source is None:
        raise RuntimeError(
            "The quantized checkpoint omits lm_head.weight while tie_word_embeddings=false. "
            "Set export.model.lm_head_source to the unquantized base checkpoint; refusing to load a randomly "
            "initialized output head."
        )

    source_dir = Path(lm_head_source).resolve()
    source_map = build_safetensors_weight_map(source_dir)
    source_head_file = source_map.get("lm_head.weight")
    if source_head_file is None:
        raise FileNotFoundError(f"lm_head.weight was not found in explicit lm_head_source={source_dir}")

    index_file = checkpoint_dir / "model.safetensors.index.json"
    if not index_file.is_file():
        raise FileNotFoundError(
            "An indexed AutoRound checkpoint is required to overlay an external lm_head.weight: "
            f"missing {index_file}"
        )
    index_data = json.loads(index_file.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="qwen35-autoround-lm-head-") as temporary_dir:
        overlay_dir = Path(temporary_dir)
        for source_path in checkpoint_dir.iterdir():
            if source_path.name == index_file.name:
                continue
            os.symlink(source_path, overlay_dir / source_path.name)

        head_link_name = f"lm-head-source-{source_head_file.name}"
        os.symlink(source_head_file, overlay_dir / head_link_name)
        index_data.setdefault("weight_map", {})["lm_head.weight"] = head_link_name
        (overlay_dir / index_file.name).write_text(
            json.dumps(index_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        yield overlay_dir
