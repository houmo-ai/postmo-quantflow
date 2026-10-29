"""Bonsai / PQ2 ternary → SSFP4-compatible Phase A conversion.

This is an intermediate compatibility path only:
- expand 2bit codes to int8 mantissas in {-1,0,+1,+2}
- duplicate each K128 FP16 scale into two K64 SSFP4 scales
- emit HF Linear `weight` + `quant_weight` for existing w4a8h0_ssfp export

It does NOT produce compiler DDR PQ2 packed weights (`pq2_weight` stays false).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


PHASE_A_FORMAT = "xh2_pq2_via_ssfp4_compat"
PQ2_CODEBOOK = {0: -1, 1: 0, 2: 1, 3: 2}
DEFAULT_GROUP_K_SRC = 128
DEFAULT_GROUP_K_EXPORTED = 64
DEFAULT_HADAMARD_BLOCK = 1024


def inverse_hadamard_embedding(weight: Tensor, signs: Tensor, *, block_size: int = 1024,
                               rows_per_chunk: int = 1024) -> Tensor:
    """Materialize the pack runtime's FP16 lookup -> inverse FWHT -> signs.

    The resulting ordinary Embedding is in the residual-stream basis. Its
    original ternary codes must not be attached to the transformed weight.
    Chunk rows so a vocabulary-sized matrix does not need an FP32 copy.
    """
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block_size must be a positive power of two")
    if weight.ndim != 2 or weight.shape[1] % block_size:
        raise ValueError("Embedding width must be divisible by the Hadamard block")
    if signs.ndim != 1 or signs.numel() != weight.shape[1]:
        raise ValueError("Embedding signs do not match the embedding width")
    if not ((signs == 1) | (signs == -1)).all():
        raise ValueError("Embedding Hadamard signs must be +/-1")
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    result = torch.empty_like(weight, dtype=torch.float16)
    signs = signs.to(device=weight.device, dtype=torch.float32)
    for start in range(0, weight.shape[0], rows_per_chunk):
        chunk = weight[start:start + rows_per_chunk].half().float()
        values = chunk.reshape(-1, block_size)
        stride = 1
        while stride < block_size:
            pairs = values.reshape(-1, 2, stride)
            left, right = pairs.unbind(1)
            values = torch.stack((left + right, left - right), dim=1)
            stride *= 2
        result[start:start + rows_per_chunk] = (
            values.reshape_as(chunk) / math.sqrt(block_size) * signs
        ).half()
    return result.contiguous()


def convert_bonsai_float_tensor(name: str, tensor: Tensor) -> Tuple[str, Tensor]:
    """Translate MLX tensor layout and normalization conventions into HF's."""
    base, leaf = name.rsplit(".", 1)
    hf_key = f"{bonsai_name_to_hf_name(base)}.{leaf}"
    if leaf == "weight":
        module_leaf = base.rsplit(".", 1)[-1]
        # MLX's standard RMSNorm stores gamma. HF Qwen's ordinary RMSNorm
        # stores gamma - 1. GatedDeltaNet's gated norm already stores gamma.
        if module_leaf in {"input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"} or base == "language_model.model.norm":
            tensor = tensor.float() - 1.0
        elif module_leaf == "conv1d":
            if tensor.ndim != 3 or tensor.shape[-1] != 1:
                raise ValueError(f"Expected MLX depthwise Conv1d [C, kernel, 1]: {name}")
            tensor = tensor.transpose(-1, -2).contiguous()
    return hf_key, tensor.contiguous()


@dataclass(frozen=True)
class BonsaiPackedLinear:
    """One Bonsai packed Linear/Embedding tensor group."""

    bonsai_name: str
    hf_name: str
    weight_u32: Tensor
    scales_k128: Tensor
    biases_k128: Tensor
    signs: Optional[Tensor]
    is_embedding: bool = False


def bonsai_name_to_hf_name(bonsai_name: str) -> str:
    """Map Bonsai MLX tensor namespace onto HF Qwen3.5 / Qwen3.8 keys."""
    name = bonsai_name
    if name.endswith((".weight", ".scales", ".biases", ".signs")):
        name = name.rsplit(".", 1)[0]

    if name.startswith("language_model.lm_head"):
        return "lm_head" + name[len("language_model.lm_head") :]
    if name.startswith("language_model.model."):
        return "model.language_model." + name[len("language_model.model.") :]
    if name.startswith("language_model."):
        # Fallback for unexpected language_model.* leaves.
        return name[len("language_model.") :]
    return name


def unpack_bonsai_u32_codes(weight_u32: Tensor) -> Tensor:
    """Unpack MLX affine U32 words into per-weight 2bit codes, shape [N, K]."""
    if weight_u32.ndim != 2:
        raise ValueError(f"Expected rank-2 U32 weight, got shape={tuple(weight_u32.shape)}")
    if weight_u32.dtype not in (torch.uint32, torch.int32, torch.int64):
        weight_u32 = weight_u32.to(torch.int64)
    rows, words = weight_u32.shape
    codes = torch.empty((rows, words * 16), dtype=torch.int64, device=weight_u32.device)
    src = weight_u32.to(torch.int64)
    for lane in range(16):
        codes[:, lane::16] = (src >> (2 * lane)) & 3
    return codes


def codes_to_pq2_mantissas(codes: Tensor) -> Tensor:
    """Map PQ2/Bonsai codes {0,1,2,3} → mantissas {-1,0,+1,+2} as int8."""
    if codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        codes = codes.to(torch.int64)
    if codes.numel() and (codes.min().item() < 0 or codes.max().item() > 3):
        raise ValueError("Unsupported PQ2/Bonsai codes: expected integer values in [0, 3]")
    mant = codes.to(torch.int16) - 1  # 0/1/2/3 → -1/0/+1/+2
    return mant.to(torch.int8)


def duplicate_k128_scales_to_k64(scales_k128: Tensor) -> Tensor:
    """Duplicate each K128 scale into two adjacent K64 SSFP4 scales."""
    if scales_k128.ndim != 2:
        raise ValueError(f"Expected [N, K/128] scales, got {tuple(scales_k128.shape)}")
    if not torch.isfinite(scales_k128.float()).all():
        raise ValueError("Non-finite K128 scales are not allowed")
    if (scales_k128.float() < 0).any():
        raise ValueError("Negative K128 scales are not allowed for Phase A SSFP4 export")
    return scales_k128.repeat_interleave(2, dim=1)


def reconstruct_from_affine(
    codes: Tensor,
    scales_k128: Tensor,
    biases_k128: Tensor,
    group_k: int = DEFAULT_GROUP_K_SRC,
) -> Tensor:
    """Bonsai affine reconstruction: groups * scales + biases."""
    rows, width = codes.shape
    if width % group_k != 0:
        raise ValueError(f"K={width} is not divisible by group_k={group_k}")
    if scales_k128.shape != (rows, width // group_k):
        raise ValueError(
            f"scales shape {tuple(scales_k128.shape)} != expected {(rows, width // group_k)}"
        )
    if biases_k128.shape != scales_k128.shape:
        raise ValueError(
            f"biases shape {tuple(biases_k128.shape)} != scales shape {tuple(scales_k128.shape)}"
        )
    groups = codes.to(torch.float32).view(rows, -1, group_k)
    recon = (
        groups * scales_k128.to(torch.float32).unsqueeze(-1)
        + biases_k128.to(torch.float32).unsqueeze(-1)
    )
    return recon.reshape(rows, width)


def reconstruct_from_mantissas_k64(mantissas: Tensor, scales_k64: Tensor) -> Tensor:
    """Phase A reconstruction using expanded mantissas and duplicated K64 scales."""
    rows, width = mantissas.shape
    if width % DEFAULT_GROUP_K_EXPORTED != 0:
        raise ValueError(f"K={width} is not divisible by {DEFAULT_GROUP_K_EXPORTED}")
    expected = (rows, width // DEFAULT_GROUP_K_EXPORTED)
    if scales_k64.shape != expected:
        raise ValueError(f"scales_k64 shape {tuple(scales_k64.shape)} != expected {expected}")
    groups = mantissas.to(torch.float32).view(rows, -1, DEFAULT_GROUP_K_EXPORTED)
    return (groups * scales_k64.to(torch.float32).unsqueeze(-1)).reshape(rows, width)


def convert_bonsai_linear_to_ssfp4(
    weight_u32: Tensor,
    scales_k128: Tensor,
    biases_k128: Optional[Tensor] = None,
    *,
    validate_bias: bool = True,
    atol: float = 0.0,
) -> Dict[str, Tensor]:
    """Convert one Bonsai packed linear into SSFP4 Phase A tensors.

    Returns:
      {
        "quant_weight": int8 [N,K] mantissas,
        "weight": float16/float32 [N,K] reconstructed values,
        "w_scale_k64": float16 [N, K/64],
        "codes": int64 [N,K],
      }
    """
    codes = unpack_bonsai_u32_codes(weight_u32)
    rows, width = codes.shape
    if width % DEFAULT_GROUP_K_SRC != 0:
        raise ValueError(f"K={width} must be divisible by {DEFAULT_GROUP_K_SRC}")
    if scales_k128.shape != (rows, width // DEFAULT_GROUP_K_SRC):
        raise ValueError(
            f"scales shape mismatch: {tuple(scales_k128.shape)} vs {(rows, width // DEFAULT_GROUP_K_SRC)}"
        )

    if biases_k128 is None:
        biases_k128 = -scales_k128
    elif validate_bias:
        if biases_k128.shape != scales_k128.shape:
            raise ValueError("biases/scales shape mismatch")
        # Bonsai affine uses biases = -scales for ternary codes {0,1,2}.
        if not torch.allclose(
            biases_k128.float(), (-scales_k128).float(), atol=1e-6, rtol=0.0
        ):
            raise ValueError("Expected Bonsai biases == -scales for ternary affine pack")

    mant = codes_to_pq2_mantissas(codes)
    scales_k64 = duplicate_k128_scales_to_k64(scales_k128)
    weight = reconstruct_from_mantissas_k64(mant, scales_k64)

    # Cross-check against original affine unpack when biases are provided.
    affine = reconstruct_from_affine(codes, scales_k128, biases_k128)
    max_diff = (weight - affine).abs().max().item()
    if max_diff > atol:
        raise ValueError(
            f"Phase A reconstruct mismatch vs Bonsai affine: max_abs_diff={max_diff} atol={atol}"
        )

    return {
        "quant_weight": mant.contiguous(),
        "weight": weight.to(torch.float16).contiguous(),
        "w_scale_k64": scales_k64.to(torch.float16).contiguous(),
        "codes": codes.contiguous(),
    }


def build_phase_a_metadata(
    *,
    source: str = "bonsai_ternary",
    hadamard_block_size: int = DEFAULT_HADAMARD_BLOCK,
    num_modules: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "format": PHASE_A_FORMAT,
        "version": 2,
        "source": source,
        "ssfp": {"man_bit": 4, "nshare": DEFAULT_GROUP_K_EXPORTED, "scale_dtype": "float16"},
        "pq2_semantic": {
            "codebook": {str(k): v for k, v in PQ2_CODEBOOK.items()},
            "group_k_src": DEFAULT_GROUP_K_SRC,
            "group_k_exported": DEFAULT_GROUP_K_EXPORTED,
            "scale_replication": "each_k128_duplicated_to_two_k64",
        },
        "final_target": "xh2_pq2_0",
        "pq2_weight": False,
        "hadamard": {
            "applied_to_weights": True,
            "runtime_activation_transform_required": True,
            "block_size": int(hadamard_block_size),
            "implementation": "matmul_weight",
            "transform": "normalized-sylvester-walsh-hadamard",
            "axis": "input-last-dimension",
        },
    }
    if num_modules is not None:
        meta["num_modules"] = int(num_modules)
    if extra:
        meta.update(dict(extra))
    return meta


def _load_hadamard_block_size(bonsai_dir: Path) -> int:
    hadamard_json = bonsai_dir / "hadamard.json"
    if not hadamard_json.exists():
        return DEFAULT_HADAMARD_BLOCK
    data = json.loads(hadamard_json.read_text())
    return int(data.get("prism.hadamard.block_size", DEFAULT_HADAMARD_BLOCK))


def iter_bonsai_packed_modules(
    tensors: Mapping[str, Tensor],
    *,
    include_names: Optional[Sequence[str]] = None,
) -> List[BonsaiPackedLinear]:
    """Collect packed Bonsai modules that have weight/scales/biases."""
    keyset = set(tensors.keys())
    include = None if include_names is None else set(include_names)
    modules: List[BonsaiPackedLinear] = []

    for key in sorted(keyset):
        if not key.endswith(".weight"):
            continue
        base = key[: -len(".weight")]
        if not base.startswith("language_model."):
            continue
        scales_key = base + ".scales"
        biases_key = base + ".biases"
        if scales_key not in keyset or biases_key not in keyset:
            continue
        if include is not None and base not in include and key not in include:
            continue

        signs_key = base + ".signs"
        signs = tensors.get(signs_key)
        is_embedding = base.endswith("embed_tokens")
        modules.append(
            BonsaiPackedLinear(
                bonsai_name=base,
                hf_name=bonsai_name_to_hf_name(base),
                weight_u32=tensors[key],
                scales_k128=tensors[scales_key],
                biases_k128=tensors[biases_key],
                signs=signs,
                is_embedding=is_embedding,
            )
        )
    return modules


def convert_bonsai_pack_to_ssfp4_state_dict(
    bonsai_dir: str | Path,
    *,
    include_names: Optional[Sequence[str]] = None,
    include_float_tensors: bool = True,
    validate_bias: bool = True,
    device: str | torch.device = "cpu",
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    """Convert a Bonsai MLX pack into HF-compatible Phase A state_dict + metadata.

    The returned state_dict contains:
    - reconstructed float `*.weight` for packed modules
    - int8 `*.quant_weight` buffers for packed modules
    - optional float non-quant tensors (norms / A_log / conv / etc.) remapped to HF names
    - optional `*.hadamard_signs` for online MatMul composition
    """
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise ImportError("safetensors is required to read Bonsai packs") from exc

    bonsai_dir = Path(bonsai_dir)
    weight_path = bonsai_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(weight_path)

    tensors = load_file(str(weight_path), device=str(device))
    block_size = _load_hadamard_block_size(bonsai_dir)
    modules = iter_bonsai_packed_modules(tensors, include_names=include_names)
    if not modules:
        raise RuntimeError(f"No packed Bonsai modules found under {bonsai_dir}")

    state: Dict[str, Tensor] = {}
    module_records: List[Dict[str, Any]] = []
    # All packed modules in the pack (not only the selected subset). Needed so
    # --include-name smoke runs do not accidentally copy raw U32 packed weights
    # from non-selected modules into the HF export state_dict.
    all_packed_modules = iter_bonsai_packed_modules(tensors, include_names=None)
    all_packed_bases = {m.bonsai_name for m in all_packed_modules}

    for module in modules:
        converted = convert_bonsai_linear_to_ssfp4(
            module.weight_u32,
            module.scales_k128,
            module.biases_k128,
            validate_bias=validate_bias,
        )
        if module.is_embedding:
            if module.signs is None:
                raise ValueError(f"Missing inverse-embedding signs: {module.bonsai_name}")
            state[f"{module.hf_name}.weight"] = inverse_hadamard_embedding(
                converted["weight"], module.signs, block_size=block_size)
        else:
            state[f"{module.hf_name}.weight"] = converted["weight"]
            state[f"{module.hf_name}.quant_weight"] = converted["quant_weight"]
        # Keep duplicated scales for debugging / Phase B packing; export path
        # currently re-derives SSFP scales from weight/quant_weight.
        if not module.is_embedding:
            state[f"{module.hf_name}.w_scale_k64"] = converted["w_scale_k64"]
        if module.signs is not None:
            state[f"{module.hf_name}.hadamard_signs"] = module.signs.to(torch.float32).contiguous()

        module_records.append(
            {
                "bonsai_name": module.bonsai_name,
                "hf_name": module.hf_name,
                "shape": list(converted["quant_weight"].shape),
                "is_embedding": module.is_embedding,
                "has_signs": module.signs is not None,
                "embedding_inverse_applied": module.is_embedding,
                "code_values": [code for code in range(4) if (converted["codes"] == code).any().item()],
            }
        )

    float_tensor_count = 0
    if include_float_tensors:
        for key, tensor in tensors.items():
            if not key.startswith("language_model."):
                continue
            if key.endswith((".scales", ".biases", ".signs")):
                continue
            if key.endswith(".weight"):
                base = key[: -len(".weight")]
                if base in all_packed_bases:
                    continue
            hf_key, converted_float = convert_bonsai_float_tensor(key, tensor)
            state[hf_key] = converted_float
            float_tensor_count += 1

    meta = build_phase_a_metadata(
        source="bonsai_ternary_mlx_pack",
        hadamard_block_size=block_size,
        num_modules=len(modules),
        extra={
            "bonsai_dir": str(bonsai_dir),
            "modules": module_records,
            "complete_language_model": include_names is None and include_float_tensors,
            "float_tensor_count": float_tensor_count,
            "float_tensors_converted": include_float_tensors,
            "embedding_inverse_applied": any(m.is_embedding for m in modules),
            "note": (
                "Phase A SSFP4 compatibility only; pq2_weight=false; "
                "not the final XH2 PQ2_0 DDR packed format."
            ),
        },
    )
    return state, meta


def split_phase_a_export_tensors(
    state_dict: Mapping[str, Tensor],
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Split HF-loadable tensors from Phase A helper / Hadamard artifacts.

    `load_quant_weight()` only understands Linear `weight` + `quant_weight`
    (plus ordinary float parameters). Helper tensors such as `w_scale_k64`,
    `hadamard_signs`, and dense Hadamard MatMul banks should be stored
    separately so export does not warn on unexpected keys.
    """
    export_sd: Dict[str, Tensor] = {}
    aux_sd: Dict[str, Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith("hadamard_matmul."):
            aux_sd[key] = tensor
        elif key.endswith((".w_scale_k64", ".hadamard_signs", ".codes")):
            aux_sd[key] = tensor
        else:
            export_sd[key] = tensor
    return export_sd, aux_sd


def save_phase_a_checkpoint(
    state_dict: Mapping[str, Tensor],
    metadata: Mapping[str, Any],
    output_dir: str | Path,
    *,
    filename: str = "quant_weight.pt",
    aux_filename: str = "phase_a_aux.pt",
    split_aux: bool = True,
) -> Path:
    """Save Phase A state_dict + metadata JSON for zoo load_quant_weight()."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_sd: Mapping[str, Tensor]
    if split_aux:
        export_sd, aux_sd = split_phase_a_export_tensors(state_dict)
        if aux_sd:
            torch.save(aux_sd, output_dir / aux_filename)
            metadata = dict(metadata)
            metadata["aux_file"] = aux_filename
            metadata["aux_keys"] = sorted(aux_sd.keys())
    else:
        export_sd = state_dict

    ckpt_path = output_dir / filename
    # load_quant_weight expects a flat state_dict with *.weight and *.quant_weight.
    torch.save(dict(export_sd), ckpt_path)
    meta_path = output_dir / "phase_a_metadata.json"
    meta_path.write_text(json.dumps(dict(metadata), indent=2, ensure_ascii=False))
    return ckpt_path
