"""GPTQModel recipe for PaddlePaddle/PaddleOCR-VL-1.6.

Language and Vision encoder blocks are quantized on separate fresh model
instances and merged into one reloadable GPTQModel HF artifact. This avoids
running a second calibration pass through already packed QuantLinear modules.
The patch embedding and cross-modal projector remain floating point.
"""

from __future__ import annotations

import gc
import json
import random
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


_MODEL_TYPE = "paddleocr_vl"
_RECIPE_NAME = "recipes.paddleocr_vl.quantize_paddleocr_vl"
_ARTIFACT_FORMAT = "gptqmodel_hf"
_SUPPORTED_METHODS = {"autoround", "auto_round", "gptq"}
_SUPPORTED_ROTATIONS = {None, "hadamard"}
_VISION_ENCODER_PREFIX = "model.visual.vision_model.encoder.layers."
_SUPPORTED_VISUAL_KEYS = {
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
}
_PROCESSOR_ASSET_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
)
_PROTECTED_OUTPUT_NAMES = {
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "quantize_config.json",
}
_PROTECTED_OUTPUT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
)


def _base_qmodel() -> type:
    from gptqmodel.models.base import BaseQModel

    return BaseQModel


@dataclass(frozen=True)
class PaddleOCRVLQuantRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    model_family: str
    precision: dict[str, Any]
    calibration: dict[str, Any]
    provenance: dict[str, Any]


class _PaddleOCRVLLoader:
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
        from transformers.models.paddleocr_vl.modeling_paddleocr_vl import (
            PaddleOCRVLForConditionalGeneration,
        )

        return PaddleOCRVLForConditionalGeneration.from_pretrained(*args, **kwargs)

    @classmethod
    def from_config(cls, *args: Any, **kwargs: Any) -> Any:
        from transformers.models.paddleocr_vl.modeling_paddleocr_vl import (
            PaddleOCRVLForConditionalGeneration,
        )

        kwargs.pop("trust_remote_code", None)
        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", None))
        model = PaddleOCRVLForConditionalGeneration(*args, **kwargs)
        if dtype is not None:
            model = model.to(dtype=dtype)
        return model


def quantize_paddleocr_vl(
    *,
    model_dir: str,
    output_dir: str,
    calibration_data: Iterable[Mapping[str, Any]] | str | Path,
    method: str = "autoround",
    bits: int = 8,
    group_size: int = 64,
    iters: int = 200,
    rotation: str | None = None,
    quantize_vision: bool = False,
    vision_rotation: str | None = None,
    vision_bits: int | None = None,
    vision_group_size: int | None = None,
    vision_iters: int | None = None,
    device: str = "cuda:0",
    seed: int = 42,
    artifact_format: str = _ARTIFACT_FORMAT,
    batch_size: int = 1,
    sym: bool = True,
    desc_act: bool = False,
    damp_percent: float | None = None,
    trust_remote_code: bool = False,
    offload_to_disk: bool = False,
    offload_to_disk_path: str | None = None,
    dry_run: bool = False,
    **quantize_kwargs: Any,
) -> PaddleOCRVLQuantRecipeResult:
    """Quantize PaddleOCR-VL language and Vision encoder blocks with GPTQModel.

    ``calibration_data`` must be either a list/iterable of already processed
    multimodal model-input dictionaries or a JSON/JSONL cases path containing
    those dictionaries. At least one calibration item must include visual
    tensors such as ``pixel_values``/``image_grid_thw``; pure text calibration is
    rejected for this multimodal recipe.
    """

    method = _normalize_method(method)
    rotation = _normalize_rotation(rotation)
    vision_rotation = _normalize_rotation(vision_rotation) if quantize_vision else None
    resolved_vision_bits = int(bits if vision_bits is None else vision_bits)
    resolved_vision_group_size = int(group_size if vision_group_size is None else vision_group_size)
    resolved_vision_iters = int(iters if vision_iters is None else vision_iters)
    _validate_recipe_args(
        method=method,
        artifact_format=artifact_format,
        bits=bits,
        group_size=group_size,
        iters=iters,
        rotation=rotation,
        batch_size=batch_size,
        damp_percent=damp_percent,
        offload_to_disk=offload_to_disk,
    )
    if quantize_vision:
        _validate_recipe_args(
            method=method,
            artifact_format=artifact_format,
            bits=resolved_vision_bits,
            group_size=resolved_vision_group_size,
            iters=resolved_vision_iters,
            rotation=vision_rotation,
            batch_size=batch_size,
            damp_percent=damp_percent,
            offload_to_disk=offload_to_disk,
        )
        if resolved_vision_bits != int(bits) or resolved_vision_group_size != int(group_size):
            raise ValueError(
                "Merged PaddleOCR-VL checkpoints currently require matching language and Vision "
                "bits/group_size so one GPTQModel QuantizeConfig can reload both scopes"
            )

    model_path = Path(model_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_path}")

    examples, calibration_source = _load_calibration_cases(calibration_data)
    _validate_multimodal_calibration(examples)
    resolved_offload_path = _resolve_offload_path(
        output_path=output_path,
        offload_to_disk=offload_to_disk,
        offload_to_disk_path=offload_to_disk_path,
    )
    explicit_untied_embeddings = _raw_config_declares_untied_embeddings(model_path)

    precision = {
        "scope": "language_model+vision_encoder" if quantize_vision else "language_model",
        "bits": int(bits),
        "components": {
            "language_model": {"bits": int(bits)},
            **({"vision_encoder": {"bits": resolved_vision_bits}} if quantize_vision else {}),
        },
        "group_size": int(group_size),
        "symmetric": bool(sym),
        "desc_act": bool(desc_act),
        "visual_modules": (
            {
                "scope": "vision_model.encoder.layers",
                "bits": resolved_vision_bits,
                "group_size": resolved_vision_group_size,
                "iters": resolved_vision_iters,
                "rotation": vision_rotation,
            }
            if quantize_vision
            else "float"
        ),
        "vision_patch_embedding": "float",
        "projector_modules": "float",
        "language_rotation": rotation,
    }
    calibration = {
        "source": calibration_source,
        "examples": len(examples),
        "requires_visual_inputs": True,
        "batch_size": int(batch_size),
    }
    provenance = {
        "recipe": _RECIPE_NAME,
        "backend": "direct_gptqmodel_api",
        "dry_run": bool(dry_run),
        "device": str(device),
        "seed": int(seed),
        "trust_remote_code": bool(trust_remote_code),
        "offload_to_disk": bool(offload_to_disk),
        "offload_to_disk_path": resolved_offload_path,
        "explicit_untied_embeddings": explicit_untied_embeddings,
        "rotation_contract": _rotation_contract(rotation),
        "vision_rotation_contract": _vision_rotation_contract(vision_rotation),
        "processor_assets": _processor_assets_provenance(model_path),
        "quantize_kwargs": dict(quantize_kwargs),
    }
    result = PaddleOCRVLQuantRecipeResult(
        output_dir=str(output_path),
        method=method,
        artifact_format=str(artifact_format),
        model_family=_MODEL_TYPE,
        precision=precision,
        calibration=calibration,
        provenance=provenance,
    )
    if dry_run:
        return result

    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_path}")

    import torch

    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.quantization.config import FORMAT, METHOD

    random.seed(int(seed))
    torch.manual_seed(int(seed))

    language_quantize_config = QuantizeConfig(
        bits=int(bits),
        group_size=int(group_size),
        sym=bool(sym),
        desc_act=bool(desc_act),
        damp_percent=damp_percent,
        quant_method=METHOD.AUTO_ROUND if method == "autoround" else METHOD.GPTQ,
        format=FORMAT.GPTQ,
        auto_round_enabled=method == "autoround",
        iters=int(iters),
        device=str(device),
        offload_to_disk=bool(offload_to_disk),
        offload_to_disk_path=resolved_offload_path,
        rotation=rotation,
        meta={
            "recipe": _RECIPE_NAME,
            "model_family": _MODEL_TYPE,
            "quantized_scope": precision["scope"],
            "visual_modules": precision["visual_modules"],
            "vision_patch_embedding": "float",
            "projector_modules": "float",
            "rotation_contract": _rotation_contract(rotation),
            "vision_rotation_contract": _vision_rotation_contract(vision_rotation),
        },
    )
    def config_patch():
        return (
            preserve_paddleocr_vl_tie_word_embeddings(False)
            if explicit_untied_embeddings
            else nullcontext()
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paddleocr_vl_quant_", dir=output_path.parent) as temp_root:
        language_dir = Path(temp_root) / "language"
        register_paddleocr_vl_gptqmodel(stage="language")
        with config_patch():
            model = GPTQModel.load(
                str(model_path),
                quantize_config=language_quantize_config,
                trust_remote_code=bool(trust_remote_code),
            )
        model.quantize(
            examples,
            batch_size=int(batch_size),
            calibration_data_min_length=1,
            **quantize_kwargs,
        )
        model.save(str(language_dir if quantize_vision else output_path))

        if quantize_vision:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            vision_quantize_config = QuantizeConfig(
                bits=resolved_vision_bits,
                group_size=resolved_vision_group_size,
                sym=bool(sym),
                desc_act=bool(desc_act),
                damp_percent=damp_percent,
                quant_method=METHOD.AUTO_ROUND if method == "autoround" else METHOD.GPTQ,
                format=FORMAT.GPTQ,
                auto_round_enabled=method == "autoround",
                iters=resolved_vision_iters,
                device=str(device),
                offload_to_disk=bool(offload_to_disk),
                offload_to_disk_path=resolved_offload_path,
                rotation=None,
                meta={
                    "recipe": _RECIPE_NAME,
                    "model_family": _MODEL_TYPE,
                    "quantized_scope": "vision_encoder",
                    "vision_rotation_contract": _vision_rotation_contract(vision_rotation),
                },
            )
            vision_dir = Path(temp_root) / "vision"
            register_paddleocr_vl_gptqmodel(stage="vision")
            with config_patch():
                model = GPTQModel.load(
                    str(model_path),
                    quantize_config=vision_quantize_config,
                    trust_remote_code=bool(trust_remote_code),
                )
            if vision_rotation is not None:
                rotate_vision_attention_vo(model.model, vision_rotation, torch.device(device))
            model.quantize(
                examples,
                batch_size=int(batch_size),
                calibration_data_min_length=1,
                **quantize_kwargs,
            )
            model.save(str(vision_dir))
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            merged_dir = Path(temp_root) / "merged"
            _merge_quantized_stages(language_dir, vision_dir, merged_dir)
            _publish_quantized_stages(merged_dir, output_path)

    register_paddleocr_vl_gptqmodel(stage="combined")
    result.provenance["processor_assets"] = _copy_processor_assets(model_path, output_path)
    _write_provenance(output_path, result)
    return result


def _publish_quantized_stages(merged_dir: Path, output_path: Path) -> None:
    """Publish a merged checkpoint without replacing a reserved output root."""
    if output_path.exists():
        if any(output_path.iterdir()):
            raise FileExistsError(f"Quant output directory is not empty: {output_path}")
        for entry in merged_dir.iterdir():
            entry.rename(output_path / entry.name)
        merged_dir.rmdir()
    else:
        merged_dir.rename(output_path)


class PaddleOCRVLQModel(_base_qmodel()):
    """GPTQModel definition for PaddleOCR-VL conditional generation models."""

    loader = _PaddleOCRVLLoader
    support_batch_quantize = False
    require_load_processor = False
    require_trust_remote_code = False
    pre_lm_head_norm_module = "model.language_model.norm"
    layer_modules_strict = True
    lm_head = "lm_head"
    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model = kwargs.get("model")
        if model is None and args:
            model = args[0]
        self._select_module_tree(model)
        super().__init__(*args, **kwargs)

    @classmethod
    def _select_module_tree(cls, model: Any) -> None:
        if _has_module_path(model, "model.language_model.layers"):
            cls.module_tree = [
                "model",
                "language_model",
                "layers",
                "#",
                cls._layer_tree(),
            ]
            cls.pre_lm_head_norm_module = "model.language_model.norm"
        elif _has_module_path(model, "model.layers"):
            cls.module_tree = ["model", "layers", "#", cls._layer_tree()]
            cls.pre_lm_head_norm_module = "model.norm"
        else:
            cls.module_tree = None

    @staticmethod
    def _layer_tree() -> dict[str, tuple[str, ...]]:
        return {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }

    def prepare_dataset(
        self,
        calibration_dataset: Iterable[Mapping[str, Any]],
        calibration_dataset_concat_size: int | None = None,
        calibration_dataset_sort: str | None = None,
        batch_size: int = 1,
        calibration_data_min_length: int = 1,
        calibration_concat_separator: str | None = None,
    ) -> list[dict[str, Any]]:
        del calibration_dataset_concat_size, calibration_dataset_sort
        del batch_size, calibration_data_min_length, calibration_concat_separator
        examples = [_tensorize_model_inputs(example) for example in calibration_dataset]
        _validate_multimodal_calibration(examples)
        return examples

    def pre_quantize_generate_hook_start(self) -> None:
        from gptqmodel.utils.model import move_to

        for module_name in self._first_pass_modules():
            module = _resolve_module(self.model, module_name)
            if module is not None:
                move_to(module, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self) -> None:
        from gptqmodel.models._const import CPU
        from gptqmodel.utils.model import move_to

        for module_name in self._first_pass_modules():
            module = _resolve_module(self.model, module_name)
            if module is not None:
                move_to(module, device=CPU)

    @classmethod
    def _first_pass_modules(cls) -> tuple[str, ...]:
        if cls.module_tree and cls.module_tree[:3] == ["model", "language_model", "layers"]:
            return (
                "model.visual",
                "model.projector",
                "model.language_model.embed_tokens",
                "model.language_model.rotary_emb",
            )
        return (
            "visual",
            "mlp_AR",
            "model.embed_tokens",
            "model.rotary_emb",
        )


class PaddleOCRVLVisionQModel(PaddleOCRVLQModel):
    """GPTQModel calibration traversal for the PaddleOCR Vision encoder."""

    pre_lm_head_norm_module = None
    module_tree = [
        "model",
        "visual",
        "vision_model",
        "encoder",
        "layers",
        "#",
        {
            "layer_norm1": ("layer_norm1:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "out_proj:1",
            ),
            "layer_norm2": ("layer_norm2:!",),
            "mlp": ("fc1:0", "fc2:1"),
        },
    ]

    @classmethod
    def _select_module_tree(cls, model: Any) -> None:
        if not _has_module_path(model, "model.visual.vision_model.encoder.layers"):
            cls.module_tree = None

    @classmethod
    def _first_pass_modules(cls) -> tuple[str, ...]:
        # The conditional-generation forward builds token embeddings before it
        # enters the Vision encoder. GPTQModel places calibration tensors on the
        # quantization device, so the embedding must accompany the visual tower
        # during the input-capture pass even though it remains floating point.
        return ("model.language_model.embed_tokens", "model.visual")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        layers = _resolve_module(self.model, "model.visual.vision_model.encoder.layers")
        if layers is not None:
            for layer in layers:
                original_forward = layer.forward

                def vision_forward(*forward_args: Any, _forward=original_forward, **forward_kwargs: Any):
                    for key in (
                        "attention_mask",
                        "position_ids",
                        "use_cache",
                        "past_key_value",
                        "past_key_values",
                    ):
                        forward_kwargs.pop(key, None)
                    return _forward(*forward_args, **forward_kwargs)

                layer.forward = vision_forward


class PaddleOCRVLCombinedQModel(PaddleOCRVLQModel):
    """Loader definition for a merged language and Vision checkpoint."""

    @classmethod
    def extract_layers_node(cls) -> list[str]:
        return [
            "model.language_model.layers",
            "model.visual.vision_model.encoder.layers",
        ]

    @classmethod
    def simple_layer_modules(
        cls,
        model_config: Any,
        quantize_config: Any,
        is_awq_quantize: bool = False,
        include_capture_only: bool = False,
    ) -> list[list[str]]:
        language = PaddleOCRVLQModel.simple_layer_modules(
            model_config,
            quantize_config,
            is_awq_quantize=is_awq_quantize,
            include_capture_only=include_capture_only,
        )
        vision = PaddleOCRVLVisionQModel.simple_layer_modules(
            model_config,
            quantize_config,
            is_awq_quantize=is_awq_quantize,
            include_capture_only=include_capture_only,
        )
        return language + vision

    @classmethod
    def get_base_modules(cls, model: Any) -> list[str]:
        del model
        return []


def register_paddleocr_vl_gptqmodel(stage: str = "combined") -> type:
    """Register PaddleOCR-VL in GPTQModel's runtime model map."""

    import gptqmodel.models.auto as auto
    from gptqmodel.quantization.rotation.model import MODEL_ROTATION_REGISTRY
    from gptqmodel.quantization.rotation.rotation import fuse_layer_norms

    stage_classes = {
        "language": PaddleOCRVLQModel,
        "vision": PaddleOCRVLVisionQModel,
        "combined": PaddleOCRVLCombinedQModel,
    }
    try:
        selected_class = stage_classes[stage]
    except KeyError as exc:
        raise ValueError("stage must be 'language', 'vision', or 'combined'") from exc

    auto.MODEL_MAP[_MODEL_TYPE] = selected_class
    if _MODEL_TYPE not in auto.SUPPORTED_MODELS:
        auto.SUPPORTED_MODELS.append(_MODEL_TYPE)
    rotation_entry = {
        "cls": PaddleOCRVLQModel,
        "vision": False,
        "fuse_norm": fuse_layer_norms,
        "rotate_fn": rotate_model_paddleocr_vl,
    }
    for index, entry in enumerate(MODEL_ROTATION_REGISTRY):
        if entry.get("cls") is PaddleOCRVLQModel:
            MODEL_ROTATION_REGISTRY[index] = rotation_entry
            break
    else:
        MODEL_ROTATION_REGISTRY.append(rotation_entry)
    return selected_class


def rotate_model_paddleocr_vl(
    model: Any,
    rotate_mode: str,
    device: Any,
    lm_head_name: str,
    layers_node: str,
    Q: Any = None,  # noqa: N803 - GPTQModel rotation callbacks pass this exact keyword.
) -> tuple[Any, Any]:
    """Apply offline R1 QuaRot to the PaddleOCR-VL language hidden basis."""

    from gptqmodel.quantization.rotation.rotation import (
        _resolve_rotation_matrix,
        _resolve_text_config,
        get_module_by_name_prefix,
        get_orthogonal_matrix,
        rotate_attention_inputs,
        rotate_attention_output,
        rotate_embeddings,
        rotate_head,
        rotate_mlp_input,
        rotate_mlp_output,
        torch_empty_cache,
    )

    q_block_spec = Q
    if isinstance(Q, dict):
        q_block_spec = Q.get("Q_block", Q.get("Q"))

    text_config = _resolve_text_config(model.config)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("PaddleOCR-VL rotation requires `text_config.hidden_size`.")

    Q_block = _resolve_rotation_matrix(q_block_spec, 0, "Q_block")
    if Q_block is None:
        Q_block = get_orthogonal_matrix(int(hidden_size), rotate_mode, device)

    rotate_embeddings(model, Q_block, device)
    _rotate_projector_output(model, Q_block, device)
    rotate_head(model, Q_block, device, lm_head_name)
    torch_empty_cache()

    layers, _ = get_module_by_name_prefix(model, layers_node)
    for index, layer in enumerate(layers):
        q_layer = _resolve_rotation_matrix(q_block_spec, index, "Q_block")
        if q_layer is None:
            q_layer = Q_block
        rotate_attention_inputs(layer, q_layer, device)
        rotate_attention_output(layer, q_layer, device)
        rotate_mlp_input(layer, q_layer, device)
        rotate_mlp_output(layer, q_layer, device)

    return model, Q_block


def rotate_vision_attention_vo(
    model: Any,
    rotate_mode: str,
    device: Any,
    matrices: Mapping[int, Any] | None = None,
) -> dict[int, Any]:
    """Apply exact per-head QuaRot V/O reparameterization to Vision attention.

    Rotating values inside each attention head commutes with the attention
    probability matrix. Applying the inverse basis to ``out_proj`` therefore
    preserves the floating-point function without touching Q/K or 2D RoPE.
    """

    import torch

    from gptqmodel.quantization.rotation.rotation import get_orthogonal_matrix

    layers = _resolve_module(model, "model.visual.vision_model.encoder.layers")
    if layers is None:
        raise ValueError("PaddleOCR-VL Vision rotation requires model.visual.vision_model.encoder.layers")

    vision_config = getattr(model.config, "vision_config", None)
    num_heads = int(getattr(vision_config, "num_attention_heads", 0))
    if num_heads <= 0:
        raise ValueError("PaddleOCR-VL Vision rotation requires vision_config.num_attention_heads")

    resolved: dict[int, Any] = {}
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        value_width = int(attention.v_proj.out_features)
        if value_width % num_heads:
            raise ValueError(
                f"Vision v_proj width {value_width} is not divisible by {num_heads} attention heads"
            )
        head_dim = value_width // num_heads
        q_head = None if matrices is None else matrices.get(index)
        if q_head is None:
            q_head = get_orthogonal_matrix(head_dim, rotate_mode, torch.device(device))
        q_head = q_head.to(device=device, dtype=torch.float64)
        if q_head.shape != (head_dim, head_dim):
            raise ValueError(
                f"Vision rotation matrix for layer {index} must have shape {(head_dim, head_dim)}"
            )

        v_proj = attention.v_proj
        out_proj = attention.out_proj
        v_dtype = v_proj.weight.dtype
        out_dtype = out_proj.weight.dtype
        value_weight = v_proj.weight.data.to(device=device, dtype=torch.float64)
        value_weight = value_weight.view(num_heads, head_dim, value_weight.shape[1])
        value_weight = torch.matmul(q_head.T.unsqueeze(0), value_weight)
        v_proj.weight.data = value_weight.reshape_as(v_proj.weight).to(device="cpu", dtype=v_dtype)
        if v_proj.bias is not None:
            value_bias = v_proj.bias.data.to(device=device, dtype=torch.float64).view(num_heads, head_dim)
            value_bias = torch.matmul(value_bias, q_head)
            v_proj.bias.data = value_bias.reshape_as(v_proj.bias).to(device="cpu", dtype=v_dtype)

        output_weight = out_proj.weight.data.to(device=device, dtype=torch.float64)
        output_weight = output_weight.view(output_weight.shape[0], num_heads, head_dim)
        output_weight = torch.matmul(output_weight, q_head)
        out_proj.weight.data = output_weight.reshape_as(out_proj.weight).to(device="cpu", dtype=out_dtype)
        resolved[index] = q_head.detach().cpu()

    return resolved


def _rotate_projector_output(model: Any, q: Any, device: Any) -> None:
    for projector_path in ("model.projector.linear_2", "mlp_AR.linear_2"):
        linear = _resolve_module(model, projector_path)
        if linear is not None:
            _rotate_linear_output(linear, q, device)
            return


def _rotate_linear_output(linear: Any, q: Any, device: Any) -> None:
    import torch

    dtype = linear.weight.data.dtype
    weight = linear.weight.data.to(device=device, dtype=torch.float64)
    linear.weight.data = torch.matmul(q.T, weight).to(device="cpu", dtype=dtype)
    if linear.bias is not None:
        bias = linear.bias.data.to(device=device, dtype=torch.float64)
        linear.bias.data = torch.matmul(q.T, bias).to(device="cpu", dtype=dtype)


@contextmanager
def preserve_paddleocr_vl_tie_word_embeddings(value: bool):
    """Patch native PaddleOCRVLConfig construction during GPTQModel loading."""

    try:
        import transformers.models.paddleocr_vl.configuration_paddleocr_vl as configuration
    except Exception:
        yield
        return

    config_cls = getattr(configuration, "PaddleOCRVLConfig", None)
    if config_cls is None:
        yield
        return

    original_init = config_cls.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if getattr(self, "model_type", None) != _MODEL_TYPE:
            return
        self.tie_word_embeddings = bool(value)
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            text_config.tie_word_embeddings = bool(value)

    config_cls.__init__ = patched_init
    try:
        yield
    finally:
        config_cls.__init__ = original_init


def _normalize_method(method: str) -> str:
    normalized = str(method).lower().replace("-", "_")
    if normalized == "auto_round":
        normalized = "autoround"
    if normalized not in _SUPPORTED_METHODS:
        raise ValueError("method must be 'autoround' or 'gptq'")
    return normalized


def _normalize_rotation(rotation: str | None) -> str | None:
    if rotation is None or rotation is False:
        return None
    normalized = str(rotation).lower().replace("-", "_")
    if normalized not in _SUPPORTED_ROTATIONS:
        raise ValueError("rotation must be None or 'hadamard'")
    return normalized


def _rotation_contract(rotation: str | None) -> dict[str, Any]:
    if rotation is None:
        return {
            "enabled": False,
            "runtime_wrapper_required": False,
        }
    return {
        "enabled": True,
        "mode": rotation,
        "scope": "offline_r1_hidden_basis",
        "rotated_modules": [
            "model.language_model.embed_tokens",
            "model.projector.linear_2",
            "model.language_model.layers.*.self_attn.{q_proj,k_proj,v_proj}:input",
            "model.language_model.layers.*.self_attn.o_proj:output",
            "model.language_model.layers.*.mlp.{gate_proj,up_proj}:input",
            "model.language_model.layers.*.mlp.down_proj:output",
            "lm_head",
        ],
        "runtime_wrapper_required": False,
        "reload_requirement": (
            "import recipes.paddleocr_vl and call "
            "register_paddleocr_vl_gptqmodel before GPTQModel.load/from_quantized"
        ),
        "online_r2_r4": "not_enabled",
    }


def _vision_rotation_contract(rotation: str | None) -> dict[str, Any]:
    if rotation is None:
        return {
            "enabled": False,
            "runtime_wrapper_required": False,
        }
    return {
        "enabled": True,
        "mode": rotation,
        "scope": "offline_per_head_attention_v_o_basis",
        "rotated_modules": [
            "model.visual.vision_model.encoder.layers.*.self_attn.v_proj:output_per_head",
            "model.visual.vision_model.encoder.layers.*.self_attn.out_proj:input_per_head",
        ],
        "preserved_modules": [
            "model.visual.vision_model.encoder.layers.*.self_attn.{q_proj,k_proj}",
            "model.visual.vision_model.embeddings.patch_embedding",
            "model.projector",
        ],
        "runtime_wrapper_required": False,
        "preserves_2d_rope_basis": True,
        "global_vision_r1": "not_enabled_across_learned_layernorm_and_projector_boundaries",
    }


def _validate_recipe_args(
    *,
    method: str,
    artifact_format: str,
    bits: int,
    group_size: int,
    iters: int,
    rotation: str | None,
    batch_size: int,
    damp_percent: float | None,
    offload_to_disk: bool,
) -> None:
    if method not in {"autoround", "gptq"}:
        raise ValueError("method must be 'autoround' or 'gptq'")
    if str(artifact_format) != _ARTIFACT_FORMAT:
        raise ValueError("artifact_format must be 'gptqmodel_hf'")
    if int(bits) not in {2, 3, 4, 5, 8}:
        raise ValueError("bits must be one of 2, 3, 4, 5, or 8")
    if int(group_size) <= 0 and int(group_size) != -1:
        raise ValueError("group_size must be positive or -1")
    if int(iters) < 0:
        raise ValueError("iters must be non-negative")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if damp_percent is not None and not (0 < float(damp_percent) < 1):
        raise ValueError("damp_percent must be between 0 and 1 when provided")
    if rotation not in _SUPPORTED_ROTATIONS:
        raise ValueError("rotation must be None or 'hadamard'")
    if rotation is not None and bool(offload_to_disk):
        raise ValueError(
            "PaddleOCR-VL offline R1 rotation currently requires offload_to_disk=False. "
            "GPTQModel offload save can persist a mixed hidden-basis artifact: rotated "
            "decoder/embed/lm_head tensors with restored unrotated float boundary modules "
            "such as projector.linear_2 or final norm. Re-run with offload_to_disk=False "
            "or add a verified materialized-save contract before enabling rotated offload."
        )


def _load_calibration_cases(
    calibration_data: Iterable[Mapping[str, Any]] | str | Path,
) -> tuple[list[dict[str, Any]], str]:
    if isinstance(calibration_data, (str, Path)):
        path = Path(calibration_data).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"calibration cases path does not exist: {path}")
        records = _read_json_records(path)
        return [_tensorize_model_inputs(record) for record in records], str(path)

    if isinstance(calibration_data, Mapping):
        raise ValueError("calibration_data must be an iterable of input dictionaries, not a single dictionary")

    examples = [_tensorize_model_inputs(record) for record in calibration_data]
    return examples, "memory"


def _read_json_records(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"]
    raise ValueError("calibration cases file must be a JSON list, JSONL file, or an object with a 'cases' list")


def _tensorize_model_inputs(example: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(example, Mapping):
        raise ValueError("each calibration item must be a dictionary of model inputs")

    import torch

    converted: dict[str, Any] = {}
    for key, value in example.items():
        if value is None or isinstance(value, (str, bytes)):
            converted[str(key)] = value
        elif torch.is_tensor(value):
            converted[str(key)] = value.detach()
        elif isinstance(value, (int, float, bool)):
            converted[str(key)] = value
        elif _is_tensor_like_sequence(value):
            dtype = torch.long if str(key).endswith(("ids", "mask")) or str(key) == "input_ids" else None
            converted[str(key)] = torch.as_tensor(value, dtype=dtype)
        else:
            converted[str(key)] = value
    return converted


def _is_tensor_like_sequence(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if not value:
        return True
    if all(isinstance(item, (int, float, bool)) for item in value):
        return True
    return all(_is_tensor_like_sequence(item) for item in value)


def _validate_multimodal_calibration(examples: list[Mapping[str, Any]]) -> None:
    if not examples:
        raise ValueError("calibration_data must contain at least one example")
    for index, example in enumerate(examples):
        if "input_ids" not in example:
            raise ValueError(f"calibration example {index} is missing input_ids")
        has_image = "pixel_values" in example and "image_grid_thw" in example
        has_video = "pixel_values_videos" in example and "video_grid_thw" in example
        if has_image or has_video:
            return
    raise ValueError(
        "PaddleOCR-VL calibration_data must include at least one processed "
        "image/video example; pure text calibration is not accepted."
    )


def _resolve_offload_path(
    *,
    output_path: Path,
    offload_to_disk: bool,
    offload_to_disk_path: str | None,
) -> str | None:
    if not offload_to_disk:
        return None
    path = (
        Path(offload_to_disk_path).expanduser()
        if offload_to_disk_path
        else output_path.parent / f"{output_path.name}_offload"
    )
    return str(path.resolve())


def _processor_assets_provenance(model_path: Path) -> dict[str, Any]:
    files = []
    missing = []
    for name in _PROCESSOR_ASSET_FILES:
        source = model_path / name
        if source.is_file():
            files.append(
                {
                    "name": name,
                    "source_sha256": _sha256_file(source),
                    "action": "available",
                }
            )
        else:
            missing.append(name)
    fingerprint = _processor_assets_fingerprint(files)
    return {
        "source_dir": str(model_path),
        "source_revision": _infer_source_revision(model_path) or (
            f"local-assets:{fingerprint[:16]}" if fingerprint is not None else None
        ),
        "source_fingerprint_sha256": fingerprint,
        "files": files,
        "missing": missing,
        "status": "available" if files else "source_missing",
    }


def _copy_processor_assets(model_path: Path, output_path: Path) -> dict[str, Any]:
    """Copy safe processor/tokenizer assets without touching quantized artifacts."""

    output_path.mkdir(parents=True, exist_ok=True)
    payload = _processor_assets_provenance(model_path)
    copied_files = []
    for item in payload["files"]:
        name = item["name"]
        _validate_processor_asset_name(name)
        source = model_path / name
        destination = output_path / name
        shutil.copy2(source, destination)
        copied_files.append(
            {
                **item,
                "action": "copied",
                "destination_sha256": _sha256_file(destination),
            }
        )
    payload["files"] = copied_files
    payload["status"] = "copied" if copied_files else "source_missing"
    return payload


def _merge_quantized_stages(
    language_dir: Path,
    vision_dir: Path,
    output_path: Path,
    max_shard_size: int = 512 * 1024 * 1024,
) -> None:
    """Merge stage checkpoints while dropping superseded floating Vision tensors."""

    from safetensors import safe_open
    from safetensors.torch import save_file

    language_map = _safetensors_weight_map(language_dir)
    vision_map = _safetensors_weight_map(vision_dir)
    selected: dict[str, tuple[Path, str]] = {
        key: (language_dir, filename)
        for key, filename in language_map.items()
        if not key.startswith(_VISION_ENCODER_PREFIX)
    }
    selected.update(
        {
            key: (vision_dir, filename)
            for key, filename in vision_map.items()
            if key.startswith(_VISION_ENCODER_PREFIX)
        }
    )
    if not any(key.endswith(".qweight") and key.startswith(_VISION_ENCODER_PREFIX) for key in selected):
        raise RuntimeError("Vision quantization stage produced no Vision encoder qweight tensors")

    output_path.mkdir(parents=True, exist_ok=False)
    for source in language_dir.iterdir():
        if source.name.endswith(".safetensors") or source.name == "model.safetensors.index.json":
            continue
        if source.is_dir():
            shutil.copytree(source, output_path / source.name)
        else:
            shutil.copy2(source, output_path / source.name)
    language_log = output_path / "quant_log.csv"
    if language_log.is_file():
        language_log.rename(output_path / "quant_log_language.csv")
    vision_log = vision_dir / "quant_log.csv"
    if vision_log.is_file():
        shutil.copy2(vision_log, output_path / "quant_log_vision.csv")

    part_paths: list[Path] = []
    tensor_to_part: dict[str, int] = {}
    tensors: dict[str, Any] = {}
    shard_bytes = 0
    total_size = 0

    def flush() -> None:
        nonlocal tensors, shard_bytes
        if not tensors:
            return
        part_path = output_path / f".model-part-{len(part_paths) + 1:05d}.safetensors"
        save_file(tensors, part_path, metadata={"format": "pt"})
        part_index = len(part_paths)
        for tensor_name in tensors:
            tensor_to_part[tensor_name] = part_index
        part_paths.append(part_path)
        tensors = {}
        shard_bytes = 0

    current_path: Path | None = None
    current_file: Any | None = None
    try:
        for key in sorted(selected):
            source_dir, filename = selected[key]
            source_path = source_dir / filename
            if source_path != current_path:
                if current_file is not None:
                    current_file.__exit__(None, None, None)
                current_file = safe_open(source_path, framework="pt", device="cpu")
                current_file.__enter__()
                current_path = source_path
            tensor = current_file.get_tensor(key)
            tensor_bytes = tensor.numel() * tensor.element_size()
            if tensors and shard_bytes + tensor_bytes > max_shard_size:
                flush()
            tensors[key] = tensor
            shard_bytes += tensor_bytes
            total_size += tensor_bytes
        flush()
    finally:
        if current_file is not None:
            current_file.__exit__(None, None, None)

    shard_count = len(part_paths)
    weight_map: dict[str, str] = {}
    for index, part_path in enumerate(part_paths, start=1):
        filename = (
            "model.safetensors"
            if shard_count == 1
            else f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        )
        part_path.rename(output_path / filename)
        for key, part_index in tensor_to_part.items():
            if part_index == index - 1:
                weight_map[key] = filename

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (output_path / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safetensors_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in payload["weight_map"].items()}

    model_path = model_dir / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"No safetensors checkpoint found under {model_dir}")
    from safetensors import safe_open

    with safe_open(model_path, framework="pt", device="cpu") as stream:
        return {key: model_path.name for key in stream.keys()}


def _validate_processor_asset_name(name: str) -> None:
    if name in _PROTECTED_OUTPUT_NAMES or name.endswith(_PROTECTED_OUTPUT_SUFFIXES) or name.endswith(".py"):
        raise ValueError(f"Refusing to copy protected PaddleOCR-VL artifact: {name}")


def _infer_source_revision(model_path: Path) -> str | None:
    config_path = model_path / "config.json"
    if config_path.is_file():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        for key in ("_commit_hash", "commit_hash", "revision"):
            value = payload.get(key)
            if value:
                return str(value)
    if model_path.parent.name == "snapshots":
        return model_path.name
    return None


def _processor_assets_fingerprint(files: list[dict[str, Any]]) -> str | None:
    if not files:
        return None
    digest = sha256()
    for item in sorted(files, key=lambda entry: str(entry["name"])):
        digest.update(str(item["name"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["source_sha256"]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_provenance(output_path: Path, result: PaddleOCRVLQuantRecipeResult) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    (output_path / "paddleocr_vl_recipe_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _raw_config_declares_untied_embeddings(model_path: Path) -> bool:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return False
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload.get("tie_word_embeddings") is False


def _has_module_path(root: Any, dotted_path: str) -> bool:
    return _resolve_module(root, dotted_path) is not None


def _resolve_module(root: Any, dotted_path: str) -> Any | None:
    current = root
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


__all__ = [
    "PaddleOCRVLCombinedQModel",
    "PaddleOCRVLQModel",
    "PaddleOCRVLQuantRecipeResult",
    "PaddleOCRVLVisionQModel",
    "preserve_paddleocr_vl_tie_word_embeddings",
    "quantize_paddleocr_vl",
    "register_paddleocr_vl_gptqmodel",
    "rotate_model_paddleocr_vl",
    "rotate_vision_attention_vo",
]
