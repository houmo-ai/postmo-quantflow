# SPDX-License-Identifier: Apache-2.0
"""Custom HuggingFace-compatible config + parser for the vllm_merak backend.

vLLM internally calls
``vllm.transformers_utils.config.get_config(...)`` very early in
``ModelConfig.__post_init__``.  By default this dispatches to the ``"hf"``
parser which calls ``transformers.AutoConfig.from_pretrained``.  We do
**not** want that: the Merak backend ships its own config schema and the
weights are not in HuggingFace format at all.

To bypass transformers entirely we:

1. Define :class:`MerakConfig` as a thin subclass of
   ``transformers.PretrainedConfig``.  It is only used as a typed value
   container - vLLM and downstream code expect a ``PretrainedConfig``
   instance, but no HF model class is ever instantiated from it.
2. Register :class:`MerakConfigParser` under the ``"merak"`` config
   format.  The parser reads our own ``merak_config.json`` (or
   ``config.json``) directly with ``json.load`` and constructs a
   :class:`MerakConfig` from the dict.

Users select this parser via::

    LLM(model="/path/to/merak_model", config_format="merak", ...)
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

from transformers import PretrainedConfig
from vllm.transformers_utils.config import HFConfigParser, register_config_parser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from .constants import SUPPORTED_HYBRID_CACHE_POLICIES
from .model_executor.register import (
    get_registered_custom_architectures,
    get_registered_model_cls,
)


__all__ = [
    "MERAK_DEFAULT_ARCH",
    "MERAK_PREFERRED_BLOCK_SIZE",
    "HMFP_NSHARE",
    "MERAK_MODEL_TYPE",
    "MerakConfig",
    "MerakConfigParser",
]

# ---------------------------------------------------------------------------
# Naming constants
# ---------------------------------------------------------------------------

#: ``model_type`` value reported on the ``PretrainedConfig`` so vLLM and
#: any downstream sniffers can recognise this backend.
MERAK_MODEL_TYPE: str = "merak_llm"

#: Default architecture string written into ``architectures`` when the
#: custom config file omits it.  Must match the name registered with
#: ``vllm.ModelRegistry.register_model``.
MERAK_DEFAULT_ARCH: str = "MerakForCausalLM"

MERAK_PREFERRED_BLOCK_SIZE: int = 64
HMFP_NSHARE: int = 64

#: File name of the Merak-specific config dropped next to the HMONNX export.
_MERAK_CONFIG_FILE_NAME: str = "merak_config.json"
_MERAK_SOURCE_WEIGHT_FORMAT_KEY: str = "merak_source_weight_format"
_SUPPORTED_KV_CACHE_FORMATS: dict[str, str] = {
    "float16": "float16",
    "hmbf16": "HMBF16",
}
_QWEN3_5_SOURCE_ARCHITECTURES = frozenset(
    {"Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"}
)
_QWEN3_5_MULTIMODAL_ARCHITECTURE = "Qwen3_5MerakForCausalLM"
_QWEN3_5_TEXT_ARCHITECTURE = "Qwen3_5MerakTextForCausalLM"
_WEMM_EMBEDDING_ARCHITECTURE = "WeMMEmbeddingMerakForEmbedding"
_GEMMA4_TOWER_SOURCE_ARCHITECTURES = frozenset(
    {"Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"}
)
_GEMMA4_UNIFIED_SOURCE_ARCHITECTURES = frozenset(
    {"Gemma4UnifiedForConditionalGeneration"}
)
_GEMMA4_SOURCE_ARCHITECTURES = (
    _GEMMA4_TOWER_SOURCE_ARCHITECTURES | _GEMMA4_UNIFIED_SOURCE_ARCHITECTURES
)
_GEMMA4_LOADER_ARCHITECTURE = "Gemma4MerakForCausalLM"
_GEMMA4_UNIFIED_LOADER_ARCHITECTURE = "Gemma4UnifiedMerakForCausalLM"
_GEMMA4_CONTRACT_FIELDS = (
    "attention_contract_version",
    "layer_types",
    "layer_cache_types",
    "layer_cache_indices",
    "layer_cache_owner_indices",
    "layer_kv_shapes",
    "sliding_window",
    "prefill_chunk_length",
    "sliding_kv_cache_input_mode",
    "bidirectional_vision_attention",
    "max_mm_ranges_per_chunk",
)


def _normalize_kv_cache_format(kv_cache_format: str | None) -> str:
    if not isinstance(kv_cache_format, str):
        return "float16"

    normalized = kv_cache_format.strip().lower()
    if normalized in _SUPPORTED_KV_CACHE_FORMATS:
        return _SUPPORTED_KV_CACHE_FORMATS[normalized]
    if normalized.startswith("sefp"):
        return kv_cache_format.strip()
    raise ValueError(
        "kv_cache_format must be one of: float16, HMBF16, or an SEFP dtype string"
    )


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------


class MerakConfig(PretrainedConfig):
    """Self-contained config for a Merak graph-IR model.

    The class extends :class:`transformers.PretrainedConfig` only because
    vLLM's internal type signature requires a ``PretrainedConfig``
    instance.  No HuggingFace ``AutoModel`` machinery is invoked - the
    model loader instantiates :class:`MerakForCausalLM` manually.

    Required fields (read from the JSON file):

    * ``hidden_size`` / ``num_attention_heads`` / ``num_hidden_layers``
      / ``vocab_size`` / ``max_position_embeddings`` - standard scalars
      vLLM's scheduler reads off the config.
    * ``ir_path`` - relative or absolute path to the graph-IR artifact
      that the loader hands to the runtime.  Optional at config-parse
      time (the loader resolves it later).

    Any additional fields are kept as plain attributes via
    ``PretrainedConfig.__init__(**kwargs)`` so user-defined metadata is
    preserved end-to-end.
    """

    model_type = MERAK_MODEL_TYPE
    # Tell HF this is a "decoder-only" causal LM-shaped config.
    is_composition = False
    keys_to_ignore_at_inference: ClassVar[list[str]] = []

    @staticmethod
    def _coerce_nested_config(value: Any) -> Any:
        if isinstance(value, PretrainedConfig):
            return value
        if isinstance(value, dict):
            return PretrainedConfig(**value)
        return value

    def __init__(
        self,
        xh_model: dict,
        config_format: Optional[str] = None,
        load_format: Optional[str] = None,
        enable_page_attention: Optional[bool] = None,
        use_v2: Optional[bool] = None,
        auto_offload: Optional[bool] = None,
        cuda_graph: Optional[bool] = None,
        chip_arch: Optional[str] = None,
        kv_cache_format: str = "float16",
        kv_cache_nshare: int = HMFP_NSHARE,
        kv_cache_k_qdata_dtype: str = "int8",
        kv_cache_v_qdata_dtype: str = "int8",
        kv_cache_scale_exp_dtype: str = "int8",
        kv_cache_slot_alignment: int = 16,
        **kwargs: Any,
    ) -> None:
        kv_cache_nshare = kwargs.pop("kv_cache_group_size", kv_cache_nshare)

        # vLLM relies on these attributes during ``ModelConfig`` validation.
        for nested_config_name in ("text_config", "vision_config"):
            if nested_config_name in kwargs:
                kwargs[nested_config_name] = self._coerce_nested_config(
                    kwargs[nested_config_name]
                )

        # Hand the rest to ``PretrainedConfig``.
        self.xh_model = xh_model
        self.config_format = config_format
        self.load_format = load_format
        self.enable_page_attention = (
            None if enable_page_attention is None else bool(enable_page_attention)
        )
        self.use_v2 = None if use_v2 is None else bool(use_v2)
        self.auto_offload = None if auto_offload is None else bool(auto_offload)
        self.cuda_graph = None if cuda_graph is None else bool(cuda_graph)
        self.chip_arch = chip_arch
        self.kv_cache_format = _normalize_kv_cache_format(kv_cache_format)
        self.kv_cache_nshare = kv_cache_nshare
        self.kv_cache_k_qdata_dtype = kv_cache_k_qdata_dtype
        self.kv_cache_v_qdata_dtype = kv_cache_v_qdata_dtype
        self.kv_cache_scale_exp_dtype = kv_cache_scale_exp_dtype
        self.kv_cache_slot_alignment = kv_cache_slot_alignment

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        config_dict = super().to_dict()
        contract = getattr(self, "gemma4_cache_contract", None)
        serialize_contract = getattr(contract, "to_dict", None)
        if callable(serialize_contract):
            config_dict["gemma4_cache_contract"] = serialize_contract()
        return config_dict

    def get_text_config(
        self,
        decoder: Optional[bool] = None,
        encoder: Optional[bool] = None,
    ) -> PretrainedConfig:
        text_config = super().get_text_config(decoder=decoder, encoder=encoder)
        coerced = self._coerce_nested_config(text_config)
        if (
            coerced is not text_config
            and getattr(self, "text_config", None) is text_config
        ):
            self.text_config = coerced
        return coerced


# ---------------------------------------------------------------------------
# Config parser
# ---------------------------------------------------------------------------


def _locate_merak_config_file(model: Union[str, Path]) -> Path | None:
    """Return the ``merak_config.json`` path under ``model``, or ``None``.

    Returning ``None`` means the input isn't a Merak directory and the parser
    should delegate to the standard HF parser. We deliberately do *not* fall
    back to ``config.json``: that fallback used to mask non-Merak inputs and
    blow up later inside ``MerakConfig.__init__``.
    """
    p = Path(model)
    if p.is_file():
        return p if p.name == _MERAK_CONFIG_FILE_NAME else None
    if not p.is_dir():
        return None
    candidate = p / _MERAK_CONFIG_FILE_NAME
    return candidate if candidate.is_file() else None


def _resolve_model_dir(model: Union[str, Path]) -> Path:
    model_path = Path(model)
    return model_path if model_path.is_dir() else model_path.parent


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_hf_assets_dir(model_dir: Path) -> Path:
    hf_assets_dir = model_dir / "hf_config"
    return hf_assets_dir if hf_assets_dir.is_dir() else model_dir


def _load_source_hf_config_dict(model_dir: Path) -> dict[str, Any]:
    hf_config_path = _resolve_hf_assets_dir(model_dir) / "config.json"
    if not hf_config_path.is_file():
        return {}
    return _read_json_file(hf_config_path)


def _register_source_transformers_layer_types(source_hf_config: dict[str, Any]) -> None:
    """Allow source-only layer labels needed while loading tokenizer assets."""

    layer_types = source_hf_config.get("mlp_layer_types")
    if not isinstance(layer_types, list) or not any(
        str(layer_type).lower() == "hash_moe" for layer_type in layer_types
    ):
        return

    _register_deepseek_v4_layer_type()


def _register_deepseek_v4_layer_type() -> None:
    """Make direct child-process loads accept DeepSeek-V4's ``hash_moe`` label."""

    # V1 EngineCore reloads the tokenizer/config in a separate process. That
    # path starts at ``<artifact>/hf_config`` and never visits the outer Merak
    # parser, so the compatibility registration must also happen at module
    # import time rather than only while merging merak_config.json.
    import transformers.configuration_utils as transformers_config

    allowed = transformers_config.ALLOWED_LAYER_TYPES
    if "hash_moe" not in allowed:
        transformers_config.ALLOWED_LAYER_TYPES = (*allowed, "hash_moe")


_register_deepseek_v4_layer_type()


def _isolate_source_weight_format(
    source_hf_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detach source-checkpoint weight metadata from the runtime config.

    A Merak export loads HMONNX initializers, not the copied Hugging Face
    checkpoint.  Leaving the source ``quantization_config`` at the standard HF
    locations makes vLLM select and validate an unrelated HF quantization
    backend (for example GPTQ) before the Merak loader is reached.  Keep a
    diagnostic copy under a Merak-owned namespace, while ensuring that only an
    explicit Merak config or ``hf_overrides`` can request an HF quantizer.
    """
    runtime_config = deepcopy(source_hf_config)
    source_weight_format: dict[str, Any] = {}

    model_weight_format = runtime_config.pop("quantization_config", None)
    if model_weight_format is not None:
        source_weight_format["model"] = model_weight_format

    text_config = runtime_config.get("text_config")
    if isinstance(text_config, dict):
        text_weight_format = text_config.pop("quantization_config", None)
        if text_weight_format is not None:
            source_weight_format["text"] = text_weight_format

    return runtime_config, source_weight_format


def _load_hmonnx_meta_info(
    model_dir: Path, merak_config_dict: dict[str, Any]
) -> dict[str, Any]:
    xh_model = merak_config_dict.get("xh_model", {})
    if not isinstance(xh_model, dict):
        return {}

    meta_info_name = str(xh_model.get("meta_info") or "golden_meta_info.json")
    meta_info_path = model_dir / meta_info_name
    if not meta_info_path.is_file():
        return {}
    return _read_json_file(meta_info_path)


def _coerce_architecture_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(architecture) for architecture in value if architecture]
    return []


def _extract_source_architectures(source_hf_config: dict[str, Any]) -> list[str]:
    architectures = _coerce_architecture_list(source_hf_config.get("architectures"))
    if architectures:
        return architectures

    text_config = source_hf_config.get("text_config")
    if isinstance(text_config, dict):
        return _coerce_architecture_list(text_config.get("architectures"))

    return []


def _resolve_export_artifact(model_dir: Path, artifact: str) -> Path:
    path = Path(artifact)
    return path if path.is_absolute() else model_dir / path


def _validate_qwen3_5_visual_gears(
    model_dir: Path,
    visual_config: dict[str, Any],
) -> None:
    gears = visual_config.get("gears")
    if gears is None:
        return
    if not isinstance(gears, list) or not gears:
        raise ValueError("Qwen3.5 visual_config.gears must be a non-empty JSON array.")

    required_contract = {
        "visual_input_mode": "patches",
        "attention_mask_format": "additive_key_padding_bias",
        "attention_mask_shape": "[1,1,1,patch_token_capacity]",
        "attention_mask_operator": "xhquant.nn.MaskedAdd+Softmax",
        "rotary_position_format": "height_width_2d_ids",
    }
    for field, expected in required_contract.items():
        actual = visual_config.get(field)
        if actual != expected:
            raise ValueError(
                f"Qwen3.5 visual token gears require {field}={expected!r}, got {actual!r}."
            )

    merge_size = int(visual_config.get("spatial_merge_size", 2))
    try:
        rope_cache_length = int(visual_config["visual_rope_cache_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Qwen3.5 visual token gears require a positive visual_rope_cache_length."
        ) from exc
    if rope_cache_length <= 0:
        raise ValueError(
            "Qwen3.5 visual token gears require a positive visual_rope_cache_length."
        )

    capacities: list[int] = []
    for index, gear in enumerate(gears):
        if not isinstance(gear, dict):
            raise ValueError(f"Qwen3.5 visual gear {index} must be a JSON object.")
        try:
            capacity = int(gear["image_token_capacity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Qwen3.5 visual gear {index} is missing a positive image_token_capacity."
            ) from exc
        if capacity <= 0:
            raise ValueError(
                f"Qwen3.5 visual gear {index} has invalid image_token_capacity {capacity}."
            )
        capacities.append(capacity)

        try:
            patch_capacity = int(gear["patch_token_capacity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Qwen3.5 visual gear {capacity} is missing a positive patch_token_capacity."
            ) from exc
        expected_patch_capacity = capacity * merge_size * merge_size
        if patch_capacity != expected_patch_capacity:
            raise ValueError(
                f"Qwen3.5 visual gear {capacity} declares patch_token_capacity "
                f"{patch_capacity}, expected {expected_patch_capacity}."
            )

        gear_hmonnx = gear.get("hmonnx")
        if not isinstance(gear_hmonnx, str) or not gear_hmonnx.strip():
            raise ValueError(
                f"Qwen3.5 visual gear {capacity} is missing its HMONNX path."
            )
        gear_path = _resolve_export_artifact(model_dir, gear_hmonnx)
        if not gear_path.is_file():
            raise FileNotFoundError(
                f"Qwen3.5 visual gear {capacity} HMONNX artifact does not exist: {gear_path}"
            )

    if capacities != sorted(set(capacities)):
        raise ValueError(
            "Qwen3.5 visual gear capacities must be strictly increasing and unique, "
            f"got {capacities}."
        )
    minimum_rope_cache_length = capacities[-1] * merge_size
    if rope_cache_length < minimum_rope_cache_length:
        raise ValueError(
            "Qwen3.5 visual_rope_cache_length does not cover the largest gear: "
            f"got {rope_cache_length}, need at least {minimum_rope_cache_length}."
        )

    manifest = visual_config.get("gear_manifest")
    if manifest is not None:
        if not isinstance(manifest, str) or not manifest.strip():
            raise ValueError(
                "Qwen3.5 visual_config.gear_manifest must be a non-empty path."
            )
        manifest_path = _resolve_export_artifact(model_dir, manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Qwen3.5 visual gear manifest does not exist: {manifest_path}"
            )


def _resolve_registered_architecture(
    model_dir: Path,
    source_architectures: list[str],
    hmonnx_meta_info: dict[str, Any],
) -> tuple[str, bool | None]:
    model_config = hmonnx_meta_info.get("model_config")
    is_wemm_embedding = (
        isinstance(model_config, dict)
        and model_config.get("model_type") == "WeMMEmbedding"
    )
    if is_wemm_embedding:
        expected_embedding_contract = {
            "embedding_output": "hidden_states",
            "embedding_pooling": "last_token",
            "embedding_normalize": True,
        }
        for field, expected in expected_embedding_contract.items():
            actual = hmonnx_meta_info.get(field)
            if actual != expected:
                raise ValueError(
                    f"WeMM embedding metadata requires {field}={expected!r}, got {actual!r}."
                )
        embedding_dimension = hmonnx_meta_info.get("embedding_dimension")
        if type(embedding_dimension) is not int or embedding_dimension <= 0:
            raise ValueError(
                "WeMM embedding metadata requires a positive integer "
                f"embedding_dimension, got {embedding_dimension!r}."
            )
        return _WEMM_EMBEDDING_ARCHITECTURE, True

    if any(
        architecture in _GEMMA4_UNIFIED_SOURCE_ARCHITECTURES
        for architecture in source_architectures
    ):
        return _GEMMA4_UNIFIED_LOADER_ARCHITECTURE, None

    if any(
        architecture in _GEMMA4_TOWER_SOURCE_ARCHITECTURES
        for architecture in source_architectures
    ):
        return _GEMMA4_LOADER_ARCHITECTURE, None

    if any(
        architecture in _QWEN3_5_SOURCE_ARCHITECTURES
        for architecture in source_architectures
    ):
        visual_config = hmonnx_meta_info.get("visual_config")
        if visual_config is None:
            return _QWEN3_5_TEXT_ARCHITECTURE, False
        if not isinstance(visual_config, dict):
            raise ValueError(
                "Qwen3.5 visual_config must be a JSON object when image input is exported."
            )

        visual_hmonnx = visual_config.get("hmonnx")
        if not isinstance(visual_hmonnx, str) or not visual_hmonnx.strip():
            raise ValueError(
                "Qwen3.5 visual_config must declare a non-empty visual HMONNX path."
            )
        visual_path = _resolve_export_artifact(model_dir, visual_hmonnx)
        if not visual_path.is_file():
            raise FileNotFoundError(
                f"Qwen3.5 visual HMONNX artifact does not exist: {visual_path}"
            )
        _validate_qwen3_5_visual_gears(model_dir, visual_config)
        return _QWEN3_5_MULTIMODAL_ARCHITECTURE, True

    for architecture in source_architectures:
        model_cls = get_registered_model_cls(architecture)
        if model_cls is not None:
            return model_cls.__name__, None

    return MERAK_DEFAULT_ARCH, None


def _infer_meta_info_sefp_kv_cache_format(
    model_dir: Path,
    hmonnx_meta_info: dict[str, Any],
) -> str | None:
    """Infer the SEFP K/V payload dtype from HMONNX FlashAttention nodes."""
    import onnx
    from onnx.helper import get_attribute_value

    bit_pairs: set[tuple[int, int]] = set()
    for field_name in ("prefill_hmonnx", "decode_hmonnx"):
        graph_name = hmonnx_meta_info.get(field_name)
        if not isinstance(graph_name, str) or not graph_name:
            continue

        graph_path = Path(graph_name)
        if not graph_path.is_absolute():
            graph_path = model_dir / graph_path
        if not graph_path.is_file():
            continue

        graph = onnx.load(graph_path, load_external_data=False).graph
        for node in graph.node:
            if node.op_type != "FlashAttention":
                continue
            attributes = {
                attribute.name: get_attribute_value(attribute)
                for attribute in node.attribute
            }
            k_bits = attributes.get("k_bits")
            v_bits = attributes.get("v_bits")
            if type(k_bits) is not int or type(v_bits) is not int:
                raise ValueError(
                    f"HMONNX FlashAttention node {node.name!r} in {graph_path} must declare integer k_bits and v_bits"
                )
            if k_bits not in (8, 16) or v_bits not in (8, 16):
                raise ValueError(
                    "HMONNX FlashAttention k_bits/v_bits must each be exactly 8 or 16; "
                    f"got k_bits={k_bits!r}, v_bits={v_bits!r} from node {node.name!r} in {graph_path}"
                )
            bit_pairs.add((k_bits, v_bits))

    if not bit_pairs:
        return None
    if len(bit_pairs) != 1:
        raise ValueError(
            f"HMONNX FlashAttention nodes disagree on k_bits/v_bits: {sorted(bit_pairs)}"
        )

    k_bits, v_bits = next(iter(bit_pairs))
    dtype_by_bits = {8: "int8", 16: "int16"}
    return f"sefp_k{dtype_by_bits[k_bits]}_v{dtype_by_bits[v_bits]}"


def _merge_merak_config(
    model_dir: Path,
    merak_config_dict: dict[str, Any],
) -> dict[str, Any]:
    source_hf_config = _load_source_hf_config_dict(model_dir)
    _register_source_transformers_layer_types(source_hf_config)
    hmonnx_meta_info = _load_hmonnx_meta_info(model_dir, merak_config_dict)

    merged_config, source_weight_format = _isolate_source_weight_format(
        source_hf_config
    )
    # ``MerakConfig`` is only the scheduler-facing shell around an HMONNX
    # artifact; the concrete loader reads architecture-specific fields from
    # ``hf_config/config.json`` itself.  New Transformers validators reject
    # DeepSeek-V4's source-only ``hash_moe`` MLP labels, even though they are
    # not consumed by the Merak scheduler or the HMONNX runtime.  Keep the
    # source file authoritative and detach only the incompatible shell field.
    source_mlp_layer_types = merged_config.get("mlp_layer_types")
    if (
        isinstance(source_mlp_layer_types, list)
        and any(
            str(layer_type).lower() == "hash_moe"
            for layer_type in source_mlp_layer_types
        )
        and "mlp_layer_types" not in merak_config_dict
    ):
        merged_config.pop("mlp_layer_types", None)
    merged_config.update(merak_config_dict)
    if source_weight_format:
        merged_config.setdefault(_MERAK_SOURCE_WEIGHT_FORMAT_KEY, source_weight_format)

    text_config = source_hf_config.get("text_config")
    if isinstance(text_config, dict):
        for key in (
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "head_dim",
            "max_position_embeddings",
            "rope_parameters",
        ):
            value = text_config.get(key)
            if value is not None:
                merged_config.setdefault(key, value)

        text_model_type = text_config.get("model_type")
        if text_model_type is not None:
            merged_config.setdefault("text_model_type", text_model_type)

    model_config = hmonnx_meta_info.get("model_config", {})
    if isinstance(model_config, dict):
        context_max_length = model_config.get("context_max_length")
        if context_max_length is not None:
            export_context_length = int(context_max_length)
            source_max_length = int(
                merged_config.get("max_position_embeddings", export_context_length)
            )
            merged_config["max_position_embeddings"] = min(
                source_max_length, export_context_length
            )

            merged_text_config = merged_config.get("text_config")
            if isinstance(merged_text_config, dict):
                text_source_max_length = int(
                    merged_text_config.get(
                        "max_position_embeddings", export_context_length
                    )
                )
                merged_text_config["max_position_embeddings"] = min(
                    text_source_max_length, export_context_length
                )

        for key in (
            "image_token_id",
            "audio_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "spatial_merge_size",
        ):
            value = model_config.get(key)
            if value is not None:
                merged_config[key] = value

        model_type = model_config.get("model_type")
        if model_type is not None:
            merged_config["source_model_type"] = model_type

        chip_arch = model_config.get("chip_arch")
        if chip_arch is not None:
            merged_config["chip_arch"] = chip_arch

        split_conv_cache = model_config.get("split_conv_cache")
        if split_conv_cache is not None:
            merged_config["split_conv_cache"] = bool(split_conv_cache)

    # PageAttention must use the K/V payload dtype declared by the exported
    # HMONNX FlashAttention nodes. A concrete user format remains an override.
    if (
        merged_config.get("enable_page_attention") is True
        and merged_config.get("kv_cache_format") is None
    ):
        inferred_kv_cache_format = _infer_meta_info_sefp_kv_cache_format(
            model_dir, hmonnx_meta_info
        )
        if inferred_kv_cache_format is not None:
            merged_config["kv_cache_format"] = inferred_kv_cache_format

    kv_cache_meta = hmonnx_meta_info.get("kv_cache", {})
    if isinstance(kv_cache_meta, dict):
        linear_kv_cache_config = kv_cache_meta.get("linear_kv_cache_config")
        if isinstance(linear_kv_cache_config, dict):
            merged_config["linear_kv_cache_config"] = dict(linear_kv_cache_config)

    visual_config = hmonnx_meta_info.get("visual_config")
    if isinstance(visual_config, dict):
        merged_config["visual_config"] = visual_config

    source_architectures = _extract_source_architectures(source_hf_config)
    if source_architectures:
        merged_config["source_architectures"] = source_architectures

    if any(
        architecture in _GEMMA4_SOURCE_ARCHITECTURES
        for architecture in source_architectures
    ):
        from .model_executor.model_loader.gemma4_cache_contract import (
            parse_gemma4_cache_contract,
        )

        parse_gemma4_cache_contract(
            hf_config=source_hf_config,
            hmonnx_meta=hmonnx_meta_info,
        )
        for field in _GEMMA4_CONTRACT_FIELDS:
            merged_config[field] = deepcopy(hmonnx_meta_info[field])

    loader_architecture, image_input_supported = _resolve_registered_architecture(
        model_dir,
        source_architectures,
        hmonnx_meta_info,
    )
    merged_config["architectures"] = [loader_architecture]
    merged_config["merak_loader_architecture"] = loader_architecture
    loader_cls = get_registered_custom_architectures().get(loader_architecture)
    hybrid_cache_policy = getattr(loader_cls, "hybrid_cache_policy", None)
    if hybrid_cache_policy is not None:
        if hybrid_cache_policy not in SUPPORTED_HYBRID_CACHE_POLICIES:
            raise ValueError(
                f"Loader {loader_architecture} declares unsupported hybrid "
                f"cache policy {hybrid_cache_policy!r}"
            )
        merged_config["merak_hybrid_cache_policy"] = hybrid_cache_policy
    if image_input_supported is not None:
        merged_config["image_input_supported"] = image_input_supported
    merged_config["model_type"] = MERAK_MODEL_TYPE

    # Set default runner_type if not already specified
    if "runner_type" not in merged_config:
        merged_config["runner_type"] = "generate"

    return merged_config


def _apply_hf_overrides(
    config_dict: dict[str, Any], hf_overrides: Any
) -> dict[str, Any]:
    if hf_overrides is None:
        return config_dict
    if isinstance(hf_overrides, dict):
        config_dict.update(hf_overrides)
        return config_dict
    if callable(hf_overrides):
        config = MerakConfig(**config_dict)
        updated_config = hf_overrides(config)
        if updated_config is None:
            updated_config = config
        if not isinstance(updated_config, PretrainedConfig):
            raise TypeError(
                "hf_overrides callable must return a PretrainedConfig instance or None."
            )
        config_dict.update(updated_config.to_dict())
        return config_dict
    raise TypeError("hf_overrides must be a dict, callable, or None.")


@register_config_parser(MERAK_MODEL_TYPE)
class MerakConfigParser(ConfigParserBase):
    """vLLM ``config_format='merak_llm'`` parser.

    For paths that contain a ``merak_config.json``, parse the Merak schema
    and return a :class:`MerakConfig`. For any other path (e.g. vLLM passing
    ``hf_config_path = <model>/hf_config`` for processor / generation-config
    bootstrapping on multimodal models), delegate to the standard
    :class:`HFConfigParser` so we don't try to coerce a plain HF directory
    through the Merak schema.
    """

    def __init__(self) -> None:
        super().__init__()
        self._hf_parser = HFConfigParser()

    def parse(  # type: ignore[override]
        self,
        model: Union[str, Path],
        trust_remote_code: bool,
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[dict, PretrainedConfig]:
        config_path = _locate_merak_config_file(model)
        if config_path is None:
            return self._hf_parser.parse(
                model,
                trust_remote_code,
                revision=revision,
                code_revision=code_revision,
                **kwargs,
            )

        del trust_remote_code, revision, code_revision
        hf_overrides = kwargs.pop("hf_overrides", None)
        merak_config_dict = _read_json_file(config_path)
        model_dir = _resolve_model_dir(model)
        config_dict = _merge_merak_config(model_dir, merak_config_dict)
        config_dict = _apply_hf_overrides(config_dict, hf_overrides)
        source_hf_config = _load_source_hf_config_dict(model_dir)
        source_architectures = _extract_source_architectures(source_hf_config)
        is_gemma4 = any(
            architecture in _GEMMA4_SOURCE_ARCHITECTURES
            for architecture in source_architectures
        )
        hmonnx_meta_info: dict[str, Any] | None = None
        if is_gemma4:
            hmonnx_meta_info = _load_hmonnx_meta_info(model_dir, merak_config_dict)
            for field in _GEMMA4_CONTRACT_FIELDS:
                config_dict[field] = deepcopy(hmonnx_meta_info[field])
            config_dict["source_architectures"] = source_architectures
            loader_architecture = (
                _GEMMA4_UNIFIED_LOADER_ARCHITECTURE
                if any(
                    architecture in _GEMMA4_UNIFIED_SOURCE_ARCHITECTURES
                    for architecture in source_architectures
                )
                else _GEMMA4_LOADER_ARCHITECTURE
            )
            config_dict["architectures"] = [loader_architecture]
            config_dict["merak_loader_architecture"] = loader_architecture
        config = MerakConfig(**config_dict)
        config._name_or_path = str(model_dir)
        if is_gemma4:
            from .model_executor.model_loader.gemma4_cache_contract import (
                parse_gemma4_cache_contract,
            )

            config.gemma4_cache_contract = parse_gemma4_cache_contract(
                hf_config=source_hf_config,
                hmonnx_meta=hmonnx_meta_info or {},
            )
        return config_dict, config
