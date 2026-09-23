#!/usr/bin/env python3
"""Gemma4 Series MTP HMONNX verifier and greedy speculative runner.

Series-local utility (do not use legacy gemma4/gemma4e/gemma4_moe paths):

* ``verify`` checks both supported target/draft graph contracts:
  legacy drafts consume shared target KV tensors with read-only KVcache nodes;
  FlashAttention drafts bind target-owned paged caches out of band through
  PageAttentionReadOnly.  In both forms the assistant head is W4 and its body
  is W8.
* ``generate`` runs a small greedy MTP loop on HMONNX target + ONNX assistant.
  Each verify round feeds ``[current_token] + draft_tokens`` to the target decode
  graph, so the exported decode sequence length must be ``num_draft_tokens + 1``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_llm.models.gemma4_series.mtp_contract import (  # noqa: E402, I001, F401
    resolve_readonly_sliding_cache_range,
)

# Retain diagnostic helper imports for existing callers of this example.
from xhmodel_merak.xh_llm.models.gemma4_series.mtp_runtime import (  # noqa: E402, F401
    _aligned,
    _meta_spec_decode,
    AssistantDraftSession,
    _load_prompt_inputs,
    _read_json_config,
    _generation_config_candidates,
    _resolve_eos_token_ids,
    _parse_target_outputs,
    _scalar_debug_value,
    _run_target_prefill,
    _run_target_verify,
    _cache_to_tensor,
    _clone_cache_tensor,
    _snapshot_hmonnx_kv_cache,
    _hmonnx_shared_cache_indices,
    _hmonnx_shared_from_cache,
    _commit_one_hmonnx_cache,
    _commit_hmonnx_verified_cache,
    _target_needs_manual_cache_commit,
    _build_draft_masks,
    _hmonnx_shared_cache_valid_lengths,
    _build_assistant_inputs,
    generate_with_mtp,
)

EXPECTED_BASE_OUTPUTS = ["logits", "target_hidden_state"]
LEGACY_DRAFT_INPUTS = [
    "input_1",
    "valid_length",
    "current_length",
    "sliding_attention_mask",
    "shared_key_cache_sliding",
    "shared_value_cache_sliding",
    "shared_key_cache_full",
    "shared_value_cache_full",
]
FLASH_DRAFT_INPUTS = [
    "input_1",
    "valid_length",
    "current_length",
    "shared_key_cache_sliding",
    "shared_value_cache_sliding",
    "shared_key_cache_full",
    "shared_value_cache_full",
    "kv_window_start_abs",
    "kv_valid_length",
]
READONLY_PAGE_DRAFT_INPUTS = [
    "input_1",
    "valid_length",
    "current_length",
    "kv_window_start_abs",
    "kv_valid_length",
]
MIXED_READONLY_PAGE_DRAFT_INPUTS = [
    "input_1",
    "valid_length",
    "current_length",
    "sliding_attention_mask",
]
EXPECTED_DRAFT_OUTPUTS = ["logits", "assistant_hidden_state"]


@dataclass(frozen=True)
class Preset:
    name: str
    slug: str
    assistant_kind: str


PRESETS: dict[str, Preset] = {
    "12b-unified": Preset(
        "12b-unified",
        "gemma4_12b_unified",
        "dense_lm_head",
    ),
    "e2b": Preset("e2b", "gemma4_e2b_unified", "ordered_embedding"),
    "e4b": Preset("e4b", "gemma4_e4b_unified", "ordered_embedding"),
    "26b-a4b": Preset("26b-a4b", "gemma4_26b_a4b_unified", "dense_lm_head"),
    "31b": Preset("31b", "gemma4_31b_unified", "dense_lm_head"),
}
DEFAULT_HF_DIRS: dict[str, tuple[str, str]] = {
    "12b-unified": (
        "weights/gemma-4-12B-it",
        "weights/gemma-4-12B-it-assistant",
    ),
    "e2b": ("weights/gemma-4-E2B-it", "weights/gemma-4-E2B-it-assistant"),
    "e4b": ("weights/gemma-4-E4B-it", "weights/gemma-4-E4B-it-assistant"),
    "26b-a4b": ("weights/gemma-4-26B-A4B-it", "weights/gemma-4-26B-A4B-it-assistant"),
    "31b": ("weights/gemma-4-31B-it", "weights/gemma-4-31B-it-assistant"),
}


def _attr_value(attr: onnx.AttributeProto) -> Any:
    value = onnx.helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _node_attrs(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: _attr_value(attr) for attr in node.attribute}


def _onnx_dim(dim) -> int:
    if getattr(dim, "dim_value", 0):
        return int(dim.dim_value)
    raise AssertionError(f"Dynamic ONNX dimension is not supported in Gemma4 MTP contract checks: {dim}")


def _onnx_value_shape(model: onnx.ModelProto, name: str) -> list[int]:
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value.name == name:
            return [_onnx_dim(dim) for dim in value.type.tensor_type.shape.dim]
    raise AssertionError(f"{name!r} not found in ONNX graph")


def _resolve_relative_path(path_value: str, base_dir: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else base_dir / path


def _latest_export_dir(base_export_root: Path) -> Path:
    if (base_export_root / "golden_meta_info.json").exists():
        return base_export_root
    candidates = sorted(path for path in base_export_root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No export directory found under {base_export_root}")
    return candidates[-1]


def _resolve_meta_path(preset: Preset, base_root: Path, meta: str | None) -> Path:
    if meta:
        meta_path = Path(meta).expanduser().resolve()
        if meta_path.is_dir():
            meta_path = meta_path / "golden_meta_info.json"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        return meta_path
    return _latest_export_dir(base_root / preset.slug / "existing-hf" / "export") / "golden_meta_info.json"


def _resolve_draft_onnx(
    preset: Preset,
    draft_root: Path,
    draft_onnx: str | None,
    *,
    meta_path: Path | None = None,
    meta_dict: dict[str, Any] | None = None,
    prefer_standalone: bool = False,
) -> Path:
    if draft_onnx:
        path = Path(draft_onnx).expanduser().resolve()
    elif meta_path is not None and meta_dict is not None:
        spec = _meta_spec_decode(meta_dict)
        draft_contract = spec.get("draft")
        if not isinstance(draft_contract, dict):
            draft_contract = {}
        standalone_rel = spec.get("standalone_draft_decode_onnx") or draft_contract.get("standalone_decode_hmonnx")
        rel = (
            (standalone_rel if prefer_standalone else None)
            or spec.get("draft_decode_onnx")
            or spec.get("draft_onnx")
            or meta_dict.get("draft_decode_onnx_file")
            or meta_dict.get("draft_onnx_file")
        )
        if not rel:
            raise FileNotFoundError(
                f"{preset.name}: manifest {meta_path} does not record spec_decode.draft_decode_onnx"
            )
        path = _resolve_relative_path(str(rel), meta_path.parent)
    else:
        path = draft_root / preset.name / "mtp_draft_decode" / f"gemma4_series_{preset.name}_assistant_decode.onnx"
        if not path.exists():
            path = draft_root / preset.name / "draft_onnx" / f"gemma4_series_{preset.name}_assistant_decode.onnx"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_outputs(onnx_path: Path) -> list[str]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return [out.name for out in model.graph.output]


def _load_inputs(onnx_path: Path) -> list[str]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return [inp.name for inp in model.graph.input]


def _quant_counter(model: onnx.ModelProto) -> Counter[tuple[int, int, str]]:
    counter: Counter[tuple[int, int, str]] = Counter()
    for node in model.graph.node:
        attrs = _node_attrs(node)
        if "hmfp_weight_man_bit" in attrs:
            counter[
                (
                    int(attrs["hmfp_weight_man_bit"]),
                    int(attrs.get("hmfp_weight_hidden_bit", -1)),
                    str(attrs.get("mode", "")),
                )
            ] += 1
    return counter


def _w4_nodes(model: onnx.ModelProto) -> list[tuple[str, str, dict[str, Any]]]:
    hits = []
    for node in model.graph.node:
        attrs = _node_attrs(node)
        if attrs.get("hmfp_weight_man_bit") == 4:
            hits.append((node.name, node.op_type, attrs))
    return hits


def _kv_nodes(model: onnx.ModelProto) -> list[tuple[str, str, dict[str, Any], list[str]]]:
    hits = []
    for node in model.graph.node:
        if node.op_type == "KVcache" or "cache" in node.op_type.lower():
            hits.append((node.name, node.op_type, _node_attrs(node), list(node.input)))
    return hits


def _target_decode_onnx_path(meta_path: Path, meta_dict: dict[str, Any]) -> Path:
    export_dir = meta_path.parent
    if meta_dict.get("decode_hmonnx"):
        return _resolve_relative_path(meta_dict["decode_hmonnx"], export_dir)
    return next((export_dir / "decode").glob("*.onnx"))


def _first_sliding_kv_trace_info(meta_path: Path, meta_dict: dict[str, Any]) -> dict[str, Any] | None:
    decode_onnx = _target_decode_onnx_path(meta_path, meta_dict)
    decode_model = onnx.load(str(decode_onnx), load_external_data=False)
    for node_name, op_type, attrs, node_inputs in _kv_nodes(decode_model):
        attention_max_length = int(attrs.get("attention_max_length", attrs.get("attention-max-length", -1)))
        if attention_max_length <= 0:
            continue
        return {
            "decode_onnx": str(decode_onnx),
            "node_name": node_name,
            "op_type": op_type,
            "attention_max_length": attention_max_length,
            "node_inputs": node_inputs,
            "accepted_count_input_index": node_inputs.index("accepted_count")
            if "accepted_count" in node_inputs
            else None,
        }
    return None


def verify_preset(
    preset: Preset, base_root: Path, draft_root: Path, *, meta: str | None = None, draft_onnx: str | None = None
) -> dict[str, Any]:
    meta_path = _resolve_meta_path(preset, base_root, meta)
    export_dir = meta_path.parent
    meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    model_config = meta_dict.get("model_config", {})
    spec = _meta_spec_decode(meta_dict)
    attention_contract_version = int(
        meta_dict.get("attention_contract_version") or model_config.get("attention_contract_version") or 1
    )
    target_attention_lowering = str(
        meta_dict.get("attention_lowering")
        or model_config.get("attention_lowering")
        or "legacy_attention"
    )
    uses_target_mixed_attention = (
        attention_contract_version >= 2
        and target_attention_lowering == "full_flash_attention"
    )
    uses_target_compact_flash_attention = (
        attention_contract_version >= 2
        and not uses_target_mixed_attention
    )
    if not model_config.get("enable_mtp_outputs"):
        raise AssertionError(f"{preset.name}: base export did not enable MTP outputs")
    if spec.get("mode") != "mtp" or meta_dict.get("spec_decode_mode") != "mtp":
        raise AssertionError(f"{preset.name}: manifest does not declare MTP spec_decode: {spec}")
    num_draft_tokens = int(spec.get("block_size") or model_config.get("num_draft_tokens") or 4)
    verify_length = int(spec.get("verify_length") or meta_dict.get("spec_decode_verify_length") or 0)
    if verify_length != num_draft_tokens + 1:
        raise AssertionError(
            f"{preset.name}: verify_length must be num_draft_tokens + 1, got {verify_length} vs {num_draft_tokens}"
        )

    prefill_onnx = (
        _resolve_relative_path(meta_dict.get("prefill_hmonnx", ""), export_dir)
        if meta_dict.get("prefill_hmonnx")
        else next((export_dir / "prefill").glob("*.onnx"))
    )
    decode_onnx = _target_decode_onnx_path(meta_path, meta_dict)
    prefill_model = onnx.load(str(prefill_onnx), load_external_data=False)
    decode_model = onnx.load(str(decode_onnx), load_external_data=False)
    prefill_outputs = [out.name for out in prefill_model.graph.output]
    decode_outputs = [out.name for out in decode_model.graph.output]
    if prefill_outputs != EXPECTED_BASE_OUTPUTS:
        raise AssertionError(f"{preset.name}: prefill outputs mismatch: {prefill_outputs}")
    if decode_outputs != EXPECTED_BASE_OUTPUTS:
        raise AssertionError(f"{preset.name}: decode outputs mismatch: {decode_outputs}")
    decode_input_shape = _onnx_value_shape(decode_model, "input_1")
    decode_inputs = _load_inputs(decode_onnx)
    if uses_target_compact_flash_attention:
        required_compact_inputs = {"kv_window_start_abs", "kv_valid_length"}
        missing = required_compact_inputs.difference(decode_inputs)
        if missing:
            raise AssertionError(
                f"{preset.name}: target FlashAttention decode graph is "
                f"missing compact metadata inputs {sorted(missing)}"
            )
        if "sliding_attention_mask" in decode_inputs:
            raise AssertionError(
                f"{preset.name}: target contract-v2 decode graph retains legacy sliding_attention_mask"
            )
    else:
        if "accepted_count" not in decode_inputs:
            raise AssertionError(
                f"{preset.name}: target MTP verify decode graph is missing accepted_count input"
            )
        if uses_target_mixed_attention and "sliding_attention_mask" not in decode_inputs:
            raise AssertionError(
                f"{preset.name}: mixed target MTP decode graph is missing sliding_attention_mask"
            )
    decode_kv_nodes = _kv_nodes(decode_model)
    if not decode_kv_nodes:
        raise AssertionError(f"{preset.name}: target MTP verify decode graph has no KVcache nodes")
    sliding_kv_nodes = []
    full_kv_nodes = []
    for node_name, _, attrs, node_inputs in decode_kv_nodes:
        attention_max_length = int(attrs.get("attention_max_length", attrs.get("attention-max-length", -1)))
        if attention_max_length > 0:
            sliding_kv_nodes.append(node_name)
            if not uses_target_compact_flash_attention and "accepted_count" not in node_inputs:
                raise AssertionError(
                    f"{preset.name}: target sliding decode KVcache {node_name} does not consume accepted_count"
                )
        else:
            full_kv_nodes.append(node_name)
            if "accepted_count" in node_inputs:
                raise AssertionError(
                    f"{preset.name}: target full decode KVcache {node_name} must not consume accepted_count"
                )
    if not sliding_kv_nodes:
        raise AssertionError(f"{preset.name}: target MTP verify decode graph has no sliding KVcache nodes")
    if not full_kv_nodes:
        raise AssertionError(f"{preset.name}: target MTP verify decode graph has no full-attention KVcache nodes")
    if decode_input_shape[1] != verify_length:
        raise AssertionError(
            f"{preset.name}: target decode input length must be verify_length={verify_length}, got {decode_input_shape}"
        )
    decode_logits_shape = _onnx_value_shape(decode_model, "logits")
    if decode_logits_shape[1] != verify_length:
        raise AssertionError(
            f"{preset.name}: target decode logits length must be "
            f"verify_length={verify_length}, got {decode_logits_shape}"
        )
    sliding_window = int(meta_dict.get("sliding_window") or model_config.get("sliding_window") or 0)
    target_decode_sliding_width = int(
        spec.get("target_decode_sliding_output_length")
        or (_aligned(sliding_window + verify_length - 1, 16) if sliding_window > 0 else 0)
    )
    if not uses_target_compact_flash_attention:
        decode_sliding_mask_shape = _onnx_value_shape(
            decode_model,
            "sliding_attention_mask",
        )
        if target_decode_sliding_width and decode_sliding_mask_shape[-1] != target_decode_sliding_width:
            raise AssertionError(
                f"{preset.name}: target decode sliding mask width must be "
                f"LLMCache compact output {target_decode_sliding_width}, "
                f"got {decode_sliding_mask_shape}"
            )
    layer_shapes = meta_dict.get("layer_kv_shapes") or []
    context_length = int(model_config.get("context_max_length") or 0)
    shape_lengths = [int(shape[2]) for shape in layer_shapes if len(shape) > 2]
    sliding_candidates = [length for length in shape_lengths if context_length <= 0 or length < context_length]
    shared_sliding_len = int(spec.get("shared_sliding_cache_length") or max(sliding_candidates or shape_lengths))
    shared_full_len = int(spec.get("shared_full_cache_length") or context_length or max(shape_lengths))
    sliding_shape = next(
        (shape for shape in layer_shapes if len(shape) > 2 and int(shape[2]) == shared_sliding_len), None
    )
    full_shape = next((shape for shape in layer_shapes if len(shape) > 2 and int(shape[2]) == shared_full_len), None)
    if sliding_shape is None or full_shape is None:
        raise AssertionError(
            f"{preset.name}: manifest layer_kv_shapes cannot locate shared KV lengths "
            f"sliding={shared_sliding_len}, full={shared_full_len}: {layer_shapes}"
        )
    target_shared_shapes = {
        "shared_key_cache_sliding": list(sliding_shape),
        "shared_value_cache_sliding": list(sliding_shape),
        "shared_key_cache_full": list(full_shape),
        "shared_value_cache_full": list(full_shape),
    }

    draft_path = _resolve_draft_onnx(preset, draft_root, draft_onnx, meta_path=meta_path, meta_dict=meta_dict)
    draft_model = onnx.load(str(draft_path), load_external_data=False)
    draft_inputs = [inp.name for inp in draft_model.graph.input]
    draft_outputs = [out.name for out in draft_model.graph.output]
    draft_tag_nodes = [node.name for node in draft_model.graph.node if node.op_type in {"Tag", "XHTag"}]
    if draft_tag_nodes:
        raise AssertionError(f"{preset.name}: assistant draft graph retains scheduling tags: {draft_tag_nodes}")
    draft_contract = dict(spec.get("draft") or {})
    uses_mixed_readonly_attention = (
        draft_contract.get("abi")
        == "gemma4_mtp_readonly_mixed_attention_v3"
    )
    readonly_page_nodes = [node for node in draft_model.graph.node if node.op_type == "PageAttentionReadOnly"]
    readonly_page_kv_nodes = [
        node
        for node in draft_model.graph.node
        if node.op_type == "PageKVCacheReadOnly"
    ]
    uses_readonly_page_attention = bool(
        readonly_page_nodes or readonly_page_kv_nodes
    ) or uses_mixed_readonly_attention
    expected_draft_inputs = (
        MIXED_READONLY_PAGE_DRAFT_INPUTS
        if uses_mixed_readonly_attention
        else READONLY_PAGE_DRAFT_INPUTS
        if uses_readonly_page_attention
        else LEGACY_DRAFT_INPUTS
    )
    kv_nodes: list[tuple[str, str, dict[str, Any], list[str]]] = []
    if draft_inputs != expected_draft_inputs:
        raise AssertionError(
            f"{preset.name}: "
            f"{'read-only PageAttention' if uses_readonly_page_attention else 'legacy/shared-tensor'} "
            f"draft inputs mismatch: {draft_inputs}"
        )
    if draft_outputs != EXPECTED_DRAFT_OUTPUTS:
        raise AssertionError(f"{preset.name}: draft outputs mismatch: {draft_outputs}")
    if uses_readonly_page_attention:
        # Production PageAttention binds the target-owned paged caches through
        # runtime context, so target-owned cache tensors are deliberately
        # absent from the graph ABI.
        if draft_contract.get("cache_binding") != "target_attention_type_owner":
            raise AssertionError(f"{preset.name}: read-only PageAttention draft must bind target_attention_type_owner")
        if any(node.op_type == "Clip" for node in draft_model.graph.node):
            raise AssertionError(
                f"{preset.name}: read-only PageAttention must not derive cache "
                "ranges with graph-side Clip"
            )
        if uses_mixed_readonly_attention:
            if draft_contract.get("attention_lowering") != "full_page_sliding_unfused":
                raise AssertionError(
                    f"{preset.name}: mixed read-only draft must use full-page/sliding-unfused lowering"
                )
            page_modules = [
                node
                for node in draft_model.graph.node
                if node.op_type
                in {"PageKVCacheReadOnly", "PageAttentionReadOnly"}
            ]
            module_attrs = [_node_attrs(node) for node in page_modules]
            layer_indices = [int(attrs.get("layer_idx", -1)) for attrs in module_attrs]
            if layer_indices != list(range(len(page_modules))):
                raise AssertionError(
                    f"{preset.name}: mixed read-only page module indices are not contiguous: {layer_indices}"
                )
            module_types = [str(attrs.get("attention_kind") or "") for attrs in module_attrs]
            declared_types = list(draft_contract.get("layer_attention_types") or [])
            if module_types != declared_types:
                raise AssertionError(
                    f"{preset.name}: mixed page modules disagree with manifest: "
                    f"graph={module_types}, manifest={declared_types}"
                )
            for node, attention_kind, attrs in zip(
                page_modules,
                module_types,
                module_attrs,
                strict=True,
            ):
                if attention_kind == "sliding_attention":
                    if (
                        node.op_type != "PageKVCacheReadOnly"
                        or int(attrs.get("sliding_window", -1)) <= 0
                        or list(node.input) != ["valid_length", "current_length"]
                    ):
                        raise AssertionError(
                            f"{preset.name}: invalid mixed sliding cache module {node.name}"
                        )
                elif attention_kind == "full_attention":
                    if (
                        node.op_type != "PageAttentionReadOnly"
                        or not bool(attrs.get("is_causal", 0))
                        or int(attrs.get("sliding_window", -1)) > 0
                        or "kv_window_start_abs" in node.input
                        or "kv_valid_length" in node.input
                    ):
                        raise AssertionError(
                            f"{preset.name}: invalid mixed full-attention module {node.name}"
                        )
                else:
                    raise AssertionError(
                        f"{preset.name}: unknown mixed attention kind {attention_kind!r}"
                    )
        else:
            if len(readonly_page_nodes) != 4:
                raise AssertionError(
                    f"{preset.name}: expected 4 PageAttentionReadOnly nodes, got {len(readonly_page_nodes)}"
                )
            if (
                draft_contract.get("abi") != "gemma4_mtp_readonly_page_attention_v2"
                or draft_contract.get("attention_lowering") != "causal"
            ):
                raise AssertionError(
                    f"{preset.name}: read-only PageAttention must use the causal v2 ABI"
                )
            for node in readonly_page_nodes:
                attrs = _node_attrs(node)
                inputs = list(node.input)
                if not bool(attrs.get("is_causal", 0)):
                    raise AssertionError(
                        f"{preset.name}: {node.name} must be causal"
                    )
                sliding_window_attr = int(attrs.get("sliding_window", -1))
                has_compact_range = (
                    len(inputs) > 7
                    and inputs[6] == "kv_window_start_abs"
                    and inputs[7] == "kv_valid_length"
                )
                if sliding_window_attr > 0 and not has_compact_range:
                    raise AssertionError(
                        f"{preset.name}: sliding {node.name} is missing compact "
                        "target-cache metadata"
                    )
                if sliding_window_attr <= 0 and has_compact_range:
                    raise AssertionError(
                        f"{preset.name}: full {node.name} must not consume sliding "
                        "target-cache metadata"
                    )
    else:
        for name, target_shape in target_shared_shapes.items():
            draft_shape = _onnx_value_shape(draft_model, name)
            if draft_shape != target_shape:
                raise AssertionError(
                    f"{preset.name}: draft {name} shape must match manifest shared KV cache for reuse; "
                    f"draft={draft_shape}, target={target_shape}"
                )
        if (
            _onnx_value_shape(draft_model, "sliding_attention_mask")[-1]
            != target_shared_shapes["shared_key_cache_sliding"][2]
        ):
            raise AssertionError(f"{preset.name}: draft sliding mask width does not match shared sliding KV length")
        kv_nodes = _kv_nodes(draft_model)
        if len(kv_nodes) != 4:
            raise AssertionError(f"{preset.name}: expected 4 KVcache nodes, got {len(kv_nodes)}")
        for node_name, _, attrs, _ in kv_nodes:
            if attrs.get("only_handle_old_cache") != 1:
                raise AssertionError(f"{preset.name}: {node_name} missing only_handle_old_cache=1: {attrs}")
            if "passthrough" in attrs:
                raise AssertionError(f"{preset.name}: {node_name} still has legacy passthrough attr")

    w4_nodes = _w4_nodes(draft_model)
    if len(w4_nodes) != 1:
        raise AssertionError(f"{preset.name}: expected exactly one W4 head node, got {len(w4_nodes)}")
    quant_counter = _quant_counter(draft_model)
    if quant_counter[(4, 0, "ssfp")] != 1:
        raise AssertionError(f"{preset.name}: missing W4A8H0 SSFP head: {quant_counter}")
    if quant_counter[(8, 1, "sefp")] <= 0:
        raise AssertionError(f"{preset.name}: missing W8A8H1 SEFP draft body: {quant_counter}")

    standalone_validation = None
    if uses_readonly_page_attention:
        standalone_path = _resolve_draft_onnx(
            preset,
            draft_root,
            None,
            meta_path=meta_path,
            meta_dict=meta_dict,
            prefer_standalone=True,
        )
        if standalone_path != draft_path:
            standalone_model = onnx.load(
                str(standalone_path),
                load_external_data=False,
            )
            standalone_inputs = [inp.name for inp in standalone_model.graph.input]
            standalone_outputs = [out.name for out in standalone_model.graph.output]
            expected_standalone_inputs = (
                LEGACY_DRAFT_INPUTS
                if uses_mixed_readonly_attention
                else FLASH_DRAFT_INPUTS
            )
            if standalone_inputs != expected_standalone_inputs:
                raise AssertionError(f"{preset.name}: standalone draft inputs mismatch: {standalone_inputs}")
            if standalone_outputs != EXPECTED_DRAFT_OUTPUTS:
                raise AssertionError(f"{preset.name}: standalone draft outputs mismatch: {standalone_outputs}")
            standalone_tags = [node.name for node in standalone_model.graph.node if node.op_type in {"Tag", "XHTag"}]
            if standalone_tags:
                raise AssertionError(f"{preset.name}: standalone draft retains scheduling tags: {standalone_tags}")
            if any(node.op_type == "Clip" for node in standalone_model.graph.node):
                raise AssertionError(
                    f"{preset.name}: standalone FlashAttention draft retains Clip"
                )
            standalone_flash_nodes = [
                node
                for node in standalone_model.graph.node
                if node.op_type == "FlashAttention"
            ]
            expected_standalone_flash_nodes = (
                sum(
                    attention_kind == "full_attention"
                    for attention_kind in (
                        draft_contract.get("layer_attention_types") or []
                    )
                )
                if uses_mixed_readonly_attention
                else 4
            )
            if len(standalone_flash_nodes) != expected_standalone_flash_nodes:
                raise AssertionError(
                    f"{preset.name}: standalone draft expected "
                    f"{expected_standalone_flash_nodes} FlashAttention "
                    f"nodes, got {len(standalone_flash_nodes)}"
                )
            for node in standalone_flash_nodes:
                attrs = _node_attrs(node)
                inputs = list(node.input)
                sliding_window_attr = int(attrs.get("sliding_window", -1))
                has_compact_range = (
                    len(inputs) > 9
                    and inputs[8] == "kv_window_start_abs"
                    and inputs[9] == "kv_valid_length"
                )
                if sliding_window_attr > 0 and not has_compact_range:
                    raise AssertionError(
                        f"{preset.name}: standalone sliding {node.name} is "
                        "missing compact target-cache metadata"
                    )
                if sliding_window_attr <= 0 and has_compact_range:
                    raise AssertionError(
                        f"{preset.name}: standalone full {node.name} consumes "
                        "sliding target-cache metadata"
                    )
            standalone_shapes = {name: _onnx_value_shape(standalone_model, name) for name in target_shared_shapes}
            if standalone_shapes != target_shared_shapes:
                raise AssertionError(
                    f"{preset.name}: standalone shared-cache shapes do not "
                    f"match target: draft={standalone_shapes}, "
                    f"target={target_shared_shapes}"
                )
            standalone_kv_nodes = _kv_nodes(standalone_model)
            if len(standalone_kv_nodes) != 4:
                raise AssertionError(
                    f"{preset.name}: standalone draft expected 4 read-only "
                    f"KVcache nodes, got {len(standalone_kv_nodes)}"
                )
            for node_name, _, attrs, _ in standalone_kv_nodes:
                if attrs.get("only_handle_old_cache") != 1:
                    raise AssertionError(
                        f"{preset.name}: standalone {node_name} missing only_handle_old_cache=1: {attrs}"
                    )
            standalone_w4 = _w4_nodes(standalone_model)
            standalone_quant = _quant_counter(standalone_model)
            if (
                len(standalone_w4) != 1
                or standalone_quant[(4, 0, "ssfp")] != 1
                or standalone_quant[(8, 1, "sefp")] <= 0
            ):
                raise AssertionError(
                    f"{preset.name}: standalone draft quant contract "
                    f"mismatch: w4={standalone_w4}, quant={standalone_quant}"
                )
            standalone_validation = {
                "onnx": str(standalone_path),
                "inputs": standalone_inputs,
                "outputs": standalone_outputs,
                "shared_shapes": standalone_shapes,
                "kv_cache_attrs": [attrs for _, _, attrs, _ in standalone_kv_nodes],
            }

    return {
        "preset": preset.name,
        "assistant_kind": preset.assistant_kind,
        "manifest": str(meta_path),
        "base_export": str(export_dir),
        "verify_length": verify_length,
        "num_draft_tokens": num_draft_tokens,
        "prefill_inputs": _load_inputs(prefill_onnx),
        "decode_inputs": decode_inputs,
        "decode_input_shape": decode_input_shape,
        "decode_logits_shape": decode_logits_shape,
        "target_attention_contract_version": attention_contract_version,
        "target_shared_shapes": target_shared_shapes,
        "base_outputs": decode_outputs,
        "draft_onnx": str(draft_path),
        "draft_inputs": draft_inputs,
        "draft_outputs": draft_outputs,
        "draft_attention_contract": (
            "readonly_mixed_attention"
            if uses_mixed_readonly_attention
            else "readonly_page_attention"
            if uses_readonly_page_attention
            else "shared_tensor"
        ),
        "standalone_draft": standalone_validation,
        "draft_shared_shapes": (
            {}
            if uses_readonly_page_attention
            else {name: _onnx_value_shape(draft_model, name) for name in target_shared_shapes}
        ),
        "kv_cache_attrs": [attrs for _, _, attrs, _ in kv_nodes],
        "target_kv_cache_inputs": {node_name: node_inputs for node_name, _, _, node_inputs in decode_kv_nodes},
        "w4_head_node": {"name": w4_nodes[0][0], "op_type": w4_nodes[0][1], "attrs": w4_nodes[0][2]},
        "quant_counter": {str(key): value for key, value in sorted(quant_counter.items())},
    }


def _tensor_dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _torch_dtype(dtype_name: str):
    import torch

    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(dtype_name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; choose one of {sorted(mapping)}")
    return mapping[key]


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


class TorchAssistantDraftSession:
    """Eager fp assistant draft session for acceptance-rate diagnosis.

    It intentionally implements the same small callable surface as
    :class:`AssistantDraftSession` so the speculative loop can isolate the
    effect of draft quantization while keeping the target prefill/verify path
    unchanged.
    """

    def __init__(
        self,
        *,
        assistant_model_dir: str | Path,
        target_model_dir: str | Path,
        meta: dict[str, Any],
        device,
        dtype,
    ):
        import torch

        from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model import Gemma4AssistantDraftModule

        model_config = meta.get("model_config", {})
        spec = _meta_spec_decode(meta)
        context_length = int(model_config.get("context_max_length") or model_config.get("max_sequence_length") or 2048)
        input_sequence_length = int(model_config.get("mtp_config", {}).get("input_sequence_length") or 1)
        self.module = Gemma4AssistantDraftModule(
            assistant_model_dir=str(assistant_model_dir),
            target_model_dir=str(target_model_dir),
            max_position_embeddings=context_length,
            input_sequence_length=input_sequence_length,
            cache_axis=int(meta.get("kv_cache", {}).get("cache_axis") or 2),
        ).to(device=device, dtype=dtype)
        self.module.eval()
        self.device = device
        self.dtype = dtype
        shared_sliding = int(
            spec.get("shared_sliding_cache_length") or meta.get("kv_cache", {}).get("kv_cache_shape", [0, 0, 0, 0])[2]
        )
        shared_full = int(
            spec.get("shared_full_cache_length") or model_config.get("context_max_length") or context_length
        )
        target_text = json.loads((Path(target_model_dir) / "config.json").read_text(encoding="utf-8"))["text_config"]
        full_heads = int(target_text.get("num_global_key_value_heads") or target_text["num_key_value_heads"])
        full_dim = int(target_text.get("global_head_dim") or target_text["head_dim"])
        sliding_heads = int(target_text["num_key_value_heads"])
        sliding_dim = int(target_text["head_dim"])
        hidden_size = int(self.module.backbone_hidden_size)
        self.input_names = list(LEGACY_DRAFT_INPUTS)
        self.output_names = list(EXPECTED_DRAFT_OUTPUTS)
        self.input_infos = {
            "input_1": SimpleNamespace(shape=(1, input_sequence_length, hidden_size * 2), dtype=dtype),
            "valid_length": SimpleNamespace(shape=(1,), dtype=torch.int32),
            "current_length": SimpleNamespace(shape=(1,), dtype=torch.int32),
            "sliding_attention_mask": SimpleNamespace(shape=(1, 1, input_sequence_length, shared_sliding), dtype=dtype),
            "shared_key_cache_sliding": SimpleNamespace(
                shape=(1, sliding_heads, shared_sliding, sliding_dim), dtype=dtype
            ),
            "shared_value_cache_sliding": SimpleNamespace(
                shape=(1, sliding_heads, shared_sliding, sliding_dim), dtype=dtype
            ),
            "shared_key_cache_full": SimpleNamespace(shape=(1, full_heads, shared_full, full_dim), dtype=dtype),
            "shared_value_cache_full": SimpleNamespace(shape=(1, full_heads, shared_full, full_dim), dtype=dtype),
        }
        self.input_aliases = {
            "input_1": "inputs_embeds",
            "valid_length": "past_seq_length",
            "current_length": "current_length",
        }

    def __call__(self, **inputs):
        import torch

        feed = {}
        for name in self.input_names:
            source_name = name if name in inputs else self.input_aliases.get(name)
            if source_name is None or source_name not in inputs:
                raise KeyError(f"Missing fp assistant input {name!r}; available={sorted(inputs)}")
            value = inputs[source_name]
            if isinstance(value, torch.Tensor):
                expected = self.input_infos[name]
                value = value.detach().to(self.device)
                if value.dtype != expected.dtype:
                    value = value.to(expected.dtype)
                if tuple(value.shape) != tuple(expected.shape):
                    raise RuntimeError(
                        f"FP assistant input {name!r} shape mismatch: got {tuple(value.shape)}, "
                        f"expected {tuple(expected.shape)}"
                    )
            feed[name] = value
        return self.module(
            feed["input_1"],
            feed["valid_length"],
            feed["current_length"],
            feed["sliding_attention_mask"],
            feed["shared_key_cache_sliding"],
            feed["shared_value_cache_sliding"],
            feed["shared_key_cache_full"],
            feed["shared_value_cache_full"],
        )


class TorchTargetSession:
    """Eager fp target verifier with Gemma4 Series MTP shared-cache outputs.

    This is a diagnosis-only backend for answering whether the low acceptance
    rate comes from quantized target prefill/verify.  It keeps the speculative
    loop unchanged and adapts HF ``DynamicCache`` into the same shared
    sliding/full cache tensors consumed by the exported assistant.
    """

    def __init__(self, *, target_model_dir: str | Path, meta: dict[str, Any], device, dtype):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_dir = str(target_model_dir)
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            dtype=dtype,
            device_map=str(device),
            trust_remote_code=True,
        ).eval()
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 9:
            for cfg in (
                getattr(self.model, "config", None),
                getattr(getattr(self.model, "config", None), "text_config", None),
            ):
                if cfg is not None and hasattr(cfg, "_experts_implementation"):
                    cfg._experts_implementation = "batched_mm"
        self.config_dict = json.loads((Path(self.model_dir) / "config.json").read_text(encoding="utf-8"))
        self.text_config = self.config_dict["text_config"]
        self.meta_info = SimpleNamespace(model_config=_to_namespace(meta.get("model_config", {})))
        self.sliding_window = int(self.text_config.get("sliding_window") or meta.get("sliding_window") or 1024)
        spec = _meta_spec_decode(meta)
        self.shared_sliding = int(
            spec.get("shared_sliding_cache_length") or meta.get("kv_cache", {}).get("kv_cache_shape", [0, 0, 0, 0])[2]
        )
        self.shared_full = int(
            spec.get("shared_full_cache_length") or meta.get("model_config", {}).get("context_max_length") or 2048
        )
        self._cache = None
        self._pre_verify_cache = None
        self._last_verify_ids: list[int] = []
        self._last_verify_past_seq_length = 0
        self._source_cache_indices = self._find_shared_source_cache_indices()

    def get_tokenizer(self, **kwargs):
        del kwargs
        return self.tokenizer

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _find_shared_source_cache_indices(self) -> dict[str, int]:
        layers = self.model.model.language_model.layers
        layer_types = list(self.text_config["layer_types"])
        layer_to_cache_idx: dict[int, int] = {}
        cache_idx = 0
        source: dict[str, int] = {}
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            if not bool(getattr(attn, "is_kv_shared_layer", False)):
                layer_to_cache_idx[layer_idx] = cache_idx
                cache_idx += 1
            if bool(getattr(attn, "store_full_length_kv", False)):
                layer_type = str(getattr(attn, "layer_type", None) or layer_types[layer_idx])
                source[layer_type] = layer_to_cache_idx[layer_idx]
        if "sliding_attention" not in source or "full_attention" not in source:
            # Dense 26B/31B mark the final sliding/full layers.  If a local HF
            # revision lacks the helper attrs, fall back to the last layer of
            # each type that owns a cache.
            for layer_idx, layer_type in enumerate(layer_types):
                if layer_idx in layer_to_cache_idx:
                    source[str(layer_type)] = layer_to_cache_idx[layer_idx]
        if "sliding_attention" not in source or "full_attention" not in source:
            raise RuntimeError(f"Cannot locate Gemma4 shared KV source layers: {source}")
        return source

    def _pad_cache(self, tensor, width: int):
        import torch

        out = torch.zeros(
            (tensor.shape[0], tensor.shape[1], width, tensor.shape[3]), dtype=tensor.dtype, device=tensor.device
        )
        take = min(int(tensor.shape[2]), int(width))
        if take > 0:
            out[:, :, :take, :] = tensor[:, :, -take:, :]
        return out

    def _shared_from_cache(self) -> dict[str, Any]:
        if self._cache is None:
            raise RuntimeError("FP target cache is not initialized")
        sliding_layer = self._cache.layers[self._source_cache_indices["sliding_attention"]]
        full_layer = self._cache.layers[self._source_cache_indices["full_attention"]]
        return {
            "shared_key_cache_sliding": self._pad_cache(sliding_layer.keys, self.shared_sliding),
            "shared_value_cache_sliding": self._pad_cache(sliding_layer.values, self.shared_sliding),
            "shared_key_cache_full": self._pad_cache(full_layer.keys, self.shared_full),
            "shared_value_cache_full": self._pad_cache(full_layer.values, self.shared_full),
        }

    def _full_cache_length(self) -> int:
        if self._cache is None:
            return 0
        full_layer = self._cache.layers[self._source_cache_indices["full_attention"]]
        return int(full_layer.keys.shape[2])

    def run_prefill_ids(self, input_ids):
        import torch

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        self._cache = outputs.past_key_values
        return outputs.logits, outputs.hidden_states[-1][:, -1:, :], self._shared_from_cache(), int(input_ids.shape[1])

    def run_verify_ids(self, token_ids: list[int], past_seq_length: int):
        import torch

        if self._cache is not None and self._full_cache_length() > int(past_seq_length):
            self._cache.crop(int(past_seq_length))
        self._pre_verify_cache = copy.deepcopy(self._cache)
        self._last_verify_ids = list(map(int, token_ids))
        self._last_verify_past_seq_length = int(past_seq_length)
        tokens = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=tokens,
                past_key_values=self._cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        self._cache = outputs.past_key_values
        return outputs.logits, outputs.hidden_states[-1], self._shared_from_cache()

    def commit_cache(self, valid_length: int) -> dict[str, Any]:
        valid_length = int(valid_length)
        accepted_steps = valid_length - int(self._last_verify_past_seq_length)
        if self._pre_verify_cache is not None and 0 <= accepted_steps < len(self._last_verify_ids):
            import torch

            self._cache = copy.deepcopy(self._pre_verify_cache)
            if accepted_steps > 0:
                tokens = torch.tensor(
                    [self._last_verify_ids[:accepted_steps]],
                    dtype=torch.long,
                    device=self.device,
                )
                with torch.inference_mode():
                    outputs = self.model(
                        input_ids=tokens,
                        past_key_values=self._cache,
                        use_cache=True,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                self._cache = outputs.past_key_values
            return self._shared_from_cache()
        if self._cache is not None and self._full_cache_length() > valid_length:
            self._cache.crop(int(valid_length))
        return self._shared_from_cache()


def run_verify(args: argparse.Namespace) -> None:
    names = sorted(PRESETS) if args.all else [args.preset]
    results = [
        verify_preset(
            PRESETS[name],
            Path(args.base_root),
            Path(args.draft_root),
            meta=args.meta if len(names) == 1 else None,
            draft_onnx=args.draft_onnx if len(names) == 1 else None,
        )
        for name in names
    ]
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    for result in results:
        print(
            f"{result['preset']}: OK | {result['assistant_kind']} | "
            f"base_outputs={result['base_outputs']} | draft_outputs={result['draft_outputs']} | "
            f"w4={result['w4_head_node']['name']} | quant={result['quant_counter']}"
        )


def run_generate(args: argparse.Namespace) -> None:
    import torch

    from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
    from xhquant.api import get_xhquant_logger, xhquant_init
    from xhquant.utils import MemoryTracker, TimeProfiler

    preset = PRESETS[args.preset]
    meta_path = _resolve_meta_path(preset, Path(args.base_root), args.meta)
    meta_for_draft = json.loads(meta_path.read_text(encoding="utf-8"))
    draft_onnx = None
    if args.draft_backend == "hmonnx":
        deploy_draft_onnx = _resolve_draft_onnx(
            preset, Path(args.draft_root), args.draft_onnx, meta_path=meta_path, meta_dict=meta_for_draft
        )
        verify_preset(
            preset,
            Path(args.base_root),
            Path(args.draft_root),
            meta=str(meta_path),
            draft_onnx=str(deploy_draft_onnx),
        )
        draft_onnx = _resolve_draft_onnx(
            preset,
            Path(args.draft_root),
            args.draft_onnx,
            meta_path=meta_path,
            meta_dict=meta_for_draft,
            prefer_standalone=True,
        )
    elif args.draft_onnx:
        # Keep the target/draft ONNX contract check available even when the
        # current run uses fp draft.  This catches stale manifests before a
        # mixed-backend diagnosis compares against the wrong exported graph.
        verify_preset(
            preset, Path(args.base_root), Path(args.draft_root), meta=str(meta_path), draft_onnx=args.draft_onnx
        )

    log_file = meta_path.parent / "gemma4_series_mtp_hmonnx_generate.log"
    xhquant_init(str(log_file), args.debug)
    logger = get_xhquant_logger()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["_meta_path"] = str(meta_path)

    device = torch.device(args.device)
    default_target_dir, default_assistant_dir = DEFAULT_HF_DIRS[args.preset]
    target_model_dir = args.target_model_dir or str(meta.get("model_config", {}).get("hf_model") or default_target_dir)
    assistant_model_dir = args.assistant_model_dir or default_assistant_dir
    if args.target_backend == "hmonnx":
        target_model = AutoLLMHONNXModel.from_pretrained(
            str(meta_path),
            enable_cuda_graph=args.enable_cuda_graph,
            enable_prefill_cuda_graph=False if args.disable_prefill_cuda_graph else None,
            enable_decode_cuda_graph=False if args.disable_decode_cuda_graph else None,
            enable_auto_offload=args.enable_auto_offload,
            device_map=args.device_map,
        )
        if args.enable_auto_offload and hasattr(target_model, "enable_auto_offload"):
            target_model.enable_auto_offload = True
        target_model.to(device)
        tokenizer = target_model.get_tokenizer(trust_remote_code=True)
    else:
        target_model = TorchTargetSession(
            target_model_dir=target_model_dir,
            meta=meta,
            device=device,
            dtype=_torch_dtype(args.target_dtype),
        )
        tokenizer = target_model.get_tokenizer(trust_remote_code=True)
    if args.draft_backend == "hmonnx":
        if draft_onnx is None:
            draft_onnx = _resolve_draft_onnx(
                preset,
                Path(args.draft_root),
                args.draft_onnx,
                meta_path=meta_path,
                meta_dict=meta_for_draft,
                prefer_standalone=True,
            )
        assistant_session = AssistantDraftSession(draft_onnx, device)
        draft_desc = str(draft_onnx)
    else:
        assistant_session = TorchAssistantDraftSession(
            assistant_model_dir=assistant_model_dir,
            target_model_dir=target_model_dir,
            meta=meta,
            device=device,
            dtype=_torch_dtype(args.draft_dtype),
        )
        draft_desc = f"torch:{assistant_model_dir}"

    logger.info(
        "Gemma4 Series MTP generate preset=%s meta=%s target_backend=%s draft_backend=%s draft=%s",
        args.preset,
        meta_path,
        args.target_backend,
        args.draft_backend,
        draft_desc,
    )
    trace_kv_input_info = _first_sliding_kv_trace_info(meta_path, meta) if args.trace_accepted_count else None
    if args.trace_accepted_count:
        if trace_kv_input_info is None:
            raise RuntimeError(f"{args.preset}: cannot find a sliding KVcache node in target decode ONNX")
        print("[MTP accepted_count trace] first sliding KVcache node")
        print(json.dumps(trace_kv_input_info, ensure_ascii=False, indent=2))
    with (
        TimeProfiler("gemma4_series_mtp_hmonnx_generate", logger),
        MemoryTracker(device=str(device), name="generate", logger=logger),
    ):
        with ExitStack() as stack:
            if args.target_backend == "hmonnx":
                stack.enter_context(LLMInferenceContextManager(target_model, [device]))
            with torch.inference_mode():
                text, output_ids, stats = generate_with_mtp(
                    target_model=target_model,
                    assistant_session=assistant_session,
                    tokenizer=tokenizer,
                    meta=meta,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    num_draft_tokens=args.num_draft_tokens,
                    trace_accepted_count=args.trace_accepted_count,
                    trace_kv_input_info=trace_kv_input_info,
                )

    print(text)
    print("\n[Gemma4 MTP HMONNX stats]")
    payload = {
        "preset": args.preset,
        "meta": str(meta_path),
        "target_backend": args.target_backend,
        "draft_backend": args.draft_backend,
        "draft_onnx": str(draft_onnx) if draft_onnx is not None else None,
        "draft": draft_desc,
        **stats,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info("[MTP stats] %s", json.dumps(payload, ensure_ascii=False))


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-root",
        default="work_dirs/gemma4_series_mtp_base_clean",
        help="Root containing <slug>/existing-hf/export/<export-dir> target HMONNX exports.",
    )
    parser.add_argument(
        "--draft-root",
        default="work_dirs/gemma4_series_mtp_clean",
        help="Legacy root containing mtp_draft_decode/draft_onnx; ignored when manifest records draft_decode_onnx.",
    )
    parser.add_argument("--meta", help="Explicit target golden_meta_info.json or export directory for one preset.")
    parser.add_argument("--draft-onnx", help="Explicit assistant decode ONNX path for one preset.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backward compatible contract smoke: ``mtp_hmonnx_inference.py --all``.
    if not argv or argv[0] not in {"verify", "generate"}:
        argv = ["verify", *argv]

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify", help="Verify exported Gemma4 Series MTP target/draft graph contract.")
    p_verify.add_argument("--preset", choices=sorted(PRESETS), default="e2b")
    p_verify.add_argument("--all", action="store_true", help="Verify all Gemma4 Series MTP presets.")
    _add_common_paths(p_verify)
    p_verify.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    p_verify.set_defaults(func=run_verify)

    p_gen = sub.add_parser("generate", help="Run greedy HMONNX target + ONNX assistant MTP decoding.")
    p_gen.add_argument("--preset", choices=sorted(PRESETS), default="e2b")
    _add_common_paths(p_gen)
    p_gen.add_argument("--prompt", default="用一句话解释 Gemma4 MTP。")
    p_gen.add_argument("--max-new-tokens", type=int, default=32)
    p_gen.add_argument("--num-draft-tokens", type=int, default=4)
    p_gen.add_argument("--device", default="cuda:0")
    p_gen.add_argument(
        "--device-map",
        nargs="+",
        default=None,
        help="Device map passed to target HMONNX runtime; e.g. --device-map cuda:0 cpu for auto offload.",
    )
    p_gen.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help="Enable target HMONNX CUDA graph execution when supported by the selected runtime.",
    )
    p_gen.add_argument(
        "--disable-prefill-cuda-graph",
        action="store_true",
        help="Keep target prefill out of CUDA graph while preserving --enable-cuda-graph for decode.",
    )
    p_gen.add_argument(
        "--disable-decode-cuda-graph",
        action="store_true",
        help="Keep target decode out of CUDA graph while preserving --enable-cuda-graph for prefill.",
    )
    p_gen.add_argument(
        "--enable-auto-offload",
        action="store_true",
        help="Enable target HMONNX auto offload; intended for HMONNXInferenceV2 large-model runs.",
    )
    p_gen.add_argument("--target-backend", choices=["hmonnx", "torch"], default="hmonnx")
    p_gen.add_argument("--target-dtype", default="float16", help="dtype for --target-backend torch")
    p_gen.add_argument("--draft-backend", choices=["hmonnx", "torch"], default="hmonnx")
    p_gen.add_argument("--draft-dtype", default="float16", help="dtype for --draft-backend torch")
    p_gen.add_argument("--target-model-dir", help="HF target dir for --draft-backend torch embedding/config.")
    p_gen.add_argument("--assistant-model-dir", help="HF assistant dir for --draft-backend torch.")
    p_gen.add_argument(
        "--trace-accepted-count",
        action="store_true",
        help="Print first sliding KVcache accepted_count inputs and per-verify accepted-token alignment.",
    )
    p_gen.add_argument("--debug", action="store_true")
    p_gen.set_defaults(func=run_generate)

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
