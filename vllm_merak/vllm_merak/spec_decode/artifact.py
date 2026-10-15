"""Parse and validate HMONNX speculative-decoding artifact metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from ..constants import (
    MERAK_MAX_NARROW_QUERY_LENGTH,
    MERAK_MIN_VERIFY_QUERY_LENGTH,
)


SpecDecodeMode = Literal["mtp", "dflash"]

_QWEN_MTP_ABI = "qwen_mtp_v1"
_QWEN_MTP_PAGED_ABI = "qwen_mtp_paged_v2"
_QWEN_DFLASH_ABI = "qwen_dflash_v1"
_QWEN_DFLASH_PAGED_ABI = "qwen_dflash_paged_v2"
_QWEN_DFLASH_PAGED_SHARED_ABI = "qwen_dflash_paged_shared_v3"
_QWEN_DFLASH2_ABI = "qwen_dflash2_v1"
_QWEN_DFLASH2_PAGED_SHARED_ABI = "qwen_dflash2_paged_shared_v1"
_QWEN_DFLASH2_HYBRID_ABI = "qwen_dflash2_hybrid_v1"
_GEMMA4_DENSE_MTP_ABI = "gemma4_mtp_dense_shared_kv_v1"
_GEMMA4_PAGED_MTP_ABI = "gemma4_mtp_readonly_page_attention_v1"
_GEMMA4_PAGED_CAUSAL_MTP_ABI = "gemma4_mtp_readonly_page_attention_v2"
_GEMMA4_PAGED_MIXED_MTP_ABI = "gemma4_mtp_readonly_mixed_attention_v3"
_GEMMA4_PAGED_MTP_ABIS = {
    _GEMMA4_PAGED_MTP_ABI,
    _GEMMA4_PAGED_CAUSAL_MTP_ABI,
    _GEMMA4_PAGED_MIXED_MTP_ABI,
}


@dataclass(frozen=True)
class SpecDecodeDraftGraphs:
    """Resolved HMONNX draft graph paths."""

    prefill: Path | None = None
    context: Path | None = None
    context_decode: Path | None = None
    decode: Path | None = None


@dataclass(frozen=True)
class HMONNXSpecDecodeArtifact:
    """Validated deployment contract extracted from golden metadata."""

    artifact_dir: Path
    metadata_path: Path
    runtime_contract_version: int
    mode: SpecDecodeMode
    num_draft_tokens: int
    verify_length: int
    target_hidden_output_name: str
    target_linear_state_outputs: str | None
    draft_abi: str
    draft_graphs: SpecDecodeDraftGraphs
    draft_cache_mutation: str | None
    draft_cache_binding: str | None
    draft_constant_positions: bool
    draft_position_semantics: str | None
    draft_layer_attention_types: tuple[str, ...]
    minimum_sliding_cache_slack: int | None
    dflash_noise_token_id: int | None
    draft_page_cache_block_size: int | None = None
    dflash_selector_top_k: int | None = None
    dflash_selector_rank: int | None = None
    dflash_sliding_window: int | None = None
    dflash_selector_scheme: str | None = None

    @property
    def is_gemma4(self) -> bool:
        return self.draft_abi.startswith("gemma4_")

    @property
    def is_qwen_mtp(self) -> bool:
        return self.draft_abi in {_QWEN_MTP_ABI, _QWEN_MTP_PAGED_ABI}

    @property
    def is_qwen_dflash(self) -> bool:
        return self.draft_abi in {
            _QWEN_DFLASH2_HYBRID_ABI,
            _QWEN_DFLASH_ABI,
            _QWEN_DFLASH_PAGED_ABI,
            _QWEN_DFLASH_PAGED_SHARED_ABI,
            _QWEN_DFLASH2_ABI,
            _QWEN_DFLASH2_PAGED_SHARED_ABI,
        }

    @property
    def is_qwen_dflash2(self) -> bool:
        return self.draft_abi in {
            _QWEN_DFLASH2_HYBRID_ABI,
            _QWEN_DFLASH2_ABI,
            _QWEN_DFLASH2_PAGED_SHARED_ABI,
        }

    @property
    def uses_hybrid_dflash_cache(self) -> bool:
        return self.draft_abi == _QWEN_DFLASH2_HYBRID_ABI

    @property
    def is_qwen_hybrid_spec_decode(self) -> bool:
        return self.is_qwen_mtp or self.is_qwen_dflash

    @property
    def uses_readonly_target_page_cache(self) -> bool:
        return self.draft_abi in _GEMMA4_PAGED_MTP_ABIS

    @property
    def uses_causal_readonly_target_page_cache(self) -> bool:
        return self.draft_abi == _GEMMA4_PAGED_CAUSAL_MTP_ABI

    @property
    def uses_mixed_readonly_target_page_cache(self) -> bool:
        return self.draft_abi == _GEMMA4_PAGED_MIXED_MTP_ABI

    @property
    def uses_private_dflash_page_cache(self) -> bool:
        return self.draft_abi in {
            _QWEN_DFLASH_PAGED_ABI,
            _QWEN_DFLASH_PAGED_SHARED_ABI,
            _QWEN_DFLASH2_PAGED_SHARED_ABI,
        }

    @property
    def uses_graph_dflash_page_cache_writes(self) -> bool:
        return self.draft_abi in {
            _QWEN_DFLASH_PAGED_SHARED_ABI,
            _QWEN_DFLASH2_PAGED_SHARED_ABI,
        }

    @property
    def uses_private_mtp_page_cache(self) -> bool:
        return self.draft_abi == _QWEN_MTP_PAGED_ABI


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"HMONNX metadata must be a JSON object: {path}")
    return value


def _metadata_path(model_dir: Path) -> Path:
    merak_config_path = model_dir / "merak_config.json"
    if merak_config_path.is_file():
        merak_config = _read_json(merak_config_path)
        xh_model = merak_config.get("xh_model")
        if not isinstance(xh_model, Mapping):
            raise ValueError(
                f"Merak config xh_model must be an object: {merak_config_path}"
            )
        raw_meta_path = xh_model.get("meta_info") or "golden_meta_info.json"
        if not isinstance(raw_meta_path, str) or not raw_meta_path.strip():
            raise ValueError(
                f"Merak config xh_model.meta_info must be a path: {merak_config_path}"
            )
        meta_path = Path(raw_meta_path)
        if not meta_path.is_absolute():
            meta_path = model_dir / meta_path
    else:
        meta_path = model_dir / "golden_meta_info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"HMONNX metadata does not exist: {meta_path}")
    return meta_path.resolve()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be positive, got {parsed}")
    return parsed


def _non_negative_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a non-negative integer, got {value!r}"
        ) from exc
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative, got {parsed}")
    return parsed


def _first(*values: object) -> object | None:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _normalize_mode(meta: Mapping[str, Any], spec: Mapping[str, Any]) -> SpecDecodeMode:
    model_config = _mapping(meta.get("model_config"))
    raw_mode = _first(
        spec.get("mode"),
        meta.get("spec_decode_mode"),
        model_config.get("spec_decode_mode"),
    )
    mode = str(raw_mode or "").strip().lower()
    if mode == "dflash2":
        if _mapping(spec.get("draft")).get("abi") != _QWEN_DFLASH2_HYBRID_ABI:
            raise ValueError("DFlash2 mode requires the hybrid draft cache ABI")
        return "dflash"
    if mode not in {"mtp", "dflash"}:
        raise ValueError(
            f"HMONNX spec_decode.mode must be 'mtp' or 'dflash', got {raw_mode!r}"
        )
    return mode  # type: ignore[return-value]


def _is_gemma4_metadata(meta: Mapping[str, Any]) -> bool:
    model_config = _mapping(meta.get("model_config"))
    candidates = (
        meta.get("model_type"),
        model_config.get("model_type"),
        model_config.get("source_model_type"),
    )
    return any("gemma4" in str(value or "").lower() for value in candidates)


def _resolve_graph_path(
    artifact_dir: Path,
    raw_path: object,
    *,
    field: str,
    contract_version: int,
) -> Path | None:
    if raw_path in (None, ""):
        return None
    if not isinstance(raw_path, str):
        raise ValueError(f"{field} must be a string path, got {raw_path!r}")
    path = Path(raw_path)
    if contract_version >= 2 and path.is_absolute():
        raise ValueError(f"{field} must be relative for contract v2: {raw_path}")
    if not path.is_absolute():
        path = artifact_dir / path
    path = path.resolve()
    if contract_version >= 2 and not path.is_relative_to(artifact_dir):
        raise ValueError(f"{field} escapes the HMONNX artifact directory: {raw_path}")
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def _draft_graph_values(
    spec: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, object]]:
    draft = _mapping(spec.get("draft"))
    values = {
        "prefill": _first(
            draft.get("prefill_hmonnx"),
            spec.get("draft_prefill_hmonnx"),
            spec.get("draft_prefill_onnx"),
            spec.get("mtp_draft_prefill_onnx"),
        ),
        "context": _first(
            draft.get("context_hmonnx"),
            spec.get("draft_context_hmonnx"),
            spec.get("draft_context_onnx"),
            spec.get("dflash_draft_context_onnx"),
        ),
        "context_decode": _first(
            draft.get("context_decode_hmonnx"),
            spec.get("draft_context_decode_hmonnx"),
            spec.get("draft_context_decode_onnx"),
            spec.get("dflash_draft_context_decode_onnx"),
        ),
        "decode": _first(
            draft.get("decode_hmonnx"),
            spec.get("draft_decode_hmonnx"),
            spec.get("draft_decode_onnx"),
            spec.get("draft_onnx"),
            spec.get("mtp_draft_decode_onnx"),
            spec.get("dflash_draft_decode_onnx"),
        ),
    }
    return draft, values


def _normalize_draft_abi(
    *,
    meta: Mapping[str, Any],
    mode: SpecDecodeMode,
    draft: Mapping[str, Any],
) -> str:
    explicit = draft.get("abi")
    if explicit not in (None, ""):
        if not isinstance(explicit, str):
            raise ValueError(f"spec_decode.draft.abi must be a string: {explicit!r}")
        return explicit
    if _is_gemma4_metadata(meta):
        return _GEMMA4_DENSE_MTP_ABI
    return _QWEN_DFLASH_ABI if mode == "dflash" else _QWEN_MTP_ABI


def _validate_graph_set(
    *,
    mode: SpecDecodeMode,
    draft_abi: str,
    graphs: SpecDecodeDraftGraphs,
    contract_version: int,
) -> None:
    if graphs.decode is None:
        raise ValueError("HMONNX speculative decoding requires a draft decode graph")
    if mode == "dflash":
        if draft_abi not in {
            _QWEN_DFLASH2_HYBRID_ABI,
            _QWEN_DFLASH_ABI,
            _QWEN_DFLASH_PAGED_ABI,
            _QWEN_DFLASH_PAGED_SHARED_ABI,
            _QWEN_DFLASH2_ABI,
            _QWEN_DFLASH2_PAGED_SHARED_ABI,
        }:
            raise ValueError(
                "HMONNX DFlash mode requires draft ABI "
                f"{_QWEN_DFLASH_ABI!r}, {_QWEN_DFLASH_PAGED_ABI!r}, or "
                f"{_QWEN_DFLASH_PAGED_SHARED_ABI!r}, {_QWEN_DFLASH2_ABI!r}, "
                f"or {_QWEN_DFLASH2_PAGED_SHARED_ABI!r}, "
                f"got {draft_abi!r}"
            )
        if graphs.context is None:
            raise ValueError("Qwen DFlash requires a draft context graph")
        if contract_version >= 2 and graphs.context_decode is None:
            raise ValueError(
                "Qwen DFlash contract v2 requires a draft context_decode graph"
            )
        return
    if draft_abi in {
        _QWEN_DFLASH2_HYBRID_ABI,
        _QWEN_DFLASH_ABI,
        _QWEN_DFLASH_PAGED_ABI,
        _QWEN_DFLASH_PAGED_SHARED_ABI,
        _QWEN_DFLASH2_ABI,
        _QWEN_DFLASH2_PAGED_SHARED_ABI,
    }:
        raise ValueError(f"HMONNX draft ABI {draft_abi!r} requires mode='dflash'")
    if draft_abi in {_QWEN_MTP_ABI, _QWEN_MTP_PAGED_ABI} and (graphs.prefill is None):
        raise ValueError("Qwen MTP requires a draft prefill graph")
    if draft_abi not in {
        _QWEN_MTP_ABI,
        _QWEN_MTP_PAGED_ABI,
        _GEMMA4_DENSE_MTP_ABI,
        _GEMMA4_PAGED_MTP_ABI,
        _GEMMA4_PAGED_CAUSAL_MTP_ABI,
        _GEMMA4_PAGED_MIXED_MTP_ABI,
    }:
        raise ValueError(f"Unsupported HMONNX MTP draft ABI: {draft_abi!r}")


def load_hmonnx_spec_decode_artifact(
    model: str | Path,
    *,
    required: bool = False,
) -> HMONNXSpecDecodeArtifact | None:
    """Load the speculative contract for a Merak model directory.

    Returns ``None`` when the metadata has no ``spec_decode`` section unless
    ``required`` is true. Referenced graph paths are validated eagerly.
    """

    model_dir = Path(model).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Merak model directory does not exist: {model_dir}")
    metadata_path = _metadata_path(model_dir)
    meta = _read_json(metadata_path)
    raw_spec = meta.get("spec_decode")
    model_config = _mapping(meta.get("model_config"))
    has_legacy_mode = _first(
        meta.get("spec_decode_mode"),
        model_config.get("spec_decode_mode"),
    )
    if not isinstance(raw_spec, Mapping) and has_legacy_mode in (None, ""):
        if required:
            raise ValueError(
                f"HMONNX metadata has no speculative-decoding contract: {metadata_path}"
            )
        return None
    spec = _mapping(raw_spec)

    mode = _normalize_mode(meta, spec)
    contract_version = _positive_int(
        spec.get("runtime_contract_version") or 1,
        field="spec_decode.runtime_contract_version",
    )
    assert contract_version is not None
    if contract_version not in (1, 2):
        raise ValueError(
            "Unsupported HMONNX speculative runtime contract version: "
            f"{contract_version}"
        )

    num_draft_tokens = _positive_int(
        _first(
            spec.get("num_draft_tokens"),
            model_config.get("num_draft_tokens"),
        ),
        field="spec_decode.num_draft_tokens",
    )
    verify_length = _positive_int(
        _first(
            spec.get("verify_length"),
            meta.get("spec_decode_verify_length"),
            model_config.get("spec_decode_verify_length"),
            model_config.get("verify_length"),
        ),
        field="spec_decode.verify_length",
    )
    if num_draft_tokens is None and verify_length is not None:
        num_draft_tokens = verify_length - 1
    if num_draft_tokens is None and mode == "mtp":
        num_draft_tokens = _positive_int(
            _first(spec.get("block_size"), meta.get("spec_decode_block_size")),
            field="spec_decode.block_size",
        )
    if num_draft_tokens is None:
        raise ValueError(
            "HMONNX speculative metadata must declare num_draft_tokens or verify_length"
        )
    if verify_length is None:
        verify_length = num_draft_tokens + 1
    if verify_length != num_draft_tokens + 1:
        raise ValueError(
            "HMONNX target verify length must equal num_draft_tokens + 1: "
            f"verify_length={verify_length}, drafts={num_draft_tokens}"
        )
    if not (
        MERAK_MIN_VERIFY_QUERY_LENGTH <= verify_length <= MERAK_MAX_NARROW_QUERY_LENGTH
    ):
        raise ValueError(
            "HMONNX target verify length must stay within the packed-W4 "
            "narrow-token contract "
            f"[{MERAK_MIN_VERIFY_QUERY_LENGTH}, "
            f"{MERAK_MAX_NARROW_QUERY_LENGTH}], got {verify_length}"
        )

    target = _mapping(spec.get("target"))
    hidden_output = _first(
        target.get("hidden_output_name"),
        spec.get("hidden_output_name"),
        meta.get("spec_decode_hidden_output_name"),
    )
    if hidden_output in (None, ""):
        hidden_output = "target_hidden" if mode == "dflash" else "post_norm_hidden"
    if not isinstance(hidden_output, str):
        raise ValueError(
            f"spec_decode target hidden output name must be a string: {hidden_output!r}"
        )

    draft, graph_values = _draft_graph_values(spec)
    artifact_dir = metadata_path.parent.resolve()
    graphs = SpecDecodeDraftGraphs(
        **{
            name: _resolve_graph_path(
                artifact_dir,
                raw_path,
                field=f"spec_decode.draft.{name}_hmonnx",
                contract_version=contract_version,
            )
            for name, raw_path in graph_values.items()
        }
    )
    draft_abi = _normalize_draft_abi(meta=meta, mode=mode, draft=draft)
    _validate_graph_set(
        mode=mode,
        draft_abi=draft_abi,
        graphs=graphs,
        contract_version=contract_version,
    )

    linear_state_outputs = target.get("linear_state_outputs")
    if linear_state_outputs is not None and not isinstance(linear_state_outputs, str):
        raise ValueError("spec_decode.target.linear_state_outputs must be a string")
    cache_mutation = draft.get("cache_mutation")
    cache_binding = draft.get("cache_binding")
    for field_name, value in (
        ("cache_mutation", cache_mutation),
        ("cache_binding", cache_binding),
    ):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"spec_decode.draft.{field_name} must be a string")
    constant_positions = draft.get("constant_draft_positions", False)
    if not isinstance(constant_positions, bool):
        raise ValueError("spec_decode.draft.constant_draft_positions must be a boolean")
    # Older Gemma4 v2 artifacts used ``position_semantics`` for the synthetic
    # attention past length, even though the assistant query's RoPE position
    # is one token later.  Parse that spelling for compatibility, but normalize
    # the public artifact field to the actual query-RoPE semantics.
    legacy_position_semantics = draft.get("position_semantics")
    query_rope_position_semantics = draft.get("query_rope_position_semantics")
    attention_past_length_semantics = draft.get("attention_past_length_semantics")
    for field_name, value in (
        ("position_semantics", legacy_position_semantics),
        ("query_rope_position_semantics", query_rope_position_semantics),
        ("attention_past_length_semantics", attention_past_length_semantics),
    ):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"spec_decode.draft.{field_name} must be a string")
    legacy_gemma_length_spelling = (
        draft_abi in _GEMMA4_PAGED_MTP_ABIS
        and legacy_position_semantics == "target_kv_valid_length_minus_one"
    )
    if query_rope_position_semantics is None and legacy_gemma_length_spelling:
        query_rope_position_semantics = "target_kv_valid_length"
    if attention_past_length_semantics is None and legacy_gemma_length_spelling:
        attention_past_length_semantics = legacy_position_semantics
    position_semantics = query_rope_position_semantics or legacy_position_semantics
    raw_layer_attention_types = draft.get("layer_attention_types") or ()
    if not isinstance(raw_layer_attention_types, (list, tuple)) or any(
        not isinstance(value, str) for value in raw_layer_attention_types
    ):
        raise ValueError(
            "spec_decode.draft.layer_attention_types must be a string list"
        )
    layer_attention_types = tuple(raw_layer_attention_types)
    invalid_attention_types = set(layer_attention_types) - {
        "sliding_attention",
        "full_attention",
    }
    if invalid_attention_types:
        raise ValueError(
            "spec_decode.draft.layer_attention_types contains unsupported "
            f"values: {sorted(invalid_attention_types)}"
        )
    minimum_sliding_cache_slack = _positive_int(
        draft.get("minimum_sliding_cache_slack"),
        field="spec_decode.draft.minimum_sliding_cache_slack",
    )
    page_cache_block_size = _positive_int(
        draft.get("page_cache_block_size"),
        field="spec_decode.draft.page_cache_block_size",
    )
    dflash_config = _mapping(model_config.get("dflash_config"))
    dflash_noise_token_id = _non_negative_int(
        _first(
            draft.get("noise_token_id"),
            spec.get("dflash_noise_token_id"),
            dflash_config.get("noise_token_id"),
            dflash_config.get("mask_token_id"),
        ),
        field="spec_decode.draft.noise_token_id",
    )
    dflash_selector_top_k = _positive_int(
        draft.get("selector_top_k"),
        field="spec_decode.draft.selector_top_k",
    )
    dflash_selector_rank = _positive_int(
        draft.get("selector_rank"),
        field="spec_decode.draft.selector_rank",
    )
    dflash_sliding_window = _positive_int(
        draft.get("sliding_window"),
        field="spec_decode.draft.sliding_window",
    )
    selector_scheme = draft.get("selector_scheme")
    if selector_scheme is not None and not isinstance(selector_scheme, str):
        raise ValueError("spec_decode.draft.selector_scheme must be a string")

    if draft_abi in {_QWEN_DFLASH2_ABI, _QWEN_DFLASH2_PAGED_SHARED_ABI, _QWEN_DFLASH2_HYBRID_ABI}:
        required_dflash2_fields = {
            "selector_top_k": dflash_selector_top_k,
            "selector_rank": dflash_selector_rank,
            "sliding_window": dflash_sliding_window,
        }
        missing = [
            name for name, value in required_dflash2_fields.items() if value is None
        ]
        if missing:
            raise ValueError(
                "Qwen DFlash2 metadata is missing required draft fields: "
                f"{', '.join(missing)}"
            )
        if selector_scheme != "precomputed_kxk_v1":
            raise ValueError(
                "Qwen DFlash2 requires spec_decode.draft.selector_scheme="
                "'precomputed_kxk_v1'"
            )
        if draft.get("is_causal") is not False:
            raise ValueError("Qwen DFlash2 requires spec_decode.draft.is_causal=false")
        block_size = _positive_int(
            draft.get("block_size"),
            field="spec_decode.draft.block_size",
        )
        if block_size != verify_length:
            raise ValueError(
                "Qwen DFlash2 block_size must equal verify_length: "
                f"block_size={block_size}, verify_length={verify_length}"
            )
        decode_outputs = draft.get("decode_outputs")
        expected_outputs = {
            "candidate_ids",
            "selector_first_scores",
            "selector_transition_scores",
        }
        if (
            not isinstance(decode_outputs, (list, tuple))
            or set(decode_outputs) != expected_outputs
        ):
            raise ValueError(
                "Qwen DFlash2 draft.decode_outputs must declare candidate_ids, "
                "selector_first_scores, and selector_transition_scores"
            )

    if contract_version >= 2:
        if draft_abi in {
            _QWEN_DFLASH2_HYBRID_ABI,
            _QWEN_MTP_ABI,
            _QWEN_DFLASH_ABI,
            _QWEN_DFLASH2_ABI,
        } and (cache_mutation != "in_place"):
            raise ValueError(
                "Qwen speculative contract v2 requires "
                "spec_decode.draft.cache_mutation='in_place'"
            )
        if draft_abi in {
            _QWEN_DFLASH2_HYBRID_ABI,
            _QWEN_MTP_ABI,
            _QWEN_MTP_PAGED_ABI,
            _QWEN_DFLASH_ABI,
            _QWEN_DFLASH_PAGED_ABI,
            _QWEN_DFLASH_PAGED_SHARED_ABI,
            _QWEN_DFLASH2_ABI,
            _QWEN_DFLASH2_PAGED_SHARED_ABI,
        } and (linear_state_outputs != "per_step"):
            raise ValueError(
                "Qwen speculative contract v2 requires "
                "spec_decode.target.linear_state_outputs='per_step'"
            )
        if draft_abi in {
            _QWEN_DFLASH_PAGED_ABI,
            _QWEN_DFLASH_PAGED_SHARED_ABI,
            _QWEN_DFLASH2_PAGED_SHARED_ABI,
        }:
            if cache_mutation != "page_attention":
                raise ValueError(
                    "Paged Qwen DFlash requires "
                    "spec_decode.draft.cache_mutation='page_attention'"
                )
            if cache_binding != "private_draft":
                raise ValueError(
                    "Paged Qwen DFlash requires "
                    "spec_decode.draft.cache_binding='private_draft'"
                )
            if page_cache_block_size is None or page_cache_block_size % 64:
                raise ValueError(
                    "Paged Qwen DFlash page_cache_block_size must be a "
                    "positive multiple of 64"
                )
        if draft_abi == _QWEN_MTP_PAGED_ABI:
            if cache_mutation != "page_attention":
                raise ValueError(
                    "Paged Qwen MTP requires "
                    "spec_decode.draft.cache_mutation='page_attention'"
                )
            if cache_binding != "private_draft":
                raise ValueError(
                    "Paged Qwen MTP requires "
                    "spec_decode.draft.cache_binding='private_draft'"
                )
            if page_cache_block_size is None or page_cache_block_size % 64:
                raise ValueError(
                    "Paged Qwen MTP page_cache_block_size must be a positive "
                    "multiple of 64"
                )
        if mode == "dflash" and dflash_noise_token_id is None:
            raise ValueError(
                "Qwen DFlash contract v2 requires spec_decode.draft.noise_token_id"
            )
        if draft_abi == _GEMMA4_DENSE_MTP_ABI:
            raise ValueError(
                "Gemma4 contract v2 requires the read-only PageAttention ABI; "
                "dense shared-KV MTP is legacy-only"
            )
        if draft_abi in _GEMMA4_PAGED_MTP_ABIS and (
            cache_binding != "target_attention_type_owner"
        ):
            raise ValueError(
                "Gemma4 paged MTP requires "
                "spec_decode.draft.cache_binding="
                "'target_attention_type_owner'"
            )
        if draft_abi in _GEMMA4_PAGED_MTP_ABIS:
            if cache_mutation != "read_only":
                raise ValueError(
                    "Gemma4 paged MTP requires "
                    "spec_decode.draft.cache_mutation='read_only'"
                )
            if not constant_positions:
                raise ValueError(
                    "Gemma4 paged MTP requires constant_draft_positions=true"
                )
            if position_semantics != "target_kv_valid_length":
                raise ValueError(
                    "Gemma4 paged MTP requires "
                    "query_rope_position_semantics="
                    "'target_kv_valid_length'"
                )
            if attention_past_length_semantics != "target_kv_valid_length_minus_one":
                raise ValueError(
                    "Gemma4 paged MTP requires "
                    "attention_past_length_semantics="
                    "'target_kv_valid_length_minus_one'"
                )
            if set(layer_attention_types) != {
                "sliding_attention",
                "full_attention",
            }:
                raise ValueError(
                    "Gemma4 paged MTP draft must declare both sliding and "
                    "full layer_attention_types"
                )
            if (
                minimum_sliding_cache_slack is None
                or minimum_sliding_cache_slack < num_draft_tokens
            ):
                raise ValueError(
                    "Gemma4 paged MTP minimum_sliding_cache_slack must be at "
                    f"least num_draft_tokens={num_draft_tokens}"
                )
            sliding_window = _positive_int(
                _first(
                    meta.get("sliding_window"),
                    model_config.get("sliding_window"),
                ),
                field="Gemma4 sliding_window",
            )
            sliding_cache_capacity = _positive_int(
                _first(
                    spec.get("shared_sliding_cache_length"),
                    meta.get("sliding_cache_storage_length"),
                ),
                field="Gemma4 shared_sliding_cache_length",
            )
            if sliding_window is None or sliding_cache_capacity is None:
                raise ValueError(
                    "Gemma4 paged MTP must declare sliding_window and "
                    "shared_sliding_cache_length"
                )
            required_capacity = sliding_window + num_draft_tokens
            if sliding_cache_capacity < required_capacity:
                raise ValueError(
                    "Gemma4 paged MTP sliding cache has insufficient rollback "
                    f"slack: capacity={sliding_cache_capacity}, "
                    f"required>={required_capacity}"
                )
            attention_lowering = draft.get("attention_lowering")
            if (
                draft_abi == _GEMMA4_PAGED_CAUSAL_MTP_ABI
                and attention_lowering != "causal"
            ):
                raise ValueError(
                    "Gemma4 read-only PageAttention v2 requires "
                    "spec_decode.draft.attention_lowering='causal'"
                )
            if (
                draft_abi == _GEMMA4_PAGED_MIXED_MTP_ABI
                and attention_lowering != "full_page_sliding_unfused"
            ):
                raise ValueError(
                    "Gemma4 read-only mixed attention v3 requires "
                    "spec_decode.draft.attention_lowering="
                    "'full_page_sliding_unfused'"
                )

    return HMONNXSpecDecodeArtifact(
        artifact_dir=artifact_dir,
        metadata_path=metadata_path,
        runtime_contract_version=contract_version,
        mode=mode,
        num_draft_tokens=num_draft_tokens,
        verify_length=verify_length,
        target_hidden_output_name=hidden_output,
        target_linear_state_outputs=linear_state_outputs,
        draft_abi=draft_abi,
        draft_graphs=graphs,
        draft_cache_mutation=cache_mutation,
        draft_cache_binding=cache_binding,
        draft_constant_positions=constant_positions,
        draft_position_semantics=position_semantics,
        draft_layer_attention_types=layer_attention_types,
        minimum_sliding_cache_slack=minimum_sliding_cache_slack,
        dflash_noise_token_id=(
            dflash_noise_token_id
            if dflash_noise_token_id is not None
            else 248070
            if mode == "dflash"
            else None
        ),
        draft_page_cache_block_size=page_cache_block_size,
        dflash_selector_top_k=dflash_selector_top_k,
        dflash_selector_rank=dflash_selector_rank,
        dflash_sliding_window=dflash_sliding_window,
        dflash_selector_scheme=selector_scheme,
    )


__all__ = [
    "HMONNXSpecDecodeArtifact",
    "SpecDecodeDraftGraphs",
    "load_hmonnx_spec_decode_artifact",
]
