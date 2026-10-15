"""Qwen fused recurrent-cache ABI used by Merak HMONNX runtimes.

``GDRChunkScan`` can treat its recurrent-state input as a CacheTensor: target
prefill mutates that input in place and intentionally exports no recurrent
state output.  Decode and speculative verify graphs remain functional and
must export one snapshot per layer and verify position.  Keeping this contract
in the plugin makes the unusual state ownership explicit and lets malformed
artifacts fail before serving requests.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


_RECURRENT_INPUT_RE = re.compile(r"^past_recurrent_state_(\d+)$")
_RECURRENT_OUTPUT_RE = re.compile(r"^recurrent_state_out_(\d+)(?:_(\d+))?$")


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean, got {value!r}")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return value


def _model_io_names(model: object, *, kind: str, phase: str) -> tuple[str, ...]:
    session = getattr(model, "hmonnx_session", None)
    method_name = f"get_{kind}_names"
    attribute_name = f"{kind}_names"
    owners = (session, model)
    for owner in owners:
        if owner is None:
            continue
        get_names = getattr(owner, method_name, None)
        if callable(get_names):
            # HMONNXGoldenInference exposes these methods before its lazy
            # execution session exists.  Calling them during model loading
            # dereferences ``_session=None``; defer to the graph file below.
            if getattr(owner, "_session", object()) is not None:
                return tuple(str(name) for name in get_names())
        names = getattr(owner, attribute_name, None)
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            return tuple(str(name) for name in names)

    for owner in owners:
        onnx_file = getattr(owner, "_onnx_file", None)
        if isinstance(onnx_file, str) and onnx_file:
            import onnx

            graph = onnx.load(onnx_file, load_external_data=False).graph
            values = graph.input if kind == "input" else graph.output
            return tuple(value.name for value in values)
    raise RuntimeError(
        "Qwen fused recurrent-cache contract cannot inspect "
        f"{phase} HMONNX {kind} names"
    )


def _recurrent_input_indices(
    names: Sequence[str],
    *,
    phase: str,
) -> set[int]:
    indices: set[int] = set()
    malformed: list[str] = []
    for name in names:
        match = _RECURRENT_INPUT_RE.fullmatch(name)
        if match:
            indices.add(int(match.group(1)))
        elif "recurrent_state" in name:
            malformed.append(name)
    if malformed:
        raise RuntimeError(
            f"Qwen {phase} graph has malformed recurrent-state inputs: {malformed}"
        )
    return indices


def _recurrent_output_indices(
    names: Sequence[str],
    *,
    phase: str,
) -> set[tuple[int, int]]:
    indices: set[tuple[int, int]] = set()
    malformed: list[str] = []
    for name in names:
        match = _RECURRENT_OUTPUT_RE.fullmatch(name)
        if match:
            layer, step = match.groups()
            indices.add((int(layer), int(step) if step is not None else 0))
        elif "recurrent_state" in name:
            malformed.append(name)
    if malformed:
        raise RuntimeError(
            f"Qwen {phase} graph has malformed recurrent-state outputs: {malformed}"
        )
    return indices


@dataclass(frozen=True)
class QwenHybridCacheContract:
    """Validated recurrent-state ownership for one Qwen target artifact."""

    prefill_recurrent_state_uses_cache: bool
    linear_layers: int
    verify_steps: int

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> "QwenHybridCacheContract":
        model_config = _mapping(
            metadata.get("model_config"),
            field="model_config",
        )
        raw_cache_contract = model_config.get(
            "prefill_recurrent_state_uses_cache",
            False,
        )
        uses_cache = _strict_bool(
            raw_cache_contract,
            field="model_config.prefill_recurrent_state_uses_cache",
        )

        kv_cache = metadata.get("kv_cache")
        linear_layers = 0
        if isinstance(kv_cache, Mapping):
            linear_config = kv_cache.get("linear_kv_cache_config")
            if isinstance(linear_config, Mapping):
                raw_layers = linear_config.get("num_layers", 0)
                if raw_layers != 0:
                    linear_layers = _positive_int(
                        raw_layers,
                        field="kv_cache.linear_kv_cache_config.num_layers",
                    )

        verify_steps = 1
        spec_decode = metadata.get("spec_decode")
        if spec_decode is not None:
            spec = _mapping(spec_decode, field="spec_decode")
            mode = spec.get("mode")
            if mode in {"mtp", "dflash", "dflash2"}:
                num_draft_tokens = _non_negative_int(
                    spec.get("num_draft_tokens"),
                    field="spec_decode.num_draft_tokens",
                )
                verify_steps = num_draft_tokens + 1

        if uses_cache:
            if model_config.get("fuse_gdr_ops") is not True:
                raise ValueError(
                    "model_config.prefill_recurrent_state_uses_cache=true "
                    "requires model_config.fuse_gdr_ops=true"
                )
            if linear_layers <= 0:
                raise ValueError(
                    "Qwen fused recurrent-cache metadata requires a positive "
                    "kv_cache.linear_kv_cache_config.num_layers"
                )

        return cls(
            prefill_recurrent_state_uses_cache=uses_cache,
            linear_layers=linear_layers,
            verify_steps=verify_steps,
        )

    def validate_graphs(self, runtime: object) -> None:
        """Validate target prefill/decode graph signatures after conversion."""

        if not self.prefill_recurrent_state_uses_cache:
            return

        runtime_capability = getattr(
            runtime,
            "_prefill_recurrent_state_uses_cache",
            None,
        )
        if not callable(runtime_capability) or runtime_capability() is not True:
            raise RuntimeError(
                "Qwen HMONNX runtime does not expose the fused prefill "
                "recurrent CacheTensor capability required by the artifact"
            )

        prefill_model = getattr(runtime, "prefill_model", None)
        decode_model = getattr(runtime, "decode_model", None)
        if prefill_model is None or decode_model is None:
            raise RuntimeError(
                "Qwen fused recurrent-cache artifact requires both prefill "
                "and decode target graphs"
            )

        expected_layers = set(range(self.linear_layers))
        prefill_inputs = _recurrent_input_indices(
            _model_io_names(prefill_model, kind="input", phase="prefill"),
            phase="prefill",
        )
        decode_inputs = _recurrent_input_indices(
            _model_io_names(decode_model, kind="input", phase="decode"),
            phase="decode",
        )
        if prefill_inputs != expected_layers:
            raise RuntimeError(
                "Qwen fused prefill recurrent-state input layers mismatch: "
                f"expected={sorted(expected_layers)}, "
                f"actual={sorted(prefill_inputs)}"
            )
        if decode_inputs != expected_layers:
            raise RuntimeError(
                "Qwen decode recurrent-state input layers mismatch: "
                f"expected={sorted(expected_layers)}, "
                f"actual={sorted(decode_inputs)}"
            )

        prefill_outputs = _recurrent_output_indices(
            _model_io_names(prefill_model, kind="output", phase="prefill"),
            phase="prefill",
        )
        if prefill_outputs:
            raise RuntimeError(
                "Qwen fused prefill recurrent CacheTensor graph must not "
                "export recurrent-state outputs, got "
                f"{sorted(prefill_outputs)}"
            )

        decode_outputs = _recurrent_output_indices(
            _model_io_names(decode_model, kind="output", phase="decode"),
            phase="decode",
        )
        expected_decode_outputs = {
            (layer, step)
            for layer in range(self.linear_layers)
            for step in range(self.verify_steps)
        }
        if decode_outputs != expected_decode_outputs:
            missing = sorted(expected_decode_outputs - decode_outputs)
            extra = sorted(decode_outputs - expected_decode_outputs)
            raise RuntimeError(
                "Qwen decode/verify recurrent-state output contract mismatch: "
                f"missing={missing}, extra={extra}"
            )

    def validate_prepared_runtime_state(self, runtime: object) -> None:
        """Ensure the request checkpoint bridge sees every mutable state."""

        if not self.prefill_recurrent_state_uses_cache:
            return
        get_mixin = getattr(runtime, "get_kvcache_mixin", None)
        if not callable(get_mixin):
            raise RuntimeError(
                "Qwen fused recurrent-cache runtime has no KV-cache mixin"
            )
        mixin = get_mixin()
        recurrent_states = list(getattr(mixin, "past_recurrent_states", None) or ())
        if len(recurrent_states) != self.linear_layers:
            raise RuntimeError(
                "Qwen fused recurrent-cache runtime state count mismatch: "
                f"expected={self.linear_layers}, actual={len(recurrent_states)}"
            )


__all__ = ["QwenHybridCacheContract"]
