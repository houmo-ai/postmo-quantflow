from __future__ import annotations

from contextlib import contextmanager
import gc
import logging
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

from xhmodel_merak.model_artifact.files import confined_file

_LOGGER = logging.getLogger(__name__)
_DIGEST_CACHE_NAME = re.compile(r"\.xhquant_.*_sha256_.*\.json")
LLM_HIDDEN_SIZE = 1024


def _is_vocab_logits(value: Any) -> bool:
    """True when a tensor is wider than the hidden size, i.e. real vocabulary logits."""
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return False
    try:
        return int(shape[-1]) > LLM_HIDDEN_SIZE
    except (TypeError, ValueError):
        return False


def convert_published_graph_to_page_attention(source: Path) -> Path:
    """Derive a PageAttention graph beside the published FA/inplace file."""
    from xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model import BaseLLMHMONNXModel

    return Path(BaseLLMHMONNXModel._convert_to_page_attention_hmonnx(source)).resolve()


class MossHMONNXRuntime:
    """Lazy HMONNX sessions for the published MOSS component graphs."""

    def __init__(
        self,
        root: str | Path,
        meta: dict[str, Any],
        device: str,
        *,
        enable_page_attention: bool | None = None,
    ):
        self.root = Path(root).resolve()
        self.meta = meta
        self.exec_devices = tuple(part.strip() for part in str(device).split(",") if part.strip())
        if not self.exec_devices:
            raise ValueError("MOSS runtime requires at least one execution device")
        self.device = self.exec_devices[0]
        self._sessions: dict[str, Any] = {}
        self._golden_components: set[str] = set()
        self._logged_decode_interpreter = False
        self._paged_kv_caches: list[Any] | None = None
        self._page_attention_block_size = 64
        self._page_attention_context_buffers: dict[tuple[str, str], dict[str, Any]] = {}
        self._inplace_layer_caches: dict[str, Any] = {}
        self._graph_overrides: dict[str, Path] = {}
        self._page_attention_loaded = False
        # FA graphs are too slow for greedy decode without PageAttention. 2k FA-off
        # still uses inplace CacheTensor and must not be auto-converted.
        if enable_page_attention is None:
            enable_page_attention = self._default_enable_page_attention()
        if enable_page_attention:
            self._materialize_page_attention_graphs()

    def available(self) -> bool:
        # Published graphs contain XH2a-specific operators. CPU is not a product path.
        if self.device.lower().startswith("cpu"):
            return False
        return all(
            self._component_path(component).is_file()
            for component in ("encode", "prefill", "decode")
        )

    def _decoder_component(self, phase: str) -> dict[str, Any]:
        components = self.meta.get("components", {})
        if not isinstance(components, dict):
            return {}
        decoder = components.get("prefill_decode", {})
        if not isinstance(decoder, dict):
            return {}
        value = decoder.get(phase, {})
        return value if isinstance(value, dict) else {}

    def _default_enable_page_attention(self) -> bool:
        if self.device.lower().startswith("cpu"):
            return False
        model_config = self.meta.get("model_config")
        if not isinstance(model_config, dict):
            return False
        requested = model_config.get("enable_page_attention")
        if isinstance(requested, bool):
            return requested
        flash = model_config.get("flash_attention")
        return bool(isinstance(flash, dict) and flash.get("enable") is True)

    def uses_inplace_cache(self) -> bool:
        """Whether the published decoder metadata declares runtime-owned KV."""
        prefill = self._decoder_component("prefill")
        decode = self._decoder_component("decode")
        return prefill.get("cache_mode") == "inplace" and decode.get("cache_mode") == "inplace"

    def uses_page_attention(self) -> bool:
        """Whether this runtime loaded PageAttention decoder graphs."""
        return self._page_attention_loaded

    def _published_page_attention_graphs(self) -> bool:
        prefill = self._decoder_component("prefill")
        decode = self._decoder_component("decode")
        return bool(prefill.get("enable_page_attention") and decode.get("enable_page_attention"))

    def _existing_page_attention_graph(self, source: Path) -> Path | None:
        """Reuse a derived PA graph next to the published FA file when it is still fresh."""
        matches = [
            path
            for path in source.parent.glob(f"{source.stem}_page_attention_*{source.suffix}")
            if path.is_file() and path.stat().st_size > 0 and path.stat().st_mtime >= source.stat().st_mtime
        ]
        if not matches:
            return None
        return max(matches, key=lambda path: path.stat().st_mtime)

    def _materialize_page_attention_graphs(self) -> None:
        """Derive PageAttention graphs beside the published FA/inplace files."""
        if self._published_page_attention_graphs():
            self._page_attention_loaded = True
            return
        _LOGGER.info("Converting MOSS FA/inplace decoder graphs to PageAttention")
        self._fill_page_attention_geometry()
        decoder = self.meta.get("components", {}).get("prefill_decode", {})
        if not isinstance(decoder, dict):
            raise RuntimeError("MOSS metadata is missing prefill_decode components")
        for phase in ("prefill", "decode"):
            spec = decoder.get(phase)
            if not isinstance(spec, dict):
                raise RuntimeError(f"MOSS metadata is missing {phase} decoder component")
            source = self._component_path(phase)
            if not source.is_file():
                raise RuntimeError(f"MOSS {phase} graph is missing: {source}")
            cached = self._existing_page_attention_graph(source)
            if cached is not None:
                self._graph_overrides[phase] = cached
                continue
            self._graph_overrides[phase] = convert_published_graph_to_page_attention(source)
        self._page_attention_loaded = True

    def _fill_page_attention_geometry(self) -> None:
        prefill = self._decoder_component("prefill")
        if prefill.get("num_layers") and prefill.get("n_kv_head") and prefill.get("head_dim"):
            return
        import onnx

        model = onnx.load(str(self._component_path("prefill")), load_external_data=False)
        k_inputs = [info for info in model.graph.input if info.name.endswith("_self_attn_kcache_input")]
        if not k_inputs:
            return
        dims = [int(dim.dim_value) for dim in k_inputs[0].type.tensor_type.shape.dim]
        if len(dims) < 4:
            return
        geometry = {
            "num_layers": len(k_inputs),
            "n_kv_head": dims[1],
            "head_dim": dims[3],
        }
        for phase in ("prefill", "decode"):
            spec = self._decoder_component(phase)
            if not isinstance(spec, dict):
                continue
            for key, value in geometry.items():
                spec.setdefault(key, value)

    def decoder_cache_length(self) -> int:
        for phase in ("prefill", "decode"):
            value = self._decoder_component(phase).get("cache_length")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        raise RuntimeError("MOSS decoder metadata is missing cache_length")

    def allocate_decoder_cache(self) -> tuple[Any, Any]:
        """Allocate persistent K/V handles for an inplace decoder artifact."""
        if self.uses_page_attention():
            self.ensure_paged_kv_caches()
            return None, None
        if not self.uses_inplace_cache():
            raise RuntimeError("decoder artifact does not declare inplace cache ABI")
        try:
            from xhquant.core import CacheTensor
        except ImportError as error:
            raise RuntimeError("inplace MOSS runtime requires xhquant CacheTensor") from error
        import torch

        session = self._session("prefill")
        names = session.get_input_names()
        layer_names = _layer_cache_names(names)
        if layer_names:
            caches = {}
            for name in layer_names:
                shape = tuple(int(dim) for dim in session.get_input(name).shape)
                caches[name] = CacheTensor(torch.zeros(shape, dtype=torch.float16, device=self.device))
            self._inplace_layer_caches = caches
            return None, None
        key_shape = self.input_shape("prefill", "past_key_cache")
        value_shape = self.input_shape("prefill", "past_value_cache")
        if key_shape != value_shape:
            raise ValueError(f"MOSS key/value cache shapes differ: {key_shape} != {value_shape}")
        zeros = torch.zeros(key_shape, dtype=torch.float16, device=self.device)
        return CacheTensor(zeros), CacheTensor(torch.zeros_like(zeros))

    def ensure_paged_kv_caches(self) -> list[Any]:
        """Allocate identity-mapped HMFP pages shared by prefill and decode."""
        if not self.uses_page_attention():
            raise RuntimeError("decoder artifact does not declare PageAttention")
        import torch
        from xhquant.core.hmfp_kv_cache import allocate_hmfp_paged_kv_cache

        caches = self._paged_kv_caches
        device = torch.device(self.device)
        if caches is not None and caches and all(torch.device(cache.device) == device for cache in caches):
            return caches
        prefill = self._decoder_component("prefill")
        context_length = self.decoder_cache_length()
        num_layers = int(prefill.get("num_layers") or 0)
        n_kv_head = int(prefill.get("n_kv_head") or 0)
        head_dim = int(prefill.get("head_dim") or 0)
        modules = _page_attention_modules(self._session("prefill"))
        if num_layers <= 0 or n_kv_head <= 0 or head_dim <= 0:
            if not modules:
                raise RuntimeError("PageAttention graph has no PageAttention modules")
            num_layers = len(modules)
            n_kv_head = int(getattr(modules[0], "num_kv_heads", 0) or n_kv_head)
            head_dim = int(getattr(modules[0], "head_size", 0) or getattr(modules[0], "head_dim", 0) or head_dim)
        if num_layers <= 0 or n_kv_head <= 0 or head_dim <= 0:
            raise RuntimeError("MOSS PageAttention metadata is missing layer/KV geometry")
        # Never default the width: the page bits decide the cache mantissa dtype,
        # so falling back to 8 here would hand a 16-bit graph an int8 cache, which
        # is the silent mismatch this reads k_bits to avoid. A missing module list
        # is already fatal above when geometry metadata is absent.
        if not modules:
            raise RuntimeError("PageAttention graph has no PageAttention modules; cannot read k_bits/v_bits")
        try:
            k_bits = int(modules[0].k_bits)
            v_bits = int(modules[0].v_bits)
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError("PageAttention module is missing k_bits/v_bits") from error
        qdata_dtype = {8: torch.int8, 16: torch.int16}
        if k_bits not in qdata_dtype or v_bits not in qdata_dtype:
            raise RuntimeError(f"MOSS PageAttention cache bits must be 8 or 16, got k={k_bits}, v={v_bits}")
        block_size = self._page_attention_block_size
        num_blocks = (context_length + block_size - 1) // block_size
        caches = [
            allocate_hmfp_paged_kv_cache(
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=n_kv_head,
                head_size=head_dim,
                head_size_v=head_dim,
                device=device,
                k_qdata_dtype=qdata_dtype[k_bits],
                v_qdata_dtype=qdata_dtype[v_bits],
            )
            for _ in range(num_layers)
        ]
        self._paged_kv_caches = caches
        return caches

    def prepare_page_attention_context(
        self,
        past_seq_length: Any,
        current_input_length: Any,
        *,
        stage: str,
    ) -> None:
        """Bind shared paged caches and pass effective_kv_length into PageAttention."""
        if not self.uses_page_attention():
            return
        import torch
        from xhquant.xhonnxruntime.parsers import PageAttentionContext

        past = _as_length(past_seq_length)
        current = _as_length(current_input_length)
        if past < 0:
            raise ValueError(f"past_seq_length must be non-negative, got {past}")
        if current < 0:
            raise ValueError(f"current_input_length must be non-negative, got {current}")
        context_length = self.decoder_cache_length()
        effective_kv_length = past + current
        if effective_kv_length > context_length:
            raise ValueError(
                f"Page-attention request position {effective_kv_length} exceeds cache_length {context_length}"
            )
        graph_length = self.input_shape(stage, "inputs_embeds")[1]
        if current > graph_length:
            raise ValueError(
                f"current_input_length {current} exceeds {stage} graph length {graph_length}"
            )
        caches = self.ensure_paged_kv_caches()
        device = torch.device(self.device)
        block_size = self._page_attention_block_size
        num_blocks = int(caches[0].num_blocks)
        block_ids = torch.arange(num_blocks, dtype=torch.int64, device=device)
        slot_mapping = torch.full((graph_length,), -1, dtype=torch.int64, device=device)
        if current:
            slot_mapping[:current] = torch.arange(past, past + current, dtype=torch.int64, device=device)
        staged_block_ids, staged_slot_mapping = self._stage_page_attention_metadata(
            stage, caches, block_ids, slot_mapping,
        )
        for name in ("prefill", "decode"):
            session = self._sessions.get(name)
            if session is None and name != stage:
                continue
            session = self._session(name)
            modules = _page_attention_modules(session)
            if len(modules) != len(caches):
                raise ValueError(
                    "PageAttention module/cache count mismatch: "
                    f"{len(modules)} modules vs {len(caches)} caches."
                )
            for module, cache in zip(modules, caches, strict=True):
                module.set_context(
                    PageAttentionContext(
                        paged_kv_cache=cache,
                        block_ids=staged_block_ids,
                        slot_mapping=staged_slot_mapping,
                        block_size=block_size,
                        effective_kv_length=effective_kv_length,
                    )
                )
            interpreter = getattr(session, "interpreter", None)
            setter = getattr(interpreter, "set_capture_cache_key", None)
            if callable(setter):
                length_class = None
                capture = getattr(modules[0], "capture_cache_length_class", None) if modules else None
                if callable(capture):
                    length_class = capture(effective_kv_length)
                setter(("page_attention", name, num_blocks, length_class))

    def _stage_page_attention_metadata(
        self,
        stage: str,
        caches: list[Any],
        block_ids: Any,
        slot_mapping: Any,
    ) -> tuple[Any, Any]:
        """Copy PA metadata into capture-stable, fixed-address buffers."""
        import torch

        device = torch.device(self.device)
        key = (stage, str(device))
        buffers = self._page_attention_context_buffers.get(key)
        if buffers is None:
            buffers = {}
            self._page_attention_context_buffers[key] = buffers
        block_ids_flat = block_ids.reshape(-1)
        slot_mapping_flat = slot_mapping.reshape(-1)
        block_capacity = max(int(cache.num_blocks) for cache in caches)
        slot_capacity = int(slot_mapping_flat.numel())
        block_buffer = buffers.get("block_ids")
        slot_buffer = buffers.get("slot_mapping")
        grew = False
        if not isinstance(block_buffer, torch.Tensor) or block_buffer.numel() < block_capacity:
            grew = isinstance(block_buffer, torch.Tensor)
            block_buffer = torch.empty(block_capacity, dtype=torch.int64, device=device)
            buffers["block_ids"] = block_buffer
        if not isinstance(slot_buffer, torch.Tensor) or slot_buffer.numel() < slot_capacity:
            grew = grew or isinstance(slot_buffer, torch.Tensor)
            slot_buffer = torch.empty(slot_capacity, dtype=torch.int64, device=device)
            buffers["slot_mapping"] = slot_buffer
        if block_ids_flat.numel() > block_buffer.numel():
            raise ValueError(
                f"block_ids length {block_ids_flat.numel()} exceeds cache capacity {block_buffer.numel()}"
            )
        if device.type == "cuda":
            with torch.cuda.device(device):
                block_buffer.zero_()
                block_buffer[: block_ids_flat.numel()].copy_(block_ids_flat, non_blocking=True)
                slot_buffer.fill_(-1)
                slot_buffer[: slot_mapping_flat.numel()].copy_(slot_mapping_flat, non_blocking=True)
        else:
            block_buffer.zero_()
            block_buffer[: block_ids_flat.numel()].copy_(block_ids_flat)
            slot_buffer.fill_(-1)
            slot_buffer[: slot_mapping_flat.numel()].copy_(slot_mapping_flat)
        if grew:
            for name in ("prefill", "decode"):
                interpreter = getattr(self._sessions.get(name), "interpreter", None)
                clear = getattr(interpreter, "clear", None)
                if callable(clear):
                    clear(clear_disabled_reason=True)
        return block_buffer[: block_ids_flat.numel()], slot_buffer[: slot_mapping_flat.numel()]

    def _component_path(self, component: str) -> Path:
        override = self._graph_overrides.get(component)
        if override is not None:
            return override
        components = self.meta.get("components", {})
        if not isinstance(components, dict):
            return self.root / "__missing_hmonnx__"
        value = (
            components.get("encode", {}).get("hmonnx_file")
            if component == "encode"
            else components.get("prefill_decode", {}).get(component, {}).get("hmonnx_file")
        )
        if not isinstance(value, str):
            return self.root / "__missing_hmonnx__"
        try:
            return confined_file(self.root, value, f"{component}.hmonnx_file")
        except Exception:
            # ``available()`` is a probe; invalid metadata must make the
            # component unavailable rather than allowing an external path.
            return self.root / "__missing_hmonnx__"

    def _hmonnx_session_config(self, component: str):
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig

        auto_offload = len(self.exec_devices) > 1
        enable_cuda_graph = (
            component == "decode"
            and self.device.lower().startswith("cuda")
            and not auto_offload
            and component not in self._golden_components
        )
        return HMONNXInferenceConfig(
            enable_fast=False,
            enable_auto_offload=auto_offload,
            enable_cuda_graph=enable_cuda_graph,
            enable_golden=component in self._golden_components,
            exec_devices=list(self.exec_devices),
        )

    def _bind_v2_session(self, session: Any) -> Any:
        # V2 pins devices through HMONNXInferenceConfig.exec_devices; session.to
        # is a no-op there and only misleads operators.
        _rewrapping_kvcache_slices(session)
        return session

    def _log_decode_interpreter(self, session: Any) -> None:
        if self._logged_decode_interpreter:
            return
        self._logged_decode_interpreter = True
        interpreter = getattr(session, "interpreter", None)
        name = type(interpreter).__name__ if interpreter is not None else None
        reason = getattr(interpreter, "_capture_disabled_reason", None)
        message = f"MOSS decode interpreter={name} capture_disabled_reason={reason!r}"
        try:
            from xhquant.api import get_xhquant_logger

            get_xhquant_logger().info(message)
        except Exception:
            _LOGGER.info(message)

    def _try_open_shared_decoder_sessions(self) -> bool:
        """Load prefill/decode once through LLMHMONNXLoader, like Laguna chat."""
        if "prefill" in self._sessions and "decode" in self._sessions:
            return True
        prefill_path = self._component_path("prefill")
        decode_path = self._component_path("decode")
        if not prefill_path.is_file() or not decode_path.is_file():
            return False
        try:
            from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceV2
            from xhquant.xhonnxruntime.llm_hmonnx_loader import LLMHMONNXLoader
        except ImportError:
            return False
        from_onnx_graph = getattr(HMONNXInferenceV2, "from_onnx_graph", None)
        if not callable(from_onnx_graph):
            return False
        before = _digest_cache_files(self.root)
        try:
            loader = LLMHMONNXLoader(str(prefill_path), str(decode_path))
            prefill = self._bind_v2_session(
                from_onnx_graph(str(prefill_path), loader.prefill_graph, self._hmonnx_session_config("prefill"))
            )
            decode_config = self._hmonnx_session_config("decode")
            get_layer_infos = getattr(prefill, "get_layer_infos", None)
            if callable(get_layer_infos):
                decode_config.layers = get_layer_infos()
            decode = self._bind_v2_session(
                from_onnx_graph(str(decode_path), loader.decode_graph, decode_config)
            )
        except Exception:
            _LOGGER.exception("MOSS shared prefill/decode load failed; falling back to independent HMONNX sessions")
            return False
        finally:
            _purge_new_digest_caches(self.root, before)
        self._sessions["prefill"] = prefill
        self._sessions["decode"] = decode
        return True

    def _open_independent_v2_session(self, component: str) -> Any:
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceV2

        before = _digest_cache_files(self.root)
        try:
            return self._bind_v2_session(
                HMONNXInferenceV2(str(self._component_path(component)), self._hmonnx_session_config(component))
            )
        finally:
            _purge_new_digest_caches(self.root, before)

    def _session(self, component: str) -> Any:
        session = self._sessions.get(component)
        if session is None:
            # Keep the injectable legacy seam used by lightweight tests, while
            # production MOSS sessions use V2 by default.
            try:
                from xhquant.api import HMONNXInference as legacy_class
            except ImportError:
                legacy_class = None
            if legacy_class is not None and not legacy_class.__module__.startswith("xhquant."):
                session = legacy_class(str(self._component_path(component)))
                session.to(self.device)
                self._sessions[component] = session
                return session
            if component in {"prefill", "decode"} and self._try_open_shared_decoder_sessions():
                return self._sessions[component]
            session = self._open_independent_v2_session(component)
            self._sessions[component] = session
        return session

    def enable_golden(self, components: tuple[str, ...]) -> None:
        """Enable HMONNX golden capture next to each published graph, matching dump_golden."""
        self._golden_components.update(components)
        if self._golden_components & {"prefill", "decode"}:
            self._sessions.pop("prefill", None)
            self._sessions.pop("decode", None)
        for component in components:
            self._sessions.pop(component, None)
            session = self._session(component)
            golden_dir = golden_dir_for_graph(self._component_path(component))
            golden_dir.mkdir(parents=True, exist_ok=True)
            session.save_golden = True
            session.save_golden_dir = str(golden_dir)

    def run(self, component: str, inputs: dict[str, Any]) -> Any:
        if component not in {"encode", "prefill", "decode"}:
            raise ValueError(f"unknown MOSS HMONNX component: {component}")
        session = self._session(component)
        names = session.get_input_names()
        mapped = _map_feed(names, inputs)
        if self._inplace_layer_caches:
            mapped.update(self._inplace_layer_caches)
        missing = [name for name in names if name not in mapped]
        if missing:
            raise ValueError(f"missing HMONNX inputs for {component}: {missing}")
        feed = {}
        for name in names:
            value = mapped[name]
            info = session.get_input(name)
            if value.__class__.__name__ == "CacheTensor":
                # CacheTensor ownership must survive the feed boundary; a
                # regular ``to`` conversion would unwrap the inplace handle.
                feed[name] = value
                continue
            if hasattr(value, "to") and getattr(info, "dtype", None) is not None:
                value = value.to(device=self.device, dtype=info.dtype)
            feed[name] = value
        with _runtime_tool_path():
            outputs = session.run(feed)
        if component == "decode":
            self._log_decode_interpreter(session)
        out_names = self.output_names(component)
        if isinstance(outputs, (tuple, list)):
            if len(outputs) != len(out_names):
                raise ValueError(
                    f"HMONNX {component} returned {len(outputs)} outputs, expected {len(out_names)}"
                )
            result = dict(zip(out_names, outputs, strict=True))
        elif len(out_names) != 1:
            raise ValueError(f"HMONNX {component} returned one output, expected {len(out_names)}")
        else:
            result = {out_names[0]: outputs}
        # Graphs without an in-graph lm_head publish the final hidden state under
        # the name "logits". Alias it so callers can read last_hidden_state, but
        # never alias a real vocabulary logits tensor: the host would then apply
        # the output projection a second time.
        if "logits" in result and "last_hidden_state" not in result and not _is_vocab_logits(result["logits"]):
            result["last_hidden_state"] = result["logits"]
        return result

    def input_shape(self, component: str, name: str) -> tuple[int, ...]:
        session = self._session(component)
        actual = _resolve_session_name(session.get_input_names(), name) or name
        info = session.get_input(actual)
        return tuple(int(dim) for dim in info.shape)

    def output_names(self, component: str) -> tuple[str, ...]:
        return tuple(self._session(component).get_output_names())

    def encode(self, input_features: Any) -> Any:
        return self.run("encode", {"input_features": input_features})

    def prefill(self, inputs_embeds: Any, past_seq_length: Any, current_input_length: Any, key_cache: Any = None, value_cache: Any = None) -> Any:
        if self.uses_page_attention():
            self.prepare_page_attention_context(past_seq_length, current_input_length, stage="prefill")
        return self.run(
            "prefill",
            _decoder_feed(inputs_embeds, past_seq_length, current_input_length, key_cache, value_cache),
        )

    def decode(self, inputs_embeds: Any, past_seq_length: Any, current_input_length: Any, key_cache: Any = None, value_cache: Any = None) -> Any:
        if self.uses_page_attention():
            self.prepare_page_attention_context(past_seq_length, current_input_length, stage="decode")
        return self.run(
            "decode",
            _decoder_feed(inputs_embeds, past_seq_length, current_input_length, key_cache, value_cache),
        )

    def generate(self, fused, lm_head, tokenizer, root, max_new_tokens=None):
        from .transcribe import generate_hmonnx
        import numpy as np
        return generate_hmonnx(
            np.asarray(fused), np.ones(len(fused)), self, lm_head, tokenizer, Path(root),
            max_new_tokens=max_new_tokens,
        )

    def close(self) -> None:
        """Drop HMONNX sessions and device caches after a transcription run."""
        sessions = list(self._sessions.values())
        self._sessions.clear()
        self._paged_kv_caches = None
        self._page_attention_context_buffers.clear()
        self._inplace_layer_caches.clear()
        for session in sessions:
            closer = getattr(session, "close", None) or getattr(session, "release", None)
            if callable(closer):
                closer()
        gc.collect()
        if not str(self.device).lower().startswith("cuda"):
            return
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()


def golden_dir_for_graph(graph: Path) -> Path:
    return graph.parent / "golden"


__all__ = ["MossHMONNXRuntime", "convert_published_graph_to_page_attention", "golden_dir_for_graph"]


def _as_length(value: Any) -> int:
    item = getattr(value, "reshape", None)
    if callable(item):
        return int(value.reshape(-1)[0].item())
    return int(value)


_INPUT_ALIASES = {
    "inputs_embeds": ("inputs_embeds", "input_1"),
    "past_seq_length": ("past_seq_length", "valid_length"),
    "current_input_length": ("current_input_length", "current_length"),
    "past_key_cache": ("past_key_cache",),
    "past_value_cache": ("past_value_cache",),
    "last_hidden_state": ("last_hidden_state", "logits"),
    "input_features": ("input_features",),
}


def _resolve_session_name(names: list[str] | tuple[str, ...], logical: str) -> str | None:
    present = set(names)
    if logical in present:
        return logical
    for candidate in _INPUT_ALIASES.get(logical, (logical,)):
        if candidate in present:
            return candidate
    return None


def _map_feed(names: list[str] | tuple[str, ...], inputs: dict[str, Any]) -> dict[str, Any]:
    mapped = {name: value for name, value in inputs.items() if name in names}
    for logical, value in inputs.items():
        actual = _resolve_session_name(names, logical)
        if actual is not None and actual not in mapped:
            mapped[actual] = value
    return mapped


def _layer_cache_names(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    found = [name for name in names if name.endswith("_kcache_input") or name.endswith("_vcache_input")]

    def _layer_index(name: str) -> int:
        for part in name.split("_"):
            if part.isdigit():
                return int(part)
        return 0

    return tuple(sorted(found, key=_layer_index))


def _decoder_feed(
    inputs_embeds: Any,
    past_seq_length: Any,
    current_input_length: Any,
    key_cache: Any,
    value_cache: Any,
) -> dict[str, Any]:
    feed = {
        "inputs_embeds": inputs_embeds,
        "past_seq_length": past_seq_length,
        "current_input_length": current_input_length,
    }
    if key_cache is not None:
        feed["past_key_cache"] = key_cache
    if value_cache is not None:
        feed["past_value_cache"] = value_cache
    return feed


def _page_attention_modules(session: Any) -> list[Any]:
    try:
        from xhquant.xhonnxruntime.parsers import PageAttention
    except ImportError:
        return []
    graph_module = getattr(getattr(session, "interpreter", None), "_graph_module", None)
    if graph_module is None:
        graph_module = getattr(session, "graph_module", None)
    if graph_module is None:
        node_modules = getattr(session, "node_modules", None)
        if node_modules is not None:
            return [module for module in node_modules if isinstance(module, PageAttention)]
        return []
    modules = []
    graph = getattr(graph_module, "graph", None)
    if graph is None:
        return [module for module in graph_module.modules() if isinstance(module, PageAttention)]
    for node in graph.nodes:
        if node.op == "call_module":
            module = graph_module.get_submodule(str(node.target))
            if isinstance(module, PageAttention):
                modules.append(module)
    return modules


def _digest_cache_files(root: Path) -> set[Path]:
    return {
        path
        for path in root.rglob("*")
        if path.is_file() and _DIGEST_CACHE_NAME.fullmatch(path.name)
    }


def _purge_new_digest_caches(root: Path, before: set[Path]) -> None:
    for path in _digest_cache_files(root) - before:
        path.unlink(missing_ok=True)


def _rewrapping_kvcache_slices(session: Any) -> None:
    """HMONNX slices stacked KV into per-layer views; KVcache still requires CacheTensor."""
    try:
        from xhquant.core import CacheTensor
        from xhquant.xhonnxruntime.parsers.llm_cache import LLMCache
    except ImportError:
        return
    graph_module = getattr(getattr(session, "interpreter", None), "_graph_module", None)
    if graph_module is None:
        graph_module = getattr(session, "graph_module", None)
    if graph_module is None:
        return
    for module in graph_module.modules():
        if not isinstance(module, LLMCache) or not callable(getattr(module, "_orig_forward", None)):
            continue
        original = module._orig_forward

        def wrapped(
            states,
            past_seq_length,
            current_seq_length,
            past_cache,
            accepted_count=None,
            _original=original,
        ):
            if not isinstance(past_cache, CacheTensor):
                past_cache = CacheTensor(past_cache)
            return _original(states, past_seq_length, current_seq_length, past_cache, accepted_count)

        module._orig_forward = wrapped


@contextmanager
def _runtime_tool_path():
    """Expose tools installed beside the active Python for JIT kernels."""
    environment_bin = Path(sys.executable).resolve().parent
    ninja = environment_bin / "ninja"
    if not ninja.is_file() or shutil.which("ninja"):
        yield
        return
    previous = os.environ.get("PATH")
    os.environ["PATH"] = f"{environment_bin}{os.pathsep}{previous}" if previous else str(environment_bin)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous
