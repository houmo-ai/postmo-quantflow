"""Gemma4 shared-KV MTP runtime, used by unified and the diagnostic CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .mtp_contract import resolve_readonly_sliding_cache_range

EXPECTED_BASE_OUTPUTS = ["logits", "target_hidden_state"]


def _load_inputs(path):
    import onnx

    return [item.name for item in onnx.load(path, load_external_data=False).graph.input]


def _aligned(size: int, alignment: int = 16) -> int:
    return ((int(size) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _meta_spec_decode(meta_dict: dict[str, Any]) -> dict[str, Any]:
    spec = dict(meta_dict.get("spec_decode") or {})
    if not spec and meta_dict.get("spec_decode_mode"):
        spec = {
            "mode": meta_dict.get("spec_decode_mode"),
            "block_size": meta_dict.get("spec_decode_block_size"),
            "verify_length": meta_dict.get("spec_decode_verify_length"),
            "hidden_output_name": meta_dict.get("spec_decode_hidden_output_name"),
            "draft_head_weight_bits": meta_dict.get("spec_decode_draft_head_weight_bits"),
        }
    return spec


class AssistantDraftSession:
    """Thin HMONNXGraph wrapper for the exported assistant ONNX."""

    def __init__(self, onnx_path: str | Path, device):
        graph_input_names = _load_inputs(Path(onnx_path))
        if "shared_key_cache_sliding" not in graph_input_names:
            # Reject before constructing/moving HMONNX. Loading the production
            # PageAttention graph as an independent session cannot bind the
            # target runtime's cache owner and may reserve substantial useless
            # device memory before failing.
            raise RuntimeError(
                "Standalone Gemma4 MTP generation requires the shared-tensor "
                "draft graph. The read-only PageAttention deploy graph relies "
                "on Merak to bind target-owned paged caches. Use the manifest's "
                "spec_decode.standalone_draft_decode_onnx artifact instead."
            )
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceV2, HMONNXInferenceConfig

        self.session = HMONNXInferenceV2(str(onnx_path), HMONNXInferenceConfig(exec_devices=[device]))
        self.input_names = self.session.get_input_names()
        self.output_names = self.session.get_output_names()
        self.input_infos = {name: self.session.get_input(name) for name in self.input_names}
        self.input_aliases = {
            "input_1": "inputs_embeds",
            "valid_length": "past_seq_length",
            "current_length": "current_length",
        }
        self.device = device

    def __call__(self, **inputs):
        import torch

        feed = {}
        for name in self.input_names:
            source_name = name if name in inputs else self.input_aliases.get(name)
            if source_name is None or source_name not in inputs:
                raise KeyError(f"Missing assistant input {name!r}; available={sorted(inputs)}")
            value = inputs[source_name]
            if isinstance(value, torch.Tensor):
                value = value.detach().to(self.device)
                expected_shape = tuple(int(dim) for dim in getattr(self.input_infos[name], "shape", ()) or ())
                if expected_shape and tuple(value.shape) != expected_shape:
                    raise RuntimeError(
                        f"Assistant input {name!r} shape mismatch: got {tuple(value.shape)}, "
                        f"expected {expected_shape}. Re-export target and draft from one "
                        "manifest instead of slicing caches."
                    )
                expected_dtype = self.input_infos[name].dtype
                if value.dtype != expected_dtype:
                    value = value.to(expected_dtype)
            feed[name] = value
        outputs = self.session.run(feed)
        if isinstance(outputs, dict):
            output_map = outputs
        else:
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            output_map = {name: output for name, output in zip(self.output_names, outputs, strict=True)}
        return output_map[self.output_names[0]], output_map[self.output_names[1]]


def _load_prompt_inputs(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    else:
        text = prompt
    return tokenizer([text], return_tensors="pt").input_ids


def _read_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc


def _generation_config_candidates(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Return generation/stop-token config sources in runtime order.

    Gemma4 chat models encode end-of-turn as EOS candidates in HF
    ``generation_config.json`` (for 31B this is ``[1, 106, 50]``).  The
    exported manifest keeps the original HF files under ``hf_config/`` instead
    of flattening every generation field into the top-level metadata, so the
    speculative loop must read those source files rather than relying only on
    ``tokenizer.eos_token_id``.
    """

    configs: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def add_file(path: Path) -> None:
        resolved = path.expanduser()
        if not resolved.is_absolute():
            resolved = resolved.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        data = _read_json_config(resolved)
        if data:
            configs.append(data)

    meta_path_value = meta.get("_meta_path") or meta.get("_meta_path_")
    export_dir = Path(meta_path_value).expanduser().resolve().parent if meta_path_value else None
    hf_config = meta.get("hf_config")
    if export_dir is not None and hf_config:
        hf_config_dir = Path(str(hf_config))
        if not hf_config_dir.is_absolute():
            hf_config_dir = export_dir / hf_config_dir
        add_file(hf_config_dir / "generation_config.json")
        add_file(hf_config_dir / "config.json")

    model_config = meta.get("model_config") if isinstance(meta.get("model_config"), dict) else {}
    hf_model = model_config.get("hf_model") if isinstance(model_config, dict) else None
    if hf_model:
        hf_model_dir = Path(str(hf_model)).expanduser()
        add_file(hf_model_dir / "generation_config.json")
        add_file(hf_model_dir / "config.json")

    inline_generation = meta.get("generation_config")
    if isinstance(inline_generation, dict):
        configs.append(inline_generation)
    if isinstance(model_config, dict):
        configs.append(model_config)
    return configs


def _resolve_eos_token_ids(tokenizer, *configs: dict[str, Any]) -> set[int]:
    token_ids: list[int] = []
    for config in configs:
        value = config.get("eos_token_id") if isinstance(config, dict) else None
        if value is None:
            continue
        if isinstance(value, int):
            value = [value]
        for token_id in value:
            token_id = int(token_id)
            if token_id not in token_ids:
                token_ids.append(token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        for token_id in eos_token_id:
            token_id = int(token_id)
            if token_id not in token_ids:
                token_ids.append(token_id)
    return set(token_ids)


def _parse_target_outputs(outputs) -> tuple[Any, Any, dict[str, Any]]:
    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError("Target HMONNX must be exported with enable_mtp_outputs=True; got one output only.")
    if len(outputs) < len(EXPECTED_BASE_OUTPUTS):
        raise RuntimeError(f"Target HMONNX returned {len(outputs)} outputs, expected {len(EXPECTED_BASE_OUTPUTS)}.")
    return outputs[0], outputs[1], {}


def _scalar_debug_value(value: Any) -> int | float | str:
    try:
        import torch

        if torch.is_tensor(value):
            if value.numel() == 1:
                return int(value.detach().cpu().reshape(-1)[0].item())
            return f"tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return repr(value)


def _run_target_prefill(target_model, input_ids):

    if hasattr(target_model, "run_prefill_ids"):
        return target_model.run_prefill_ids(input_ids)

    prefill_chunk_length = int(target_model.meta_info.model_config.prefill_chunk_length)
    past_seq_length = 0
    logits = hidden = shared = None
    for start in range(0, input_ids.shape[1], prefill_chunk_length):
        chunk = input_ids[:, start : start + prefill_chunk_length]
        target_model.set_prefill()
        target_model.set_input_sequence_length(prefill_chunk_length)
        processor = target_model.get_data_preprocessor()
        model_inputs = processor({"input_ids": chunk, "past_seq_length": past_seq_length})
        outputs = target_model.forward(*model_inputs)
        logits, hidden, _ = _parse_target_outputs(outputs)
        shared = _hmonnx_shared_from_cache(target_model)
        past_seq_length += int(chunk.shape[1])
    assert logits is not None and hidden is not None and shared is not None
    return logits, hidden, shared, past_seq_length


def _run_target_verify(target_model, token_ids: list[int], past_seq_length: int):
    import torch

    if hasattr(target_model, "run_verify_ids"):
        return target_model.run_verify_ids(token_ids, past_seq_length)

    if not token_ids:
        raise ValueError("target verify requires at least one token")
    tokens = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=target_model.device)
    target_model.set_decode()
    target_model.set_input_sequence_length(len(token_ids))
    processor = target_model.get_data_preprocessor()
    verify_accepted_count = int(getattr(target_model, "_mtp_next_verify_accepted_count", 0))
    model_inputs = processor(
        {
            "input_ids": tokens,
            "past_seq_length": int(past_seq_length),
            "accepted_count": verify_accepted_count,
        }
    )
    if getattr(target_model, "_mtp_trace_accepted_count", False):
        input_names = list(getattr(getattr(target_model, "llm", None), "input_names", []) or [])
        if not input_names and hasattr(target_model, "get_input_names"):
            try:
                input_names = list(target_model.get_input_names())
            except Exception:
                input_names = []
        accepted_tensor = model_inputs[-3] if len(model_inputs) >= 3 else verify_accepted_count
        print(
            "[MTP accepted_count trace] "
            f"verify_round={int(getattr(target_model, '_mtp_verify_round_index', 0))} "
            f"target_decode_input accepted_count={_scalar_debug_value(accepted_tensor)} "
            f"expected_previous_accepted={verify_accepted_count} "
            f"past_seq_length={int(past_seq_length)} "
            f"verify_tokens={list(map(int, token_ids))}"
        )
        if input_names:
            print(f"[MTP accepted_count trace] target graph inputs={input_names}")
    outputs = target_model.forward(*model_inputs)
    logits, hidden, _ = _parse_target_outputs(outputs)
    return logits, hidden, _hmonnx_shared_from_cache(target_model)


def _cache_to_tensor(cache):
    import torch

    if hasattr(cache, "data") and torch.is_tensor(cache.data):
        return cache.data
    if torch.is_tensor(cache):
        return cache
    raise TypeError(f"Unsupported cache object: {type(cache)!r}")


def _clone_cache_tensor(cache):
    import torch

    from xhquant.core import CacheTensor

    if hasattr(cache, "data") and torch.is_tensor(cache.data):
        cloned = type(cache)(cache.data.detach().clone())
        if hasattr(cache, "cache_valid_len"):
            cloned.cache_valid_len = int(cache.cache_valid_len)
        return cloned
    if torch.is_tensor(cache):
        return CacheTensor(cache.detach().clone())
    raise TypeError(f"Unsupported cache object: {type(cache)!r}")


def _snapshot_hmonnx_kv_cache(target_model):
    return (
        [_clone_cache_tensor(cache) for cache in target_model.past_key_caches],
        [_clone_cache_tensor(cache) for cache in target_model.past_value_caches],
    )


def _hmonnx_shared_cache_indices(target_model) -> dict[str, int]:
    source: dict[str, int] = {}
    layer_cache_types = list(getattr(target_model, "layer_cache_types", []) or [])
    if layer_cache_types:
        for cache_idx, layer_type in enumerate(layer_cache_types):
            if layer_type in {"sliding_attention", "full_attention"}:
                source[str(layer_type)] = cache_idx
    else:
        layer_kv_shapes = list(
            getattr(getattr(target_model, "_kvcache_mixin", None), "layer_kv_shapes", None)
            or getattr(getattr(target_model, "meta_info", None), "layer_kv_shapes", None)
            or []
        )
        full_cache_len = int(getattr(getattr(target_model, "kvcache_config", None), "max_sequence_length", 0) or 0)
        if full_cache_len <= 0 and layer_kv_shapes:
            full_cache_len = max(int(shape[2]) for shape in layer_kv_shapes if len(shape) > 2)
        for cache_idx, shape in enumerate(layer_kv_shapes):
            if len(shape) <= 2:
                continue
            cache_len = int(shape[2])
            layer_type = "sliding_attention" if full_cache_len > 0 and cache_len < full_cache_len else "full_attention"
            source[layer_type] = cache_idx
        if not source:
            layer_types = list(getattr(target_model, "layer_types", []) or [])
            cache_count = len(getattr(target_model, "past_key_caches", []) or [])
            # Safe compatibility fallback only when layer_types already describes
            # the cache list.  Do not use full model layer indices when shared-KV
            # layers were filtered out of past_key_caches.
            if layer_types and cache_count and len(layer_types) == cache_count:
                for cache_idx, layer_type in enumerate(layer_types):
                    if layer_type in {"sliding_attention", "full_attention"}:
                        source[str(layer_type)] = cache_idx
    if "sliding_attention" not in source or "full_attention" not in source:
        raise RuntimeError(
            "Cannot locate Gemma4 HMONNX shared KV cache indices from cache metadata: "
            f"layer_cache_types={layer_cache_types}, source={source}"
        )
    return source


def _hmonnx_shared_from_cache(target_model) -> dict[str, Any]:
    source = _hmonnx_shared_cache_indices(target_model)
    sliding_idx = source["sliding_attention"]
    full_idx = source["full_attention"]
    return {
        "shared_key_cache_sliding": _cache_to_tensor(target_model.past_key_caches[sliding_idx]),
        "shared_value_cache_sliding": _cache_to_tensor(target_model.past_value_caches[sliding_idx]),
        "shared_key_cache_full": _cache_to_tensor(target_model.past_key_caches[full_idx]),
        "shared_value_cache_full": _cache_to_tensor(target_model.past_value_caches[full_idx]),
    }


def _commit_one_hmonnx_cache(
    before_cache, verified_cache, past_seq_length: int, commit_length: int, verify_length: int
):
    del before_cache
    committed_cache = _clone_cache_tensor(verified_cache)
    committed_tensor = _cache_to_tensor(committed_cache)

    if hasattr(committed_cache, "cache_valid_len"):
        verified_valid = min(
            max(int(getattr(verified_cache, "cache_valid_len", 0)), 0),
            int(committed_tensor.shape[2]),
        )
        rollback = max(0, int(verify_length) - int(commit_length))
        new_valid = max(0, verified_valid - rollback)
        if new_valid < int(committed_tensor.shape[2]):
            committed_tensor[:, :, new_valid:, :].zero_()
        committed_cache.cache_valid_len = new_valid
        return committed_cache

    start = max(int(past_seq_length), 0)
    accepted_end = min(int(committed_tensor.shape[2]), start + int(commit_length))
    verify_end = min(int(committed_tensor.shape[2]), start + int(verify_length))
    if accepted_end < verify_end:
        committed_tensor[:, :, accepted_end:verify_end, :].zero_()
    return committed_cache


def _commit_hmonnx_verified_cache(
    target_model,
    snapshot,
    *,
    past_seq_length: int,
    commit_length: int,
    verify_length: int,
) -> dict[str, Any]:
    before_key_caches, before_value_caches = snapshot
    committed_key_caches = [
        _commit_one_hmonnx_cache(before, verified, past_seq_length, commit_length, verify_length)
        for before, verified in zip(
            before_key_caches,
            target_model.past_key_caches,
            strict=True,
        )
    ]
    committed_value_caches = [
        _commit_one_hmonnx_cache(before, verified, past_seq_length, commit_length, verify_length)
        for before, verified in zip(
            before_value_caches,
            target_model.past_value_caches,
            strict=True,
        )
    ]
    target_model._kvcache_mixin.past_key_caches.clear()
    target_model._kvcache_mixin.past_key_caches.extend(committed_key_caches)
    target_model._kvcache_mixin.past_value_caches.clear()
    target_model._kvcache_mixin.past_value_caches.extend(committed_value_caches)
    return _hmonnx_shared_from_cache(target_model)


def _target_needs_manual_cache_commit(target_model) -> bool:
    return (
        not hasattr(target_model, "commit_cache")
        and hasattr(target_model, "past_key_caches")
        and hasattr(target_model, "past_value_caches")
        and hasattr(target_model, "_kvcache_mixin")
    )


def _build_draft_masks(
    target_model,
    assistant_session: AssistantDraftSession,
    cache_valid_length: int | None = None,
    *,
    sliding_valid_length: int | None = None,
):
    import torch

    dtype = target_model.dtype
    device = target_model.device
    neg = torch.tensor(torch.finfo(torch.float16).min, dtype=torch.float16, device=device).to(dtype=dtype)
    model_config = target_model.meta_info.model_config
    sliding_window = int(getattr(target_model, "sliding_window", getattr(model_config, "sliding_window", 1024)) or 1024)
    sliding_width = int(assistant_session.input_infos["sliding_attention_mask"].shape[-1])
    sliding_mask = torch.full((1, 1, 1, sliding_width), neg, dtype=dtype, device=device)
    if sliding_valid_length is None:
        sliding_valid_length = int(cache_valid_length or 0)

    # Sliding shared KV is already in LLMCache's compact local coordinates.
    # Expose the suffix ending at its HybridCacheTensor cache_valid_len.  This
    # value is not always the absolute full-cache length after sliding compacts
    # a saturated window during MTP verify.
    local_valid = min(sliding_width, max(1, int(sliding_valid_length)))
    sliding_end = local_valid
    sliding_start = max(0, sliding_end - sliding_window)
    if sliding_end <= sliding_start:
        sliding_mask[0, 0, 0, 0] = 0
    else:
        sliding_mask[0, 0, 0, sliding_start:sliding_end] = 0
    return sliding_mask


def _hmonnx_shared_cache_valid_lengths(target_model, fallback_full_valid: int) -> tuple[int, int]:
    try:
        source = _hmonnx_shared_cache_indices(target_model)
        sliding_cache = target_model.past_key_caches[source["sliding_attention"]]
        sliding_valid = int(getattr(sliding_cache, "cache_valid_len", fallback_full_valid) or fallback_full_valid)
    except Exception:
        sliding_valid = int(fallback_full_valid)
    return int(fallback_full_valid), int(sliding_valid)


def _build_assistant_inputs(
    target_model,
    assistant_session: AssistantDraftSession,
    last_token_id: int,
    current_hidden,
    shared: dict[str, Any],
    position_index: int,
    cache_valid_length: int | None = None,
    *,
    sliding_valid_length: int | None = None,
):
    import torch

    device = target_model.device
    input_names = set(
        getattr(
            assistant_session,
            "input_names",
            assistant_session.input_infos.keys(),
        )
    )
    token = torch.tensor([[int(last_token_id)]], dtype=torch.long, device=device)
    token_embed = target_model.get_input_embeddings().to(device)(token).to(dtype=current_hidden.dtype)
    inputs_embeds = torch.cat([token_embed, current_hidden], dim=-1)
    sliding_mask = None
    if "sliding_attention_mask" in input_names:
        sliding_mask = _build_draft_masks(
            target_model,
            assistant_session,
            cache_valid_length,
            sliding_valid_length=sliding_valid_length,
        )
    draft_valid_length = max(int(position_index) - 1, 0)
    result = {
        "inputs_embeds": inputs_embeds,
        # Both supported graph ABIs use N-1 as their *attention length input*,
        # not as the assistant token's RoPE position:
        # * legacy/non-FA MaskedSoftmax exposes valid_length + 1 keys for q=1;
        # * read-only Flash/PageAttention treats q as position N-1 and receives
        #   current_length=1, so its visible target-KV length is N.
        # RoPE independently uses valid_length + current_length = N, matching
        # the latest sampled token which is not yet in target KV. The assistant
        # never appends draft KV, therefore N stays constant for every proposal
        # in one speculative round.
        "past_seq_length": torch.tensor([draft_valid_length], dtype=torch.int32, device=device),
        "current_length": torch.tensor([1], dtype=torch.int32, device=device),
    }
    if sliding_mask is not None:
        result["sliding_attention_mask"] = sliding_mask
    for name, value in shared.items():
        if name in input_names:
            result[name] = value
    compact_names = {"kv_window_start_abs", "kv_valid_length"}
    requested_compact_names = compact_names.intersection(input_names)
    if requested_compact_names:
        if requested_compact_names != compact_names:
            raise RuntimeError("Gemma4 MTP FlashAttention requires both compact cache inputs")
        model_config = target_model.meta_info.model_config
        sliding_window = int(
            getattr(
                target_model,
                "sliding_window",
                getattr(model_config, "sliding_window", 0),
            )
            or 0
        )
        start, valid = resolve_readonly_sliding_cache_range(
            int(position_index),
            sliding_window,
        )
        result["kv_window_start_abs"] = torch.tensor(
            [start],
            dtype=torch.int32,
            device=device,
        )
        result["kv_valid_length"] = torch.tensor(
            [valid],
            dtype=torch.int32,
            device=device,
        )
    return result


def generate_with_mtp(
    target_model,
    assistant_session,
    tokenizer,
    meta: dict[str, Any],
    prompt: str,
    max_new_tokens: int,
    num_draft_tokens: int,
    *,
    trace_accepted_count: bool = False,
    trace_kv_input_info: dict[str, Any] | None = None,
):
    import torch

    input_ids = _load_prompt_inputs(tokenizer, prompt).to(target_model.device)
    output_ids = input_ids[0].tolist()
    prompt_length = int(input_ids.shape[1])
    context_length = int(meta.get("model_config", {}).get("context_max_length", 0))
    verify_capacity = num_draft_tokens + 1
    if max_new_tokens <= 0 or num_draft_tokens <= 0:
        raise ValueError("max_new_tokens and num_draft_tokens must be positive")
    if context_length and prompt_length + max_new_tokens + verify_capacity > context_length:
        raise ValueError("Prompt, generation budget and verify block exceed exported context capacity")
    eos_token_ids = _resolve_eos_token_ids(tokenizer, *_generation_config_candidates(meta))
    spec = _meta_spec_decode(meta)
    verify_length = int(spec.get("verify_length") or meta.get("spec_decode_verify_length") or (num_draft_tokens + 1))

    logits, last_hidden, shared, past_seq_length = _run_target_prefill(target_model, input_ids)
    current_token_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
    output_ids.append(current_token_id)
    generated = 1
    stats = {
        "preset_prompt_tokens": prompt_length,
        "verify_rounds": 0,
        "draft_proposed": 0,
        "draft_accepted": 0,
        "committed_tokens": 1,
        "accepted_per_round": [],
        "accepted_count_inputs": [],
    }
    if current_token_id in eos_token_ids or generated >= max_new_tokens:
        text = tokenizer.decode(output_ids[prompt_length:], skip_special_tokens=True).strip()
        stats["generated_tokens"] = int(len(output_ids) - prompt_length)
        stats["acceptance_rate"] = 0.0
        return text, output_ids, stats

    while generated < max_new_tokens:
        draft_tokens: list[int] = []
        assistant_hidden = last_hidden
        assistant_last_token_id = int(current_token_id)
        # Gemma4's Q-only assistant shares target KV and does not append draft
        # K/V. Match the official proposer: every draft in this round uses the
        # same last-target rotary position and target-cache visibility.
        base_position = max(int(past_seq_length), 0)
        _, sliding_valid_length = _hmonnx_shared_cache_valid_lengths(target_model, past_seq_length)
        for _ in range(num_draft_tokens):
            round_inputs = _build_assistant_inputs(
                target_model=target_model,
                assistant_session=assistant_session,
                last_token_id=assistant_last_token_id,
                current_hidden=assistant_hidden,
                shared=shared,
                position_index=base_position,
                sliding_valid_length=sliding_valid_length,
            )
            draft_logits, assistant_hidden = assistant_session(**round_inputs)
            assistant_token = int(torch.argmax(draft_logits[:, -1, :], dim=-1).item())
            draft_tokens.append(assistant_token)
            assistant_last_token_id = assistant_token
            if assistant_token in eos_token_ids:
                break

        if not draft_tokens:
            break

        stats["verify_rounds"] += 1
        stats["draft_proposed"] += len(draft_tokens)
        verify_ids = [int(current_token_id)] + draft_tokens
        if len(verify_ids) > verify_length:
            raise RuntimeError(f"verify_ids length {len(verify_ids)} exceeds exported verify_length={verify_length}")
        if len(verify_ids) < verify_length:
            verify_ids = verify_ids + [verify_ids[-1]] * (verify_length - len(verify_ids))
        round_past_seq_length = int(past_seq_length)
        verify_input_accepted_count = int(getattr(target_model, "_mtp_next_verify_accepted_count", 0))
        target_model._mtp_trace_accepted_count = bool(trace_accepted_count)
        target_model._mtp_verify_round_index = int(stats["verify_rounds"])
        cache_snapshot = (
            _snapshot_hmonnx_kv_cache(target_model) if _target_needs_manual_cache_commit(target_model) else None
        )
        verify_logits, verify_hidden, verify_shared = _run_target_verify(
            target_model, verify_ids, round_past_seq_length
        )
        predicted_tokens = [
            int(torch.argmax(verify_logits[:, idx : idx + 1, :], dim=-1)[0, 0].item()) for idx in range(len(verify_ids))
        ]

        accepted_count = 0
        for predicted_token, draft_token in zip(
            predicted_tokens,
            draft_tokens,
            strict=False,
        ):
            if int(predicted_token) != int(draft_token):
                break
            accepted_count += 1
        target_model._mtp_next_verify_accepted_count = int(accepted_count)
        stats["accepted_count_inputs"].append(verify_input_accepted_count)
        stats["draft_accepted"] += accepted_count
        stats["accepted_per_round"].append(accepted_count)
        if trace_accepted_count:
            kv_desc = trace_kv_input_info or {}
            previous_accepted_count = stats["accepted_per_round"][-2] if len(stats["accepted_per_round"]) > 1 else 0
            print(
                "[MTP accepted_count trace] "
                f"verify_round={int(stats['verify_rounds'])} "
                f"first_sliding_kvcache={kv_desc.get('node_name')} "
                f"accepted_count_input_index={kv_desc.get('accepted_count_input_index')} "
                f"input_accepted_count={verify_input_accepted_count} "
                f"current_round_accepted_tokens={accepted_count} "
                f"next_verify_expected_accepted_count={accepted_count} "
                f"match_previous={verify_input_accepted_count == previous_accepted_count}"
            )

        for token in draft_tokens[:accepted_count]:
            if generated >= max_new_tokens:
                break
            output_ids.append(int(token))
            generated += 1
            stats["committed_tokens"] += 1
            if int(token) in eos_token_ids:
                break
        if output_ids[-1] in eos_token_ids or generated >= max_new_tokens:
            break

        if accepted_count < len(draft_tokens):
            next_token_id = int(predicted_tokens[accepted_count])
        else:
            next_token_id = int(predicted_tokens[len(draft_tokens)])

        output_ids.append(next_token_id)
        generated += 1
        stats["committed_tokens"] += 1
        accepted_steps = accepted_count + 1
        past_seq_length += accepted_steps
        if hasattr(target_model, "commit_cache"):
            shared = target_model.commit_cache(past_seq_length)
        elif cache_snapshot is not None and accepted_steps < verify_length:
            # HMONNX LLMCache mutates KV in-place for the entire exported
            # verify block.  Speculative decoding may accept only a prefix, so
            # rebuild the compact cache from the pre-verify snapshot plus the
            # accepted prefix of the verified cache.  This is equivalent to a
            # fixed-length replay with current_length=accepted_steps, but does
            # not spend another target decode pass.
            shared = _commit_hmonnx_verified_cache(
                target_model,
                cache_snapshot,
                past_seq_length=round_past_seq_length,
                commit_length=accepted_steps,
                verify_length=verify_length,
            )
        else:
            shared = verify_shared
        if verify_hidden is not None and getattr(verify_hidden, "ndim", 0) >= 3:
            last_hidden = verify_hidden[:, accepted_count : accepted_count + 1, :]
        current_token_id = next_token_id
        if current_token_id in eos_token_ids:
            break

    text = tokenizer.decode(output_ids[prompt_length:], skip_special_tokens=True).strip()
    stats["generated_tokens"] = int(len(output_ids) - prompt_length)
    stats["acceptance_rate"] = (stats["draft_accepted"] / stats["draft_proposed"]) if stats["draft_proposed"] else 0.0
    return text, output_ids, stats


def run_mtp_generation(meta_file, *, prompt, max_new_tokens, device):
    """Load a self-contained artifact and run greedy draft/verify decoding."""
    import gc
    import time
    from contextlib import nullcontext
    import torch
    from xhmodel_merak._environment import scoped_environment
    from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager

    meta_file = Path(meta_file).resolve()
    meta = json.loads(meta_file.read_text())
    meta["_meta_path"] = str(meta_file)
    spec = meta["spec_decode"]
    device = torch.device(device)
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    target = assistant = None
    with context, scoped_environment({"ENABLE_HMINFERENCE_V2": "1", "XHQUANT_PACKED_W4": "1"}):
        try:
            target = AutoLLMHONNXModel.from_pretrained(str(meta_file), device_map=[device])
            target.to(device)
            tokenizer = target.get_tokenizer(trust_remote_code=True)
            assistant = AssistantDraftSession(meta_file.parent / spec["standalone_draft_decode_onnx"], device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with LLMInferenceContextManager(target, devices=[device]), torch.inference_mode():
                text, ids, stats = generate_with_mtp(
                    target, assistant, tokenizer, meta, prompt, max_new_tokens, spec["num_draft_tokens"]
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            count = stats["generated_tokens"]
            rounds = stats["verify_rounds"]
            stats.update(
                prompt_tokens=stats["preset_prompt_tokens"],
                num_draft_tokens=spec["num_draft_tokens"],
                average_accepted_tokens=stats["draft_accepted"] / rounds if rounds else 0.0,
                output_token_ids=ids[stats["preset_prompt_tokens"] :],
                backend="HMONNXInferenceV2",
            )
            return text, count, elapsed, count / elapsed if elapsed else 0.0, stats
        finally:
            del assistant, target
            gc.collect()
