"""Single-request greedy text generation for Qwen4 serving packages."""

from __future__ import annotations

import json
import time
from collections import ChainMap
from contextlib import nullcontext
from pathlib import Path

import torch

from ...hmonnx import AutoLLMHONNXModel
from .cache_contract import QWEN4_EXP_LLM_CACHE_ABI


_BLOCK_SIZE = 64


def _snapshot_mtp_state(state, page_caches, past_length, current_length):
    """Save recurrent state and only the cache rows a verify step can write."""
    snapshots = []
    pool_start = past_length // 4
    pool_count = (past_length % 4 + current_length) // 4
    for spec in state.contract.tensors:
        tensor = state.tensors.get(spec.name)
        if tensor is None:
            continue
        if spec.state_kind in {"kv_key", "kv_value"}:
            target = tensor.narrow(2, past_length, current_length)
        elif spec.state_kind == "qsa_indexer_key":
            target = tensor.narrow(2, pool_start, pool_count)
        else:
            target = tensor
        snapshots.append((spec.name, tensor, target, target.detach().clone()))
    first_page = past_length // _BLOCK_SIZE
    page_count = (past_length + current_length - 1) // _BLOCK_SIZE - first_page + 1
    pages = []
    for cache in page_caches:
        target = cache.narrow(1, first_page, page_count)
        pages.append((target, target.detach().clone()))
    return snapshots, pages, state.group_start_position.detach().clone()


def _restore_mtp_state(state, snapshot):
    snapshots, pages, group_start_position = snapshot
    for name, tensor, target, saved in snapshots:
        target.copy_(saved)
        state.tensors[name] = tensor
    for target, saved in pages:
        target.copy_(saved)
    state.group_start_position.copy_(group_start_position)


def _allocate_page_caches(model):
    """Allocate floating-point storage required by the PageKVCache graph ABI."""
    if getattr(model, "qwen4_cache_abi", None) == QWEN4_EXP_LLM_CACHE_ABI:
        return []
    prefill = model._get_page_attention_modules(model.prefill_model)
    decode = model._get_page_attention_modules(model.decode_model)

    def signature(modules):
        return [(m.num_kv_heads, m.k_bits, m.v_bits) for m in modules]

    if signature(prefill) != signature(decode):
        raise ValueError("Qwen4 prefill/decode PageKV layouts differ")
    if getattr(model, "verify_model", None) is not None:
        verify = model._get_page_attention_modules(model.verify_model)
        if signature(prefill) != signature(verify):
            raise ValueError("Qwen4 prefill/verify PageKV layouts differ")
    if len(prefill) != model.kvcache_config.num_layers:
        raise ValueError("Qwen4 PageKV writer count differs from runtime metadata")
    contract = model.qwen4_cache_contract
    blocks = (model.meta_info.model_config.context_max_length + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    caches = [
        torch.zeros(
            (2, blocks, module.num_kv_heads, _BLOCK_SIZE, contract.head_dim),
            dtype=model.dtype,
            device=model.device,
        )
        for module in prefill
    ]
    return caches


def _page_step(
    model, caches, input_ids, past_length, *, inputs_embeds=None, position_ids=None, capture=None, return_all=False
):
    valid = int(input_ids.shape[1])
    graph_length = model.get_input_sequence_length()
    total = past_length + valid
    # Padding must never overwrite a real cache slot.
    slots = torch.full((graph_length,), -1, dtype=torch.int64)
    slots[:valid] = torch.arange(past_length, total)
    blocks = torch.arange((total + _BLOCK_SIZE - 1) // _BLOCK_SIZE, dtype=torch.int64)
    if getattr(model, "qwen4_cache_abi", None) != QWEN4_EXP_LLM_CACHE_ABI:
        model.set_page_attention_context(
            caches,
            blocks,
            slots,
            _BLOCK_SIZE,
            effective_kv_length=total,
        )
    data = {"past_seq_length": past_length, "current_input_length": valid}
    if inputs_embeds is None:
        data["input_ids"] = input_ids
    else:
        data.update(inputs_embeds=inputs_embeds, ple_input_ids=input_ids)
    if position_ids is not None:
        data["position_ids"] = position_ids
    processor = model.get_data_preprocessor()
    inputs = processor(data)
    graph = None
    component = None
    graph_name = None
    if capture is not None:
        graph = (
            model.active_graph()
            if hasattr(model, "active_graph")
            else (model.prefill_model if model.is_prefill() else model.decode_model)
        )
        component = "prefill" if model.is_prefill() else "decode"
        if getattr(model, "_verify_active", False):
            component, graph_name = "mtp", model.active_component()
    context = (
        capture.step(
            component,
            graph.hmonnx_session,
            graph_name=graph_name,
            runtime_state=ChainMap(
                {
                    **(
                        {"group_start_position": model._qwen4_state.group_start_position}
                        if hasattr(getattr(model, "_qwen4_state", None), "group_start_position")
                        else {}
                    ),
                    **(
                        {
                            **{f"page_kv_{index}": cache for index, cache in enumerate(caches)},
                            "block_ids": blocks,
                            "slot_mapping": slots,
                        }
                        if caches
                        else {}
                    ),
                },
                getattr(getattr(model, "_qwen4_state", None), "tensors", {}),
            ),
            ple_indices=getattr(getattr(processor, "external_ple", None), "last_indices", None),
        )
        if capture is not None
        else nullcontext()
    )
    with context:
        logits = model.forward(*inputs)
    # The exported head either retains the last valid token or all tokens.
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] not in (1, graph_length):
        raise ValueError(f"Unsupported Qwen4 logits shape: {tuple(logits.shape)}")
    if return_all:
        if logits.shape[1] != graph_length:
            raise ValueError("MTP target must retain logits for every verify position")
        return logits[:, :valid]
    return logits[:, 0 if logits.shape[1] == 1 else valid - 1, :]


def resolve_eos_token_ids(model, eos_token_id=None, *, tokenizer=None):
    """Prefer caller EOS, then the packaged HF generation config and tokenizer."""
    eos = eos_token_id
    if eos is None:
        generation_path = Path(model.hf_model_dir) / "generation_config.json"
        if generation_path.is_file():
            eos = json.loads(generation_path.read_text()).get("eos_token_id")
        if eos is None:
            if tokenizer is None:
                tokenizer = model.get_tokenizer()
            eos = tokenizer.eos_token_id
    return [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))


def create_mtp_runtime(model, *, meta_file=None, device_map=None, capture=None):
    """Open the package's draft graphs with the target's execution devices."""
    spec = getattr(model.meta_info, "spec_decode", None)
    if spec is None or spec.get("mode") != "mtp":
        return None
    from .mtp_runtime import MTPGraphRuntime

    version = int(spec.get("runtime_contract_version", 2))
    if version >= 3 and capture is not None and "mtp" in capture.components:
        from .mtp_contract import mtp_verify_phase

        verify_phase = spec.get("target_verify_graph", mtp_verify_phase(
            model._qwen4_verify_length, runtime_version=version,
        ))
        capture.required_graphs.update({"mtp_prefill", "mtp_decode", verify_phase})
    existing = getattr(model, "_qwen4_mtp_runtime", None)
    if existing is not None and getattr(existing, "session", True) is not None:
        return existing
    root = Path(meta_file).parent if meta_file is not None else Path(model.meta_info.prefill_hmonnx).parent.parent
    draft = spec.draft
    options = {"device_map": device_map or getattr(model, "_valid_devices", None)}
    if version >= 3:
        options.update(
            prefill_path=root / draft.prefill_hmonnx,
            prefill_graph=model._qwen4_mtp_prefill_graph,
            prefill_chunk_length=int(draft.prefill_chunk_length),
        )
    runtime = MTPGraphRuntime(
        root / draft.decode_hmonnx,
        cache_contract=dict(draft.cache_contract),
        cache_abi=draft.get("cache_abi"),
        onnx_graph=model._qwen4_mtp_graph,
        context_length=int(model.meta_info.model_config.context_max_length),
        device=model.device,
        **options,
    )
    model._qwen4_mtp_runtime = runtime
    return runtime


def _mtp_candidate_count(model):
    spec = getattr(model.meta_info, "spec_decode", None)
    if spec is None:
        return 1  # Legacy callers without a speculative metadata section.
    value = spec.get("num_draft_tokens", 1 if spec.get("runtime_contract_version", 2) <= 3 else 4)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("MTP num_draft_tokens must be a positive integer")
    return value


@torch.inference_mode()
def generate_token_ids(
    model,
    input_ids,
    *,
    max_new_tokens,
    eos_token_ids=(),
    inputs_embeds=None,
    position_ids=None,
    capture=None,
    mtp_runtime=None,
):
    """Run one isolated request, retaining GDR/QSA/PLE state across chunks."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Qwen4 generation requires a nonempty batch-one token sequence")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    config = model.meta_info.model_config
    prompt_length = int(input_ids.shape[1])
    if prompt_length + max_new_tokens > config.context_max_length:
        raise ValueError("Prompt plus max_new_tokens exceeds exported context capacity")
    if position_ids is not None and tuple(position_ids.shape) != (3, 1, prompt_length):
        raise ValueError("Qwen4 prompt positions must have shape [3, 1, prompt_length]")
    chunk = int(config.prefill_chunk_length)
    if mtp_runtime is not None:
        count = _mtp_candidate_count(model)
        verify_width = getattr(model, "_qwen4_verify_length", chunk)
        if verify_width < count + 1:
            raise ValueError("MTP verify length must cover pending token plus every draft candidate")
    model.reset_internal_state()
    caches = _allocate_page_caches(model)
    model.set_prefill()
    past = 0
    previous_hidden = None
    model.last_mtp_stats = None
    if mtp_runtime is not None:
        mtp_runtime.reset()
    while past < prompt_length:
        # Internal QSA caches have the exported context capacity, so even
        # masked padding positions must fit their fixed-width scatter inputs.
        # Consume the final prompt tokens with decode when a full prefill
        # graph would extend beyond that capacity.
        if past + chunk > config.context_max_length:
            if model.is_prefill():
                model.set_decode()
            width = 1
        else:
            width = min(chunk, prompt_length - past)
        logits = _page_step(
            model,
            caches,
            input_ids[:, past : past + width],
            past,
            inputs_embeds=None if inputs_embeds is None else inputs_embeds[:, past : past + width],
            position_ids=None if position_ids is None else position_ids[:, :, past : past + width],
            capture=capture,
        )
        if mtp_runtime is not None:
            current_hidden = model.last_hidden_states[:, :width]
            chunk_embeddings = (
                model.embed_tokens(input_ids[:, past : past + width])
                if inputs_embeds is None
                else inputs_embeds[:, past : past + width]
            )
            chunk_positions = (
                torch.arange(past, past + width, device=model.device).view(1, 1, -1).expand(3, 1, -1)
                if position_ids is None
                else position_ids[:, :, past : past + width]
            )
            embeddings, hidden, positions = [], [], []
            if previous_hidden is not None:
                embeddings.append(chunk_embeddings[:, :1])
                hidden.append(previous_hidden)
                positions.append(chunk_positions[:, :, :1])
            if width > 1:
                embeddings.append(chunk_embeddings[:, 1:])
                hidden.append(current_hidden[:, :-1])
                positions.append(chunk_positions[:, :, 1:])
            if embeddings:
                mtp_runtime.prefill(
                    torch.cat(embeddings, 1),
                    torch.cat(hidden, 1),
                    position_ids=torch.cat(positions, 2),
                    capture=capture,
                )
            previous_hidden = current_hidden[:, -1:].clone()
        past += width
    generated = []
    eos = set(eos_token_ids)
    decode_position = prompt_length if position_ids is None else int(position_ids.max()) + 1
    if mtp_runtime is not None:
        return _generate_mtp(
            model,
            caches,
            input_ids,
            previous_hidden,
            logits,
            mtp_runtime,
            max_new_tokens,
            eos,
            decode_position,
            capture,
        )

    for index in range(max_new_tokens):
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        if token in eos or index + 1 == max_new_tokens:
            break
        if model.is_prefill():
            model.set_decode()
        logits = _page_step(
            model,
            caches,
            torch.tensor([[token]], dtype=torch.long, device=model.device),
            prompt_length + index,
            position_ids=torch.full((3, 1, 1), decode_position + index, device=model.device),
            capture=capture,
        )
    return generated


def _generate_mtp(
    model, caches, input_ids, target_hidden, logits, draft, max_new_tokens, eos, decode_position, capture
):
    """Greedy N-candidate speculation with complete target/draft transactions."""
    from .mtp_runtime import verify_draft_transaction
    from .mtp_snapshot import snapshot_state

    prompt_length = int(input_ids.shape[1])
    capacity = _mtp_candidate_count(model)
    pending = int(logits.argmax(-1).item())
    generated = [pending]
    past = prompt_length
    decode_executed = False
    stats = dict(
        num_rounds=0,
        draft_tokens_total=0,
        accepted_drafts_total=0,
        rejected_drafts_total=0,
        avg_accepted_per_round=0.0,
        acceptance_rate=0.0,
        draft_capacity_per_round=capacity,
        accepted_drafts_per_round=[],
        output_tokens=1,
        target_verify_calls=0,
        target_replay_tokens=0,
    )
    model.last_mtp_stats = stats

    def positions(count, absolute_past):
        start = decode_position + absolute_past - prompt_length
        return torch.arange(start, start + count, device=model.device).view(1, 1, -1).expand(3, 1, -1)

    def target_snapshot(width):
        contract = getattr(model, "qwen4_cache_contract", None)
        specs = {} if contract is None else {spec.name: spec for spec in contract.tensors}
        return (
            snapshot_state(model._qwen4_state.tensors, caches, specs=specs, past_length=past, write_length=int(width)),
            model._qwen4_state.group_start_position.clone(),
        )

    def restore_target(saved):
        tensors, group_start = saved
        tensors.restore(model._qwen4_state.tensors, caches)
        model._qwen4_state.group_start_position.copy_(group_start)
        model._data_processor = None

    while len(generated) < max_new_tokens and generated[-1] not in eos:
        model.set_decode()
        remaining = max_new_tokens - len(generated)
        dedicated_verify = getattr(model, "verify_model", None) is not None
        verify_width = (
            model._qwen4_verify_length if dedicated_verify else model.meta_info.model_config.prefill_chunk_length
        )
        can_speculate = (
            verify_width >= capacity + 1 and past + verify_width <= model.meta_info.model_config.context_max_length
        )
        needs_decode = capture is not None and "decode" in capture.components and not decode_executed
        if remaining == 1 or not can_speculate or needs_decode:
            single = torch.tensor([[pending]], device=model.device)
            next_logits = _page_step(model, caches, single, past, position_ids=positions(1, past), capture=capture)
            decode_executed = True
            draft.step(model.embed_tokens(single), target_hidden, position_ids=positions(1, past), capture=capture)
            target_hidden = model.last_hidden_states[:, :1].clone()
            past += 1
            pending = int(next_logits.argmax(-1).item())
            generated.append(pending)
            continue

        candidate_count = min(capacity, remaining - 1)
        saved_target = target_snapshot(verify_width)
        # Full acceptance consumes one final shifted pair after the K proposals.
        saved_draft = draft.snapshot(write_length=candidate_count + 1)
        round_hidden = target_hidden
        try:
            candidates = []
            candidate_input = pending
            draft_hidden = target_hidden
            for index in range(candidate_count):
                proposal_logits, draft_hidden = draft.step(
                    model.embed_tokens(torch.tensor([[candidate_input]], device=model.device)),
                    draft_hidden,
                    position_ids=positions(1, past + index),
                    capture=capture,
                )
                candidate_input = int(proposal_logits[:, -1].argmax(-1).item())
                candidates.append(candidate_input)
                if candidate_input in eos:
                    break

            committed_hidden = None

            def verify(tokens, past=past, dedicated_verify=dedicated_verify):
                nonlocal committed_hidden
                model.set_verify() if dedicated_verify else model.set_prefill()
                values = _page_step(
                    model,
                    caches,
                    torch.tensor([tokens], device=model.device),
                    past,
                    position_ids=positions(len(tokens), past),
                    capture=capture,
                    return_all=True,
                )
                committed_hidden = model.last_hidden_states[:, : len(tokens)].clone()
                return values, committed_hidden

            def replay(tokens, past=past):
                nonlocal committed_hidden
                model.set_decode()
                values, hidden = [], []
                for index, token in enumerate(tokens):
                    value = _page_step(
                        model,
                        caches,
                        torch.tensor([[token]], device=model.device),
                        past + index,
                        position_ids=positions(1, past + index),
                        capture=capture,
                    )
                    values.append(value.unsqueeze(1))
                    hidden.append(model.last_hidden_states[:, :1].clone())
                committed_hidden = torch.cat(hidden, 1)
                stats["target_replay_tokens"] += len(tokens)
                return torch.cat(values, 1), committed_hidden

            result = verify_draft_transaction(
                pending_token=pending,
                draft_tokens=candidates,
                snapshot=lambda saved=saved_target: saved,
                restore=restore_target,
                verify=verify,
                replay=replay,
            )
            accepted = result.accepted_draft_tokens
            if accepted < len(candidates):
                draft.restore(saved_draft)
                # Discard every speculative draft suffix, including recurrent
                # state, then replay shifted pairs with the actual target HC.
                for index, token in enumerate(result.committed_input_tokens):
                    hidden = round_hidden if index == 0 else committed_hidden[:, index - 1 : index]
                    draft.step(
                        model.embed_tokens(torch.tensor([[token]], device=model.device)),
                        hidden,
                        position_ids=positions(1, past + index),
                        capture=capture,
                    )
            else:
                # N proposals consume pending through candidate[N-2]. Complete
                # the final accepted pair so draft stays one input behind target.
                draft.step(
                    model.embed_tokens(torch.tensor([[candidates[-1]]], device=model.device)),
                    committed_hidden[:, len(candidates) - 1 : len(candidates)],
                    position_ids=positions(1, past + len(candidates)),
                    capture=capture,
                )
        except BaseException:
            restore_target(saved_target)
            draft.restore(saved_draft)
            raise

        stats["num_rounds"] += 1
        stats["target_verify_calls"] += 1
        stats["draft_tokens_total"] += len(candidates)
        stats["accepted_drafts_total"] += accepted
        stats["rejected_drafts_total"] += len(candidates) - accepted
        stats["accepted_drafts_per_round"].append(accepted)
        stats["avg_accepted_per_round"] = stats["accepted_drafts_total"] / stats["num_rounds"]
        stats["acceptance_rate"] = stats["accepted_drafts_total"] / stats["draft_tokens_total"]
        generated.extend(candidates[:accepted])
        past += len(result.committed_input_tokens)
        target_hidden = result.target_hidden
        pending = result.next_token
        if generated[-1] in eos or len(generated) >= max_new_tokens:
            break
        generated.append(pending)
    stats["output_tokens"] = len(generated)
    return generated


def _run_text_generation(
    meta_file,
    *,
    prompt,
    max_new_tokens,
    device,
    image=None,
    video=None,
    golden_components=(),
    golden_dir=None,
    device_map=None,
):
    """Return text, token count, latency, throughput and request statistics."""
    model = None
    started = time.perf_counter()
    try:
        load_options = {"enable_auto_offload": True} if torch.device(device).type == "cuda" else {}
        model = AutoLLMHONNXModel.from_pretrained(
            str(meta_file),
            enable_page_attention=True,
            enable_cuda_graph=False,
            **load_options,
            device_map=device if device_map is None else device_map,
        )
        model.to(device=device)
        capture = None
        if golden_components:
            from .golden_runtime import GoldenCapture

            capture = GoldenCapture(golden_dir or Path(meta_file).parent / "golden", golden_components)
            capture.add_ple_resources(getattr(model, "external_ple", {}))
        tokenizer = model.get_tokenizer()
        if image is not None or video is not None:
            from .multimodal import prepare_media_prompt

            prepared = prepare_media_prompt(model.get_processor(), prompt, image=image, video=video)
        elif getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prepared = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        else:
            prepared = tokenizer(prompt, return_tensors="pt")
        input_ids = prepared["input_ids"]
        # Processor metadata describes sampling and is consumed while preparing
        # timestamp tokens; model.generate receives tensor media inputs only.
        prepared = {
            name: value.to(model.device) if isinstance(value, torch.Tensor) else value
            for name, value in prepared.items()
            if name not in {"video_metadata"}
        }
        generated = model.generate(**prepared, max_new_tokens=max_new_tokens, do_sample=False, _capture=capture)
        output = generated[0, input_ids.shape[1] :].tolist()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        if capture is not None and "visual" in capture.components:
            if model.vision_encoder is None:
                raise ValueError("visual Golden requires a visual encoder")
            model.vision_encoder.capture_all(capture)
        artifacts = capture.finish() if capture is not None else {}
        elapsed = time.perf_counter() - started
        return (
            tokenizer.decode(output, skip_special_tokens=True),
            len(output),
            elapsed,
            len(output) / elapsed if elapsed else 0.0,
            {
                "input_tokens": int(input_ids.shape[1]),
                "decoding": "greedy",
                "page_attention": bool(model.enable_page_attention),
                **({"mtp": model.last_mtp_stats} if getattr(model, "last_mtp_stats", None) is not None else {}),
                **({"golden_artifacts": artifacts} if artifacts else {}),
            },
        )
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()
        del model


@torch.inference_mode()
def run_text_generation(
    meta_file,
    *,
    prompt,
    max_new_tokens,
    device,
    image=None,
    video=None,
    golden_components=(),
    golden_dir=None,
    device_map=None,
):
    def execution_device(value):
        resolved = torch.device(value)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        return str(resolved)

    device = execution_device(device)
    if device_map is not None:
        if not isinstance(device_map, (list, tuple)) or not device_map:
            raise ValueError("device_map must be a nonempty sequence of execution devices")
        device_map = [execution_device(value) for value in device_map]
        if device not in device_map:
            raise ValueError("The primary device must be included in device_map")
        if len(device_map) > 1 and any(torch.device(value).type != "cuda" for value in device_map):
            raise ValueError("Multiple execution devices must all be CUDA devices")
    context = torch.cuda.device(device) if torch.device(device).type == "cuda" else nullcontext()
    with context:
        return _run_text_generation(
            meta_file,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device=device,
            image=image,
            video=video,
            golden_components=golden_components,
            golden_dir=golden_dir,
            device_map=device_map,
        )
