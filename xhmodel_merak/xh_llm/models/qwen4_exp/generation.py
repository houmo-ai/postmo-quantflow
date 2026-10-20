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
        graph = model.active_graph() if hasattr(model, "active_graph") else (
            model.prefill_model if model.is_prefill() else model.decode_model
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
                    **({**{f"page_kv_{index}": cache for index, cache in enumerate(caches)},
                        "block_ids": blocks, "slot_mapping": slots} if caches else {}),
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

    root = Path(meta_file).parent if meta_file is not None else Path(model.meta_info.prefill_hmonnx).parent.parent
    draft = spec.draft
    options = {"device_map": device_map or getattr(model, "_valid_devices", None)}
    if int(spec.get("runtime_contract_version", 2)) >= 3:
        options.update(
            prefill_path=root / draft.prefill_hmonnx,
            prefill_graph=model._qwen4_mtp_prefill_graph,
            prefill_chunk_length=int(draft.prefill_chunk_length),
        )
        if capture is not None and "mtp" in capture.components:
            from .mtp_contract import mtp_verify_phase

            verify_phase = spec.get("target_verify_graph", mtp_verify_phase(model._qwen4_verify_length))
            capture.required_graphs.update({"mtp_prefill", "mtp_decode", verify_phase})
    return MTPGraphRuntime(
        root / draft.decode_hmonnx,
        cache_contract=dict(draft.cache_contract),
        cache_abi=draft.get("cache_abi"),
        onnx_graph=model._qwen4_mtp_graph,
        context_length=int(model.meta_info.model_config.context_max_length),
        device=model.device,
        **options,
    )


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
    model.reset_internal_state()
    caches = _allocate_page_caches(model)
    model.set_prefill()
    chunk = int(config.prefill_chunk_length)
    past = 0
    prompt_hidden = []
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
            prompt_hidden.append(model.last_hidden_states[:, :width].clone())
        past += width
    generated = []
    eos = set(eos_token_ids)
    decode_position = prompt_length if position_ids is None else int(position_ids.max()) + 1
    if mtp_runtime is not None:
        return _generate_mtp(
            model,
            caches,
            input_ids,
            prompt_hidden,
            logits,
            mtp_runtime,
            max_new_tokens,
            eos,
            decode_position,
            capture,
            prompt_embeds=inputs_embeds,
            prompt_position_ids=(position_ids if position_ids is not None else
                                 torch.arange(prompt_length, device=model.device).view(1, 1, -1).expand(3, 1, -1)),
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
    model,
    caches,
    input_ids,
    prompt_hidden,
    logits,
    draft,
    max_new_tokens,
    eos,
    decode_position,
    capture,
    *,
    prompt_embeds=None,
    prompt_position_ids,
):
    """Greedy one-token speculation with complete recurrent-state rollback."""
    from .mtp_runtime import verify_draft_transaction
    from .mtp_snapshot import snapshot_state

    prompt_length = int(input_ids.shape[1])
    hidden = torch.cat(prompt_hidden, dim=1)
    draft.reset()
    # The assistant predicts x[t+2] from (target HC[t], embedding x[t+1]).
    if hasattr(draft, "prefill"):
        draft.prefill(
            model.embed_tokens(input_ids[:, 1:]) if prompt_embeds is None else prompt_embeds[:, 1:],
            hidden[:, :-1],
            position_ids=prompt_position_ids[:, :, 1:prompt_length],
            capture=capture,
        )
    else:
        for index in range(prompt_length - 1):
            position = index + 1
            draft.step(
                model.embed_tokens(input_ids[:, position:position + 1])
                if prompt_embeds is None else prompt_embeds[:, position:position + 1],
                hidden[:, index:index + 1],
                position_ids=prompt_position_ids[:, :, position:position + 1],
            )
    target_hidden = hidden[:, -1:]
    pending = int(logits.argmax(-1).item())
    generated = [pending]
    past = prompt_length
    decode_executed = False

    def positions(count, absolute_past):
        start = decode_position + absolute_past - prompt_length
        return torch.arange(start, start + count, device=model.device).view(1, 1, -1).expand(3, 1, -1)

    while len(generated) < max_new_tokens and generated[-1] not in eos:
        model.set_decode()
        remaining = max_new_tokens - len(generated)
        dedicated_verify = getattr(model, "verify_model", None) is not None
        verify_width = (model._qwen4_verify_length if dedicated_verify
                        else model.meta_info.model_config.prefill_chunk_length)
        can_speculate = verify_width >= 2 and past + verify_width <= model.meta_info.model_config.context_max_length
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
        proposal_logits, _ = draft.step(
            model.embed_tokens(torch.tensor([[pending]], device=model.device)),
            target_hidden,
            position_ids=positions(1, past),
            capture=capture,
        )
        candidate = int(proposal_logits[:, -1].argmax(-1).item())

        def snapshot(past=past, verify_width=verify_width):
            contract = getattr(model, "qwen4_cache_contract", None)
            specs = {} if contract is None else {spec.name: spec for spec in contract.tensors}
            return (snapshot_state(
                model._qwen4_state.tensors, caches, specs=specs,
                past_length=past, write_length=int(verify_width),
            ), model._qwen4_state.group_start_position.clone())

        def restore(saved):
            tensors, group_start_position = saved
            tensors.restore(model._qwen4_state.tensors, caches)
            model._qwen4_state.group_start_position.copy_(group_start_position)
            model._data_processor = None

        def verify(tokens, past=past, dedicated_verify=dedicated_verify):
            model.set_verify() if dedicated_verify else model.set_prefill()
            ids = torch.tensor([tokens], device=model.device)
            values = _page_step(
                model, caches, ids, past, position_ids=positions(len(tokens), past), capture=capture, return_all=True
            )
            return values, model.last_hidden_states[:, : len(tokens)].clone()

        def replay(tokens, past=past):
            model.set_decode()
            values, states = [], []
            for index, token in enumerate(tokens):
                ids = torch.tensor([[token]], device=model.device)
                value = _page_step(
                    model, caches, ids, past + index, position_ids=positions(1, past + index), capture=capture
                )
                values.append(value.unsqueeze(1))
                states.append(model.last_hidden_states[:, :1].clone())
            return torch.cat(values, 1), torch.cat(states, 1)

        result = verify_draft_transaction(
            pending_token=pending,
            draft_tokens=[candidate],
            snapshot=snapshot,
            restore=restore,
            verify=verify,
            replay=replay,
        )
        if result.accepted_draft_tokens:
            generated.append(candidate)
            if candidate in eos or len(generated) == max_new_tokens:
                break
            # Advance the assistant over the accepted candidate, using the
            # target HC state after the pending token (not after candidate).
            pending_hc = model.last_hidden_states[:, :1].clone()
            draft.step(
                model.embed_tokens(torch.tensor([[candidate]], device=model.device)),
                pending_hc,
                position_ids=positions(1, past + 1),
                capture=capture,
            )
        past += len(result.committed_input_tokens)
        target_hidden = result.target_hidden
        pending = result.next_token
        generated.append(pending)
    return generated


def _capture_visual(visual, capture):
    for gear in visual.gears:
        graph = visual.models[gear]
        with capture.step("visual", graph.hmonnx_session, graph_name=f"visual_m{gear}"):
            graph.forward(*visual._full_capacity_inputs(gear))


def _encode_visual(visual, prepared, capture):
    grid = prepared["image_grid_thw"]
    if int(grid.shape[0]) != 1:
        raise ValueError("Qwen4 unified inference accepts one image")
    valid = int(grid.prod()) // int(visual.visual_meta.spatial_merge_size) ** 2
    gear = next(value for value in visual.gears if value >= valid)
    context = (
        capture.step("visual", visual.models[gear].hmonnx_session, graph_name=f"visual_m{gear}")
        if capture is not None
        else nullcontext()
    )
    with context:
        return visual.encode_many(prepared["pixel_values"], grid)


def _run_text_generation(
    meta_file, *, prompt, max_new_tokens, device, image=None, golden_components=(), golden_dir=None, device_map=None
):
    """Return text, token count, latency, throughput and request statistics."""
    model = None
    mtp_runtime = None
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
        prepared = None
        if image is not None:
            from .multimodal import prepare_image_prompt

            visual_config = model.meta_info.visual_config
            gears = visual_config.gears
            prepared = prepare_image_prompt(
                model.hf_model_dir, prompt, image, max_image_tokens=max(int(g.image_token_capacity) for g in gears)
            )

        if prepared is not None:
            input_ids = prepared["input_ids"]
        elif getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        else:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        eos_ids = resolve_eos_token_ids(model, tokenizer=tokenizer)
        inputs_embeds = None
        positions = None
        visual = None
        if prepared is not None or (capture is not None and "visual" in capture.components):
            from .vision import VisualTokenGearHMONNXModel

            visual = VisualTokenGearHMONNXModel(model.meta_info.visual_config, device=device)
            for graph in visual.models.values():
                graph.to(device=device)
        if prepared is not None:
            from .multimodal import merge_image_embeddings

            features = _encode_visual(visual, prepared, capture)
            source_config = json.loads((Path(model.hf_model_dir) / "config.json").read_text())
            inputs_embeds = merge_image_embeddings(
                input_ids.to(model.device), model.embed_tokens, features, source_config["image_token_id"]
            )
            positions = prepared["position_ids"].to(model.device)
        mtp_runtime = None
        mtp_runtime = create_mtp_runtime(model, meta_file=meta_file, device_map=device_map, capture=capture)
        output = generate_token_ids(
            model,
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_ids,
            inputs_embeds=inputs_embeds,
            position_ids=positions,
            capture=capture,
            mtp_runtime=mtp_runtime,
        )
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        if capture is not None and "visual" in capture.components:
            # Exercise the actual request before writing large static baselines.
            # Draft weights are no longer needed during the largest visual gear.
            if mtp_runtime is not None:
                mtp_runtime.close()
                mtp_runtime = None
            _capture_visual(visual, capture)
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
                **({"golden_artifacts": artifacts} if artifacts else {}),
            },
        )
    finally:
        # Runtime sessions do not expose close(); dropping the owning model also
        # drops its graph modules, paged caches, and internal recurrent state.
        if mtp_runtime is not None:
            mtp_runtime.close()
        del model


@torch.inference_mode()
def run_text_generation(
    meta_file, *, prompt, max_new_tokens, device, image=None, golden_components=(), golden_dir=None, device_map=None
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
            golden_components=golden_components,
            golden_dir=golden_dir,
            device_map=device_map,
        )
