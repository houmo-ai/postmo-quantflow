"""MTP draft graph export using the Qwen4 target's graph/quant pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from .mtp_model import Qwen4ExpMTPExportWrapper, Qwen4ExpMTPModel


def default_mtp_quant_scheme(head_weight_bits=8):
    from ..qwen3_5.xh_qwen3_5_config import build_spec_draft_quant_scheme

    scheme = build_spec_draft_quant_scheme(head_weight_bits)
    scheme["ops"] = {
        name: {"quant_type": "w4a8h0_ssfp"} for name in ("MoeBlock", "GPTQPackedMoeBlock")
    }
    return scheme


def export_qwen4_exp_mtp(
    model_dir,
    output_dir,
    *,
    cache_length,
    input_sequence_length=1,
    quant_config=None,
    device="cpu",
    dtype=torch.float16,
    lm_head=None,
    assistant=None,
    convert_to_hmonnx=True,
    validate_export_graph_parity=False,
    cache_abi="llm_cache_inplace_v1",
):
    """Export a fixed-size assistant chunk with explicit in-place K/V by default.

    ``model_dir`` must retain the checkpoint's MTP payload. The target head
    can be passed by ownership to avoid loading it twice. This exports a
    separate assistant; target verify graphs belong to serving_workflow.
    """
    from ...shared_head import load_checkpoint_head
    from .cache_contract import QWEN4_EXP_LLM_CACHE_ABI, resolve_qwen4_exp_cache_abi, resolve_qwen4_exp_cache_shape
    from .hmonnx_export import (
        _qwen4_exp_cache_input_names,
        _sample_mrope_position_inputs,
        convert_qwen4_exp_kv_to_llm_cache,
        convert_qwen4_exp_kv_to_page_cache,
        export_qwen4_exp_hmonnx,
    )

    cache_abi = resolve_qwen4_exp_cache_abi(cache_abi)
    if convert_to_hmonnx and dtype != torch.float16:
        raise ValueError("Qwen4 MTP HMONNX requires float16 graph tensors")
    if not 1 <= input_sequence_length <= cache_length:
        raise ValueError("Require 1 <= MTP input_sequence_length <= cache_length")
    if quant_config is None:
        quant_config = default_mtp_quant_scheme()
    if "quant_scheme" in quant_config:
        raise ValueError("MTP quant_config accepts the quant scheme directly, not a nested quant_scheme")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    assistant = assistant if assistant is not None else Qwen4ExpMTPModel.from_pretrained(model_dir, dtype=dtype)
    if lm_head is None:
        lm_head, _ = load_checkpoint_head(model_dir, dtype=dtype)
    assistant = assistant.to(device=device, dtype=dtype).eval()
    lm_head = lm_head.to(device=device, dtype=dtype).eval()
    wrapper = Qwen4ExpMTPExportWrapper(
        assistant, lm_head, input_sequence_length=input_sequence_length,
        cache_length=cache_length,
    ).eval()
    c = assistant.config
    from ...mrope import MROPE_CURRENT_POSITION_NAMES, MROPE_POOL_POSITION_NAMES

    cache_names = _qwen4_exp_cache_input_names(wrapper.cache_contract)
    inputs = (
        "next_token_embedding",
        "target_hidden",
        "past_seq_length",
        "current_input_length",
        *MROPE_CURRENT_POSITION_NAMES,
        *MROPE_POOL_POSITION_NAMES,
        *cache_names,
    )
    outputs = ("logits", "mtp_hidden_states", *(f"present_{name}" for name in cache_names))
    cache_tensors = tuple(
        torch.zeros(
            resolve_qwen4_exp_cache_shape(spec, batch_size=1, context_length=cache_length),
            device=device,
            dtype=torch.int32 if spec.dtype == "integer" else dtype,
        )
        for spec in wrapper.cache_contract.tensors
    )
    sample = (
        torch.zeros((1, input_sequence_length, c.hidden_size), device=device, dtype=dtype),
        torch.zeros((1, input_sequence_length, c.hidden_size * c.hc_count), device=device, dtype=dtype),
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.full((1,), input_sequence_length, device=device, dtype=torch.int32),
        *_sample_mrope_position_inputs(
            batch_size=1, input_length=input_sequence_length, past_length=0, device=device,
        ),
        *cache_tensors,
    )
    exported = export_qwen4_exp_hmonnx(
        assistant,
        root / "raw",
        text_config=c,
        sample_inputs=sample,
        quant_config=quant_config,
        input_sequence_length=input_sequence_length,
        cache_length=cache_length,
        quant_device=device,
        convert_to_hmonnx=convert_to_hmonnx,
        validate_export_graph_parity=validate_export_graph_parity,
        export_wrapper=wrapper,
        export_input_names=inputs,
        export_output_names=outputs,
    )
    graph_path = exported.onnx_path
    if convert_to_hmonnx:
        converter = (convert_qwen4_exp_kv_to_llm_cache
                     if cache_abi == QWEN4_EXP_LLM_CACHE_ABI else convert_qwen4_exp_kv_to_page_cache)
        graph_path = converter(
            graph_path,
            root / "model.onnx",
            context_length=cache_length,
            num_kv_heads=c.num_key_value_heads,
            head_dim=c.head_dim,
        )
    import onnx

    graph = onnx.load(str(graph_path), load_external_data=False).graph
    return {
        "onnx_path": str(graph_path),
        "io_manifest_path": exported.io_manifest_path,
        "cache_contract": wrapper.cache_contract.to_dict(),
        "cache_abi": cache_abi,
        "input_names": tuple(value.name for value in graph.input),
        "output_names": tuple(value.name for value in graph.output),
        "hidden_size": c.hidden_size,
        "hc_count": c.hc_count,
        "input_sequence_length": input_sequence_length,
    }
