# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Export the four DSpark components with one target-owned quantized head."""
from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from xhquant.api import (PrecisionMode, ptq_quantize, to_export_hmonnx_from_quanted_graph,
                         to_frontend_graph, to_quant_graph)

from ...shared_head import bind_quantized_head, quantized_head
from .dspark import DSparkContext, DSparkDraft, DSparkMarkov, _share_operator_tensors
from .dspark_runtime import DSparkSlidingCache, dspark_attention_mask


class _HeadOwner(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.lm_head = _share_operator_tensors(head)

    def forward(self, hidden):
        return self.lm_head(hidden)


def export_dspark_components(draft: DSparkDraft, main_projection: nn.Module,
                             main_norm: nn.Module, markov: DSparkMarkov,
                             token_embedding: nn.Module, output_dir: str | Path,
                             *, activation_bits: int, device: str,
                             fixed_target_head: nn.Module | None = None) -> dict:
    """Export deployable graphs; target head calibration is reused when supplied.

    Zero-valued shape inputs exercise the graph ABI; SEFP weights are rounded
    per G64 group by the SDK rather than calibrated against those inputs.
    End-to-end answer quality and acceptance are measured separately.
    """
    spec = draft.spec
    scheme = spec.quant_scheme(activation_bits)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target_device = torch.device(device)
    scalar = lambda value: torch.tensor([value], dtype=torch.int32, device=target_device)
    context = DSparkContext(main_projection, main_norm, list(draft.layers), spec,
                             input_sequence_length=spec.context_chunk_length, position_rope=draft.position_rope)
    context_decode = DSparkContext(main_projection, main_norm, list(draft.layers), spec,
                                    input_sequence_length=spec.block_size + 1, position_rope=draft.position_rope)
    components = {"dspark_context": context, "dspark_context_decode": context_decode,
                  "dspark_draft_decode": draft, "dspark_markov": markov}
    if fixed_target_head is None:
        head_args = [torch.zeros(1, 1, spec.hidden_size, dtype=torch.float16, device=target_device)]
        owner = _HeadOwner(draft.lm_head).to(target_device)
        graph = to_quant_graph(to_frontend_graph(owner, "TorchFX", head_args), "XH2a",
                                {"quant_type": f"w{spec.head_weight_bits}a8h1_sefp"})
        ptq_quantize(graph, [head_args], PrecisionMode.ALIGNED, [device], infer_shape=False)
        fixed_target_head = quantized_head(graph)
    if fixed_target_head.w_cfg.qspec.man_bit != spec.head_weight_bits:
        raise ValueError("DSpark must reuse the requested target head precision")
    # Streamed target export offloads fixed weights to CPU. PTQ skips this
    # shared operator, so place it with the draft before binding its tensors.
    fixed_target_head.to(target_device)
    entries = {}
    reports = {}
    for role, model in components.items():
        model = model.to(device=target_device, dtype=torch.float16).eval()
        cache = DSparkSlidingCache(spec, device=target_device)
        if role in ("dspark_context", "dspark_context_decode"):
            length = spec.context_chunk_length if role == "dspark_context" else spec.block_size + 1
            args = [torch.zeros(1, length, len(spec.target_layer_ids) * spec.hidden_size,
                                 dtype=torch.float16, device=target_device),
                    scalar(0), scalar(length), *cache.caches]
        elif role == "dspark_draft_decode":
            ids = torch.full((1, spec.block_size), spec.noise_token_id, dtype=torch.int32, device=target_device)
            # Host embedding stays target-owned and is not copied into this package.
            embedding_device = next(token_embedding.parameters()).device
            embedding = token_embedding(ids.to(embedding_device).long()).to(target_device).half()
            args = [embedding, scalar(0), scalar(spec.block_size),
                    dspark_attention_mask(spec, 0, device=target_device), *cache.caches]
        else:
            args = [torch.zeros(1, 1, dtype=torch.int32, device=target_device),
                    torch.zeros(1, 1, spec.hidden_size, dtype=torch.float16, device=target_device)]
        graph = to_quant_graph(to_frontend_graph(model, "TorchFX", args), "XH2a", scheme)
        if role == "dspark_draft_decode":
            bind_quantized_head(graph, fixed_target_head)
        ptq_quantize(graph, [args], PrecisionMode.ALIGNED, [device], infer_shape=False)
        with torch.inference_mode():
            outputs = graph(*args)
            if not all(torch.isfinite(value.dequantize()).all() for value in outputs):
                raise ValueError(f"DSpark component {role} produced nonfinite values")
        graph_file = root / role / f"hmquant_{role}_with_act.onnx"
        to_export_hmonnx_from_quanted_graph(graph, args, str(graph_file), save_as_external_data=True)
        entries[role] = {"path": graph_file.relative_to(root).as_posix(), "type": "hmonnx"}
        reports[role] = {"output_shapes": [list(value.shape) for value in outputs]}
        del outputs, args, graph, cache
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
    metadata = {"format_version": 1, "spec": asdict(spec), "components": entries,
                "metadata_file": "dspark_runtime_meta_info.json",
                "weight_precision": {"routed_experts": 4, "shared_experts": 8, "dense": 8,
                                     "group_size": 64, "shared_lm_head": spec.head_weight_bits},
                "activation_bits": activation_bits, "shared_lm_head_owner": "target",
                "shared_embedding_owner": "target", "shape_validation": reports}
    (root / "dspark_runtime_meta_info.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


__all__ = ["export_dspark_components"]
