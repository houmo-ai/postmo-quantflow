# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Tiny complete Phase A checkpoint with both real attention kinds."""
import json
from dataclasses import replace

import torch


def write_checkpoint(source):
    from _unified_small_model import read_spec, write_processor
    from xhmodel_merak.xh_llm.models.bonsai.hadamard_matmul import build_hadamard_matmul_weight
    from xhmodel_merak.xh_llm.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from xhmodel_merak.xh_llm.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    source.mkdir(parents=True)
    text = dict(
        vocab_size=512, hidden_size=128, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=64, linear_value_head_dim=64,
        layer_types=['linear_attention', 'full_attention'], max_position_embeddings=128,
        rope_parameters=dict(rope_type='default', rope_theta=10000.0, mrope_section=[3, 3, 2]),
        eos_token_id=0, pad_token_id=0, tie_word_embeddings=False,
    )
    vision = dict(depth=1, hidden_size=8, intermediate_size=16, num_heads=2, patch_size=2,
                  spatial_merge_size=1, temporal_patch_size=1, out_hidden_size=128,
                  num_position_embeddings=4)
    config = Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=False)
    # FLA's gated norm explicitly allocates on CUDA, bypassing this device
    # context. Move the complete model to CPU before preparing the checkpoint.
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.manual_seed(4891)
        model = Qwen3_5ForConditionalGeneration(config).half().cpu().eval()
    signs = torch.ones(128, device='cpu')
    aux = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'norm' in name and parameter.ndim == 1 and parameter.numel() > 1:
                parameter.add_(torch.linspace(-0.02, 0.02, parameter.numel(), device=parameter.device))
        for name, module in model.named_modules():
            language = name.startswith('model.language_model.') or name == 'lm_head'
            if isinstance(module, torch.nn.Linear) and language and not name.endswith(('in_proj_a', 'in_proj_b')):
                codes = (torch.arange(module.weight.numel(), device=module.weight.device)
                         .reshape_as(module.weight) % 3 - 1).to(torch.int8)
                module.weight.copy_(codes.half() * 0.03125)
                module.register_buffer('quant_weight', codes)
                aux[name + '.hadamard_signs'] = signs
    aux['hadamard_matmul.forward.w128'] = build_hadamard_matmul_weight(128, signs, block_size=64)
    # Keep serialized storage portable even if a module/state_dict hook allocates on CUDA.
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()
             if name.startswith(('model.language_model.', 'lm_head.'))}
    aux = {name: value.detach().cpu() for name, value in aux.items()}
    config.save_pretrained(source)
    write_processor(source, read_spec('qwen_dense'))
    torch.save(state, source / 'quant_weight.pt')
    torch.save(aux, source / 'phase_a_aux.pt')
    (source / 'phase_a_metadata.json').write_text(json.dumps(dict(
        format='xh2_pq2_via_ssfp4_compat', version=2, complete_language_model=True,
        float_tensors_converted=True, embedding_inverse_applied=True,
        ssfp={'man_bit': 4}, hadamard={'block_size': 64},
    )))


def prepare(case, root, monkeypatch):
    from _unified_preparation import prepare_context

    catalog, recipe, source, _, _, kwargs, inputs, _, _ = prepare_context(case, root)
    write_checkpoint(source)
    kwargs.update(context_length=32, prefill_chunk_length=8)
    monkeypatch.setitem(catalog._recipes, case.model_id, replace(
        recipe, max_pe_length=128,
        source_constraints=(("text_config.hidden_size", 128), ("text_config.num_hidden_layers", 2)),
    ))
    return kwargs, inputs


def validate_build(case, result, record):
    import onnx
    from pathlib import Path

    root = Path(result.output_dir)
    meta = json.loads((root / 'golden_meta_info.json').read_text())
    for mode in ('prefill', 'decode'):
        graph = onnx.load(root / meta[mode + '_hmonnx']).graph
        assert not any('placeholder' in node.op_type.lower() for node in graph.node)
        assert any('recurrent' in value.name for value in graph.output)
    record['phase_a_components'] = ['prefill', 'decode']
