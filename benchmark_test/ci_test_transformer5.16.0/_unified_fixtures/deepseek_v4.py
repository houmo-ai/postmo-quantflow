# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

from _unified_preparation import prepare_context


def prepare(case, root, monkeypatch):
    catalog, recipe, source, calibration, image, kwargs, inputs, replacements, family = prepare_context(case, root)
    from _deepseek_v4_small_fixture import write_deepseek_v4_checkpoint
    config = write_deepseek_v4_checkpoint(source)
    geometry = {key: config[key] for key in ("hidden_size", "num_attention_heads", "head_dim",
        "q_lora_rank", "sliding_window", "hc_mult", "hc_sinkhorn_iters", "index_topk", "index_head_dim")}
    replacements.update({"export.model.max_layers": 4, "export.model.graph_geometry": geometry})
    if case.variant == "dspark":
        replacements["export.model.dspark.target_layer_ids"] = [1, 2, 3]
        kwargs["speculative"] = {"mode": "dspark", "num_draft_tokens": 5}
        inputs["max_new_tokens"] = 8
    kwargs.update(context_length=2048, prefill_chunk_length=256, quant_checkpoint=str(source))
    monkeypatch.setitem(catalog._recipes, case.model_id, replace(recipe,
        source_constraints=(("model_type", "deepseek_v4"),), max_pe_length=2048,
        config_replacements=tuple(replacements.items())))
    inputs["devices"] = ["cuda:0"]
    return kwargs, inputs
