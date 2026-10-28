# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Small real OCR/RM builds, relocation and image inference without source weights."""

import copy
import json
from dataclasses import replace

from _unified_preparation import prepare_context
from _unified_small_model import read_spec, write_processor


def prepare(case, root, monkeypatch):
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForSequenceClassification
    from xhmodel_merak.xh_llm.models.minicpm_ocr.native import MiniCPMOcrConfig, MiniCPMOcrNativeModel

    catalog, recipe, source, _, image, kwargs, _, replacements, _ = prepare_context(case, root)
    spec = copy.deepcopy(read_spec(case.fixture))
    config_dict = spec["config"]
    if case.fixture == "minicpm_ocr":
        config = MiniCPMOcrConfig(**config_dict)
        model_cls = MiniCPMOcrNativeModel
        replacements.update(
            {"export.model.visual_config.token_capacity": 16, "export.model.visual_config.validation_target": [4, 4]}
        )
        inputs = {"image": str(image), "prompt": "OCR:", "max_new_tokens": 2}
        extra = {}
    else:
        config = Qwen3_5Config(**config_dict)
        model_cls = Qwen3_5ForSequenceClassification
        inputs = {"orig_image": str(image), "render_image": str(image), "ocr_text": "42"}
        extra = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4891)
        model = model_cls(config).eval()
        with torch.no_grad():
            for name, value in model.named_parameters():
                if "norm" in name and value.ndim == 1 and value.numel() > 1:
                    value.add_(torch.linspace(-0.02, 0.02, value.numel()))
    source.mkdir()
    model.to(torch.bfloat16).save_pretrained(source)
    # Retain the delivered architecture strings, including RM's incorrect one,
    # so the explicit classifier selection remains part of the real build.
    path = source / "config.json"
    saved = json.loads(path.read_text())
    saved["architectures"] = config_dict["architectures"]
    path.write_text(json.dumps(saved))
    write_processor(source, spec)
    monkeypatch.setitem(
        catalog._recipes,
        case.model_id,
        replace(
            recipe,
            source_constraints=(("model_type", config_dict["model_type"]),),
            max_pe_length=256,
            config_replacements=tuple(replacements.items()),
            **extra,
        ),
    )
    kwargs.update(context_length=256, prefill_chunk_length=64)
    return kwargs, inputs


def validate_output(case, output, result, moved, inputs, root, record):
    import onnx

    meta = json.loads((moved / "golden_meta_info.json").read_text())
    graph = onnx.load(moved / meta["prefill_hmonnx"], load_external_data=False)
    cumsum = []
    for node in graph.graph.node:
        if any("cumsum_matmul" in value for value in node.input):
            attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            cumsum.append((attributes["hmfp_act_man_bit"], attributes["hmfp_weight_man_bit"]))
    # Qwen's FX pragma keeps cumsum activations at 16 bits; the constant
    # matrix uses 8 bits. This documented exception also applies to W8 OCR.
    assert cumsum and set(cumsum) == {(16, 8)}
    assert any(node.op_type == "GDRChunkScan" for node in graph.graph.node)
    for node in graph.graph.node:
        if node.op_type not in {"Linear", "MatMul", "GroupMatMul"}:
            continue
        attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        expected = (16, 8) if any("cumsum_matmul" in value for value in node.input) else (8, 8)
        assert (attributes["hmfp_act_man_bit"], attributes["hmfp_weight_man_bit"]) == expected
    if case.fixture == "ocr_rm":
        import pytest
        from xhmodel_merak.xh_llm.models.ocr_rm.precision import audit_sefp_h1, check_sefp_h1_graph

        audit = audit_sefp_h1(moved / "golden_meta_info.json")
        assert audit["passed"] and len(audit["graphs"]) == 5
        # Inject an h0 attribute into an actual SDK-exported graph. The audit
        # must reject it even though the declared QuantScheme still says h1.
        faulty = onnx.ModelProto()
        faulty.CopyFrom(graph)
        flag = next(
            attr for node in faulty.graph.node for attr in node.attribute if attr.name == "hmfp_weight_hidden_bit"
        )
        flag.i = 0
        with pytest.raises(ValueError, match="forbids SEFP h0"):
            check_sefp_h1_graph(faulty, "injected_h0")
        assert output.output_type == "classification"
        assert 0 <= output.stats["prob_1"] <= 1
        assert abs(output.stats["prob_0"] + output.stats["prob_1"] - 1) < 1e-6
    else:
        assert output.output_type == "text" and output.output_tokens > 0
    assert not list(moved.rglob("*.log"))
