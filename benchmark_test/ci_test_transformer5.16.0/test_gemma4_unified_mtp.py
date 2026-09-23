"""Gemma4 public MTP source, independent-head and decode-only contracts."""

import copy
import json
from types import SimpleNamespace

import onnx
import pytest
from onnx import helper, TensorProto

from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_build.workflow import materialize_workflow_config
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.model_integrations.gemma4.artifact import validate_mtp

MODELS = ["gemma4-e2b", "gemma4-e4b", "gemma4-12b-unified", "gemma4-31b", "gemma4-26b-a4b"]


def sources(tmp_path, model):
    recipe = get_recipe(model)
    target = {}
    for dotted, value in recipe.source_constraints:
        if dotted == "architectures.0":
            target["architectures"] = [value]
            continue
        cursor = target
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    text = target["text_config"]
    text.update(vocab_size=256, num_key_value_heads=1, head_dim=32, layer_types=["sliding_attention", "full_attention"])
    draft = {
        "model_type": target["model_type"] + "_assistant",
        "backbone_hidden_size": text["hidden_size"],
        "text_config": {**text, "hidden_size": 128, "num_hidden_layers": 2, "num_kv_shared_layers": 2},
    }
    for folder, config in [("target", target), ("assistant", draft)]:
        path = tmp_path / folder
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config))
    return tmp_path / "target", tmp_path / "assistant"


@pytest.mark.parametrize("model", MODELS)
def test_mtp_reuses_public_model_identity_and_binds_sources(tmp_path, model):
    target, assistant = sources(tmp_path, model)
    kwargs = dict(model_id=model, model_dir=str(target), output_dir=str(tmp_path / "out"))
    base = resolve_build_plan(**kwargs)
    plan = resolve_build_plan(
        **kwargs, sources={"mtp": str(assistant)}, speculative={"mode": "mtp", "num_draft_tokens": 3}
    )
    assert plan.variant == "mtp" and base.variant == "full"
    assert plan.quant_method == base.quant_method
    assert plan.precision == base.precision
    assert plan.low_memory is False
    assert plan.config_replacements["export.model.mtp_config.target_hf_model"] == str(target)
    assert plan.config_replacements["export.model.mtp_config.assistant_hf_model"] == str(assistant)
    output = tmp_path / "effective.yaml"
    materialize_workflow_config(plan, output)
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    resolved = WorkflowConfig.from_file(str(output)).with_overrides(plan.config_overrides)
    cfg = {"export": resolved.export}
    assert cfg["export"]["model"]["num_draft_tokens"] == 3
    assert cfg["export"]["model"]["mtp_config"]["lm_head_quant_type"] == "w4a8h0_ssfp"
    assert cfg["export"]["model"]["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
    with pytest.raises(ValueError):
        resolve_build_plan(**kwargs, speculative={"mode": "mtp"})
    with pytest.raises(ValueError):
        resolve_build_plan(**kwargs, sources={"mtp": str(assistant)}, speculative={"mode": "mtp"}, low_memory=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("backbone_hidden_size", 32),
        ("model_type", "gemma4_unified_assistant"),
        ("vocab_size", 128),
        ("head_dim", 64),
        ("num_kv_shared_layers", 1),
    ],
)
def test_mtp_rejects_wrong_assistant(tmp_path, field, value):
    target, assistant = sources(tmp_path, "gemma4-e2b")
    cfg = json.loads((assistant / "config.json").read_text())
    (cfg if field in cfg else cfg["text_config"])[field] = value
    (assistant / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="incompatible assistant"):
        resolve_build_plan(
            model_id="gemma4-e2b",
            model_dir=str(target),
            output_dir=str(tmp_path / "out"),
            sources={"mtp": str(assistant)},
            speculative={"mode": "mtp"},
        )


def draft_artifact(root):
    root.mkdir(exist_ok=True)
    inputs = [
        helper.make_tensor_value_info(f"shared_{kind}_cache_{mode}", TensorProto.FLOAT16, [1, 1, 8, 32])
        for kind in ["key", "value"]
        for mode in ["sliding", "full"]
    ]
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 1, width])
        for name, width in [("logits", 256), ("assistant_hidden_state", 128)]
    ]
    onnx.save(helper.make_model(helper.make_graph([], "abi-fixture", inputs, outputs)), root / "draft.onnx")
    for phase, count in [("prefill", 1), ("decode", 5)]:
        target_outputs = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, count, width])
            for name, width in [("logits", 256), ("target_hidden_state", 128)]
        ]
        onnx.save(helper.make_model(helper.make_graph([], phase, [], target_outputs)), root / f"{phase}.onnx")
    return {
        "prefill_hmonnx": "prefill.onnx",
        "decode_hmonnx": "decode.onnx",
        "spec_decode": {
            "mode": "mtp",
            "runtime_contract_version": 2,
            "num_draft_tokens": 4,
            "verify_length": 5,
            "draft_decode_onnx": "draft.onnx",
            "draft_onnx": "draft.onnx",
            "standalone_draft_decode_onnx": "draft.onnx",
            "draft": {
                "abi": "gemma4_mtp_dense_shared_kv_v1",
                "decode_hmonnx": "draft.onnx",
                "standalone_decode_hmonnx": "draft.onnx",
                "cache_mutation": "read_only",
            },
        },
    }


def test_build_load_share_decode_only_contract_and_inventory(tmp_path):
    from xhmodel_merak.model_integrations.autoregressive.export_info import describe_export

    meta = draft_artifact(tmp_path)
    plan = SimpleNamespace(components=("mtp",), config_overrides={"export.model.num_draft_tokens": 4})
    validate_mtp(tmp_path, meta, plan)
    validate_mtp(tmp_path, meta)
    meta.update(
        prefill_hmonnx="prefill.onnx",
        decode_hmonnx="decode.onnx",
        hf_config="hf_config",
        quant_embedding="embedding.pt",
    )
    info = describe_export(tmp_path, meta)
    assert info["mtp_draft_decode"] == {"path": "draft.onnx", "type": "decode"}
    assert "mtp_draft_prefill" not in info
    for key, value in [
        ("shared_head", {"owner": "target"}),
        ("verify_length", 4),
        ("draft_decode_onnx", "../draft.onnx"),
    ]:
        bad = copy.deepcopy(meta)
        bad["spec_decode"][key] = value
        with pytest.raises((ValueError, RuntimeError)):
            validate_mtp(tmp_path, bad, plan)


def test_draft_runtime_uses_target_embedding_and_leaves_cache_untouched():
    import torch
    from xhmodel_merak.xh_llm.models.gemma4_series.mtp_runtime import _build_assistant_inputs

    embedding = torch.nn.Embedding(16, 8)
    hidden = torch.randn(1, 1, 8)
    kv = {
        f"shared_{kind}_cache_{mode}": torch.randn(1, 1, 8, 4)
        for kind in ["key", "value"]
        for mode in ["sliding", "full"]
    }
    before = {k: v.clone() for k, v in kv.items()}
    target = SimpleNamespace(
        device=torch.device("cpu"),
        sliding_window=4,
        get_input_embeddings=lambda: embedding,
        meta_info=SimpleNamespace(model_config=SimpleNamespace(sliding_window=4)),
    )
    session = SimpleNamespace(
        input_names=["inputs_embeds", "kv_window_start_abs", "kv_valid_length", *kv], input_infos={}
    )
    result = _build_assistant_inputs(target, session, 3, hidden, kv, 6)
    torch.testing.assert_close(result["inputs_embeds"][..., :8], embedding(torch.tensor([[3]])))
    torch.testing.assert_close(result["inputs_embeds"][..., 8:], hidden)
    assert result["kv_window_start_abs"].item() == 2
    assert result["kv_valid_length"].item() == 4
    for key in kv:
        assert result[key].data_ptr() == kv[key].data_ptr()
        torch.testing.assert_close(kv[key], before[key])


@pytest.mark.parametrize("family,ordered", [("gemma4", False), ("gemma4", True), ("gemma4_unified", False)])
def test_real_assistant_export_and_v2_readonly_decode(tmp_path, family, ordered):
    """Keep both attention types and the actual ordered/full head in a tiny export."""
    import torch
    from safetensors.torch import save_file
    from xhquant.api import ConfigDict, PrecisionMode, ptq_quantize
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model import (
        _resolve_assistant_text_types,
        XHGemma4SeriesAssistantDraftModel,
    )
    from xhmodel_merak.xh_llm.models.gemma4_series.mtp_runtime import AssistantDraftSession
    from xhmodel_merak.xh_llm.models.gemma4_series.mtp_workflow import draft_quant_config

    torch.manual_seed(52)
    config_cls, model_cls, _ = _resolve_assistant_text_types({"model_type": family + "_assistant"})
    config = config_cls(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=32,
        global_head_dim=64,
        layer_types=["sliding_attention", "full_attention"],
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        sliding_window=8,
        max_position_embeddings=64,
    )
    native = model_cls(config)
    state = {
        "model." + k: v
        for k, v in native.state_dict().items()
        if not any(part in k for part in (".k_proj.", ".v_proj.", ".k_norm.", ".v_norm."))
    }
    state.update(
        {"pre_projection.weight": torch.randn(128, 512) * 0.02, "post_projection.weight": torch.randn(256, 128) * 0.02}
    )
    if ordered:
        state.update(
            {
                "masked_embedding.centroids.weight": torch.randn(8, 128) * 0.02,
                "masked_embedding.token_ordering": torch.randperm(256),
            }
        )
    target, assistant = tmp_path / "target", tmp_path / "assistant"
    target.mkdir()
    assistant.mkdir()
    text = config.to_dict()
    text.update(head_dim=32, global_head_dim=64, num_key_value_heads=2, num_global_key_value_heads=1)
    text["num_kv_shared_layers"] = 2
    (target / "config.json").write_text(json.dumps({"model_type": family, "text_config": {**text, "hidden_size": 256}}))
    (assistant / "config.json").write_text(
        json.dumps(
            {
                "model_type": family + "_assistant",
                "backbone_hidden_size": 256,
                "use_ordered_embeddings": ordered,
                "num_centroids": 8,
                "centroid_intermediate_top_k": 2,
                "text_config": text,
            }
        )
    )
    save_file(state, str(assistant / "model.safetensors"))
    model = XHGemma4SeriesAssistantDraftModel(
        str(assistant),
        str(target),
        ConfigDict(
            input_sequence_length=1,
            context_length=16,
            dtype="float16",
            cache_axis=2,
            shared_sliding_cache_length=16,
            shared_full_cache_length=16,
            target_max_pe_length=64,
        ),
        draft_quant_config("w8a8h1_sefp", "w4a8h0_ssfp"),
    )
    wrapped = model.init_wrap_model()
    assert wrapped.lm_head.weight.data_ptr() == wrapped.model.embed_tokens.weight.data_ptr()
    dummy = model.prepare_inputs(None)
    model.convert_to_fronted_graph(dummy)
    model.convert_to_quant_graph("xh2a")
    ptq_quantize(model.quanted_model, [dummy], PrecisionMode.ALIGNED, [torch.device("cpu")])
    model.convert_to_export_graph(dummy)
    path = model.to_export_onnx(dummy, str(tmp_path / "export"), prefix="assistant")[0]
    session = AssistantDraftSession(path, torch.device("cpu"))
    inputs = dict(zip(model.export_cfg.input_names, dummy, strict=True))
    inputs["current_length"] = inputs.pop("current_input_length")
    inputs["inputs_embeds"].normal_(std=0.1)
    inputs["past_seq_length"].fill_(3)
    inputs["sliding_attention_mask"].fill_(-65504)
    inputs["sliding_attention_mask"][..., :4] = 0
    cache = {k: v for k, v in inputs.items() if k.startswith("shared_")}
    for value in cache.values():
        value.normal_(std=0.1)
    before = {k: v.clone() for k, v in cache.items()}
    first = session(**inputs)
    second = session(**inputs)
    assert [tuple(v.shape) for v in first] == [(1, 1, 256), (1, 1, 256)]
    for a, b in zip(first, second, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key, value in cache.items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    if ordered:
        assert (first[0] > -65504).sum() == 64
