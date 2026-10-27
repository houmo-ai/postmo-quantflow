"""Real MiniCPM routing, optional LES frontend branches and SDK execution."""

import copy

from _minicpm_v47_fixture import minicpmv47_modeling  # noqa: F401

import pytest
import torch
from torch import nn


def _native_layers():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    config = Qwen3_5MoeTextConfig(
        hidden_size=32,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
    )
    layers = nn.ModuleList([nn.ModuleDict({"mlp": Qwen3_5MoeSparseMoeBlock(config)}) for _ in range(3)])
    # Standalone fused experts allocate empty tensors; PreTrainedModel's
    # post_init normally initializes them. Keep this block harness finite.
    for parameter in layers.parameters():
        nn.init.normal_(parameter, std=0.02)
    return layers


def _run_layers(layers, hidden):
    for layer in layers:
        hidden = hidden + layer.mlp(hidden)
    return hidden


def test_native_prediction_uses_current_input_next_gate_and_preserves_routing():
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.les import LESObserver

    torch.manual_seed(47)
    layers = _native_layers()
    hidden = torch.randn(1, 1, 32)
    events = []
    baseline = _run_layers(layers, hidden)
    expected_prediction = torch.nn.functional.linear(hidden, layers[1].mlp.gate.weight).topk(2).indices
    with LESObserver(
        layers,
        2,
        prefetch=lambda layer, ids: events.append(("predict", layer)),
        require=lambda layer, ids: events.append(("require", layer)),
    ) as probe:
        actual = _run_layers(layers, hidden)
        assert [record[0] for record in probe.records] == [1, 2]
        torch.testing.assert_close(probe.records[0][1], expected_prediction.reshape(1, 2))
        assert events.index(("predict", 1)) < events.index(("require", 1))
        assert events.index(("predict", 2)) < events.index(("require", 2))
        probe.reset()
        _run_layers(layers, torch.randn(1, 4, 32))
        assert not probe.records  # padded prefill / multi-token verify excluded
        _run_layers(layers, hidden)
        assert len(probe.records) == 2  # no stale cross-call predictions
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    assert all(not layer.mlp.gate._forward_hooks and not layer.mlp.gate._forward_pre_hooks for layer in layers)


def test_scheduling_callbacks_cannot_mutate_native_routing_or_prediction_records():
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.les import LESObserver

    torch.manual_seed(47)
    layers = _native_layers()
    hidden = torch.randn(1, 1, 32)
    with LESObserver(layers, 2) as reference:
        expected = _run_layers(layers, hidden)

    def callback(layer, ids):
        ids.fill_(0)  # A backend may reuse/mutate its private scheduling buffer.

    with LESObserver(layers, 2, prefetch=callback, require=callback) as probe:
        actual = _run_layers(layers, hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for expected_record, actual_record in zip(reference.records, probe.records, strict=True):
        torch.testing.assert_close(actual_record[1], expected_record[1], rtol=0, atol=0)
        torch.testing.assert_close(actual_record[2], expected_record[2], rtol=0, atol=0)


@pytest.mark.parametrize(
    "predicted,actual,recall,complete,exact",
    [
        ([1, 2], [2, 1], 1.0, 1.0, 1.0),
        ([1, 2], [2, 3], 0.5, 0.0, 0.0),
        ([1, 2], [3, 4], 0.0, 0.0, 0.0),
        ([1, 2, 3], [2, 1], 1.0, 1.0, 0.0),
    ],
)
def test_metrics_use_expert_sets(predicted, actual, recall, complete, exact):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.les import prediction_counts, prediction_rates

    rates = prediction_rates(prediction_counts(torch.tensor([predicted]), torch.tensor([actual])))
    assert rates["expert_recall"] == recall
    assert rates["complete_topk_coverage"] == complete
    assert rates["exact_topk_match"] == exact


def _wrapped_model():
    from xhquant.api import ConfigDict
    from xhmodel_merak.xh_llm.models.qwen3_5_moe._moe_model import register_wrap_modules
    from xhmodel_merak.xh_llm.wrap_model import wrap_llm_model

    register_wrap_modules()
    layers = _native_layers()
    cfg = ConfigDict(dict(input_sequence_length=1, batch_size=1))
    for layer in layers:
        layer["mlp"] = wrap_llm_model(layer.mlp, cfg)

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = layers

        def forward(self, hidden):
            return _run_layers(self.layers, hidden)

    return Body().eval()


def test_les_real_hmonnx_export_callbacks_and_original_outputs(tmp_path):
    import onnx
    from xhquant.api import (
        PrecisionMode,
        convert_quanted_model_to_hmonnx,
        ptq_quantize,
        to_frontend_graph,
        to_quant_graph,
    )
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.les_graph import (
        HMONNXLESScheduler,
        configure_les_decode_graph,
    )

    torch.manual_seed(47)
    inputs = (torch.randn(1, 1, 32).half(),)
    model = _wrapped_model().half()
    twin = copy.deepcopy(model)
    plain = to_frontend_graph(model, "TorchFX", inputs)
    les = to_frontend_graph(twin, "TorchFX", inputs)
    contract = configure_les_decode_graph(les, 2)
    assert len(contract["outputs"]) == 5
    assert les.les_predict_layer_1.weight is les.get_submodule("layers.1.mlp.gate").weight
    assert len(list(plain.graph.nodes)) + 2 == len(list(les.graph.nodes))
    torch.testing.assert_close(plain(*inputs), les(*inputs)[0], rtol=0, atol=0)
    sessions = []
    for name, frontend in (("plain", plain), ("les", les)):
        graph = to_quant_graph(frontend, "XH2a", {"quant_type": "w8a8h1_sefp"})
        ptq_quantize(graph, [inputs], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
        path = tmp_path / f"{name}.onnx"
        export_cfg = dict(input_names=["hidden"], output_names=["result"])
        if name == "les":
            export_cfg["output_names"] += [entry["name"] for entry in contract["outputs"]]
        convert_quanted_model_to_hmonnx(graph, inputs, str(path), **export_cfg)
        sessions.append(HMONNXInferenceV2(str(path), HMONNXInferenceConfig(exec_devices=["cpu"])))
        exported = onnx.load(path)
        assert len(exported.graph.output) == (6 if name == "les" else 1)
    events = []
    scheduler = HMONNXLESScheduler(
        sessions[1],
        contract,
        prefetch=lambda layer, ids: events.append(("predict", layer, ids.clone())),
        require=lambda layer, ids: events.append(("require", layer, ids.clone())),
    )
    compute_hooks = []
    moes = [module for module in sessions[1].node_modules if module.op_type == "MoeBlock"]
    for layer, module in enumerate(moes):
        compute_hooks.append(
            module.register_forward_pre_hook(lambda module, args, layer=layer: events.append(("compute", layer, None)))
        )
    with torch.inference_mode():
        baseline = sessions[0](*inputs)
        observed = sessions[1](*inputs)
    torch.testing.assert_close(observed[0], baseline, rtol=0, atol=0)
    assert len(scheduler.records) == 2
    assert [(event[0], event[1]) for event in events] == [
        ("predict", 1),
        ("require", 0),
        ("compute", 0),
        ("predict", 2),
        ("require", 1),
        ("compute", 1),
        ("require", 2),
        ("compute", 2),
    ]
    # True IDs come from the probabilities actually fed into MoeBlock.
    for layer, predicted, actual in scheduler.records:
        real_probs = observed[1 + 2 * layer]
        torch.testing.assert_close(actual, real_probs.topk(2, dim=-1).indices)
    saved_records = list(scheduler.records)
    scheduler.prefetch = lambda layer, ids: ids.zero_()
    scheduler.require = lambda layer, ids: ids.zero_()
    with torch.inference_mode():
        with_buffer_mutation = sessions[1](*inputs)
    torch.testing.assert_close(with_buffer_mutation[0], baseline, rtol=0, atol=0)
    for before, after in zip(saved_records, scheduler.records, strict=True):
        torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)
        torch.testing.assert_close(before[2], after[2], rtol=0, atol=0)
    scheduler.close()
    for handle in compute_hooks:
        handle.remove()
    assert not scheduler.handles and not scheduler.pending


@pytest.mark.parametrize("value", [True, -1, 1.5, "8"])
def test_model_rejects_invalid_les_before_loading_weights(value):
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.text_model import MiniCPMV47MoeTextModel

    cfg = XHQwen3_5MoeModelConfig(model_name="les-test", les_prediction_k=value)
    with pytest.raises(ValueError, match="nonnegative integer"):
        MiniCPMV47MoeTextModel(cfg)


def test_workflow_cli_les_is_opt_in(monkeypatch):
    from examples_merak.llm.minicpm_v_4_7.minicpm_v_4_7_workflow import parse_args, _overrides

    monkeypatch.setattr("sys.argv", ["workflow", "--model-dir", "/unused"])
    assert "export.model.les_prediction_k" not in _overrides(parse_args())
    monkeypatch.setattr("sys.argv", ["workflow", "--model-dir", "/unused", "--les-prediction-k", "8"])
    assert _overrides(parse_args())["export.model.les_prediction_k"] == 8


@pytest.mark.parametrize("precision", ["w8a8", "w4a8_autoround", "w4a8_autoround_head_w5"])
def test_workflow_configs_allow_strict_les_override(precision):
    from pathlib import Path
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b"
        / f"minicpm_v_4_7_moe_xh2a_{precision}.yaml"
    )
    config = WorkflowConfig.from_file(str(path))
    assert config.export["model"]["les_prediction_k"] == 0
    enabled = config.with_overrides({"export.model.les_prediction_k": 8})
    assert enabled.export["model"]["les_prediction_k"] == 8
    assert config.export["model"]["les_prediction_k"] == 0
    with pytest.raises(KeyError, match="does not exist"):
        config.with_overrides({"export.model.les_predicton_k": 8})


def test_full_mini_hybrid_les_export_metadata_and_cache_runtime(
    tmp_path,
    monkeypatch,
    restore_model_initialization_state,
):
    import json
    import onnx
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from transformers.models.minicpmv4_7 import modeling_minicpmv4_7 as modeling
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.text_model import MiniCPMV47MoeTextModel
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.inference import MiniCPMV47HMONNXRuntime
    from xhmodel_merak.model_integrations.minicpm_v_4_7.artifact import validate_runtime_contract
    from xhmodel_merak.xh_llm.infer_mixin import LLMInferenceContextManager
    from test_minicpm_v_4_7_auto_classes import _tiny_model

    monkeypatch.setenv("XHQUANT_STRICT_EXPORT", "0")
    monkeypatch.delenv("ENABLE_HMINFERENCE_V2", raising=False)
    torch.set_num_threads(2)
    torch.manual_seed(47)
    config = _tiny_model(modeling, moe=True).config
    config.text_config.num_hidden_layers = 2
    config.text_config.layer_types = ["linear_attention", "full_attention"]
    native = modeling.MiniCPMV4_7ForConditionalGeneration(config).half().eval()
    source = tmp_path / "native"
    native.save_pretrained(source)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "test": 2}, unk_token="[UNK]")),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(source)
    cfg = XHQwen3_5MoeModelConfig(
        model_name="tiny_minicpm_les",
        hf_model=str(source),
        visual_config=None,
        context_max_length=16,
        max_pe_length=128,
        prefill_chunk_length=4,
        les_prediction_k=2,
        quant_scheme={"quant_type": "w8a8h1_sefp"},
    )
    model = MiniCPMV47MoeTextModel(cfg)
    with restore_model_initialization_state():
        loaded = model.get_hf_model(str(source), dtype=torch.float16, attn_implementation="eager")
    model.to_wrap(loaded)
    model.to("cpu")
    meta = model.export_hmonnx(str(tmp_path / "out"))
    exported = next((tmp_path / "out").rglob("golden_meta_info.json"))
    payload = json.loads(exported.read_text())
    assert payload["les"] == meta.les
    prefill = onnx.load(str(exported.parent / meta.prefill_hmonnx), load_external_data=False)
    decode = onnx.load(str(exported.parent / meta.decode_hmonnx), load_external_data=False)
    assert not any(output.name.startswith("les_") for output in prefill.graph.output)
    assert [output.name for output in decode.graph.output][-3:] == [
        "les_router_probs_0",
        "les_prediction_logits_1",
        "les_router_probs_1",
    ]
    root = dict(
        model_type="MiniCPM-V-4.7",
        merak_model_type="MiniCPMV47MoeForConditionalGeneration",
        components=["llm"],
        vision={},
        hf_model=meta.hf_config,
        llm=dict(
            metadata=exported.name,
            prefill_hmonnx=meta.prefill_hmonnx,
            decode_hmonnx=meta.decode_hmonnx,
            hf_config=meta.hf_config,
            quant_embedding=meta.quant_embedding,
            context_max_length=16,
            prefill_chunk_length=4,
            les=meta.les,
        ),
    )
    root_file = exported.parent / "export_meta_info.json"
    root_file.write_text(json.dumps(root))
    validate_runtime_contract(root, components=["llm"], resolve_file=lambda value, field: exported.parent / value)
    family = MiniCPMV47HMONNXRuntime(root_file, device="cpu")
    runtime, scheduler = family.text, family.les_scheduler
    import os

    assert "ENABLE_HMINFERENCE_V2" not in os.environ
    with LLMInferenceContextManager(runtime), torch.inference_mode():
        runtime.set_prefill()
        args = runtime.get_data_preprocessor()({"input_ids": torch.tensor([[2, 2]]), "past_seq_length": 0})
        first = runtime.forward(*args)[0]
        assert torch.isfinite(first).all()
        assert not scheduler.records
        runtime.set_decode()
        args = runtime.get_data_preprocessor()({"input_ids": torch.tensor([[2]]), "past_seq_length": 2})
        decoded = runtime.forward(*args)[0]
        assert torch.isfinite(decoded).all()
        assert len(scheduler.records) == 1
    scheduler.close()


@pytest.mark.parametrize("kind", ["dense", "spec", "prune"])
def test_unsupported_les_combinations_fail_before_model_load(kind):
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5ModelConfig
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.text_model import (
        MiniCPMV47MoeTextModel,
        MiniCPMV47TextModel,
    )

    cls = MiniCPMV47TextModel if kind == "dense" else MiniCPMV47MoeTextModel
    config_cls = XHQwen3_5ModelConfig if kind == "dense" else XHQwen3_5MoeModelConfig
    config = config_cls(model_name="invalid-les", les_prediction_k=2)
    if kind == "spec":
        config.spec_decode_mode = "dflash"
    elif kind == "prune":
        config.dynamic_prune = {"mode": "decode"}
    with pytest.raises(ValueError, match={"dense": "dense", "spec": "single-token", "prune": "ACEP"}[kind]):
        cls(config)


@pytest.mark.parametrize("mutation", ["phase", "layer", "k", "missing_output"])
def test_les_metadata_rejects_inconsistent_contract(mutation):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.les_graph import les_contract, validate_les_contract

    contract = les_contract(3, 2, 2)
    if mutation == "phase":
        contract["phase"] = "prefill"
    elif mutation == "layer":
        contract["outputs"][1]["layer"] = 2
    elif mutation == "k":
        contract["prediction_k"] = True
    else:
        contract["outputs"].pop()
    with pytest.raises(ValueError, match="LES"):
        validate_les_contract(contract)


def test_evaluation_aggregates_token_counts_and_has_ten_cases_per_domain():
    from examples_merak.llm.minicpm_v_4_7.evaluate_les import QUESTIONS, aggregate_results

    assert len(QUESTIONS) == 5 and all(len(questions) == 10 for questions in QUESTIONS.values())
    samples = [
        dict(
            domain="test",
            layers={
                "1": dict(
                    tokens=tokens,
                    expert_hits=hits,
                    actual_experts=tokens * 2,
                    predicted_experts=tokens * 2,
                    complete_hits=complete,
                    exact_hits=complete,
                )
            },
        )
        for tokens, hits, complete in [(1, 2, 1), (3, 3, 0)]
    ]
    result = aggregate_results(samples)
    assert result["overall"]["expert_recall"] == 5 / 8
    assert result["overall"]["exact_topk_match"] == 1 / 4
    assert result["domains"]["test"] == result["per_layer"]["1"] == result["overall"]
