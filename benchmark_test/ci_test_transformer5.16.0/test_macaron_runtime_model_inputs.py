from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

import xhmodel_merak.xh_llm.models.macaron_v1_tall.workflow as macaron_workflow_module
from xhmodel_merak.xh_llm.builder import get_model_class
from xhmodel_merak.xh_llm.configuration_auto import (
    MODEL_TYPE_MAPPING_MODULES,
)
from xhmodel_merak.xh_llm.models.macaron_v1_tall.lora_export_config import (
    load_macaron_lora_export_config,
)
from xhmodel_merak.xh_llm.models.macaron_v1_tall.macaron_v1_tall_hmonnx_inference import (
    XHMacaronV1TallHMONNXModel,
)
from xhmodel_merak.xh_llm.models.macaron_v1_tall.macaron_v1_tall_model import (
    XHMacaronV1TallModel,
)
from xhmodel_merak.xh_llm.models.macaron_v1_tall.workflow import (
    MacaronV1TallWorkflow,
)
from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_hmonnx_inference import (
    XHQwen3_5MoeHMONNXModel,
)
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_model import (
    _Qwen3_5HFCompatible,
)
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow
from xhmodel_merak.xh_llm.workflows.result import (
    ExportResult,
    QuantResult,
)


class _RecordingModel:
    device = torch.device("cpu")

    def __init__(self):
        self.args = None

    def to(self, *, device):
        assert device == self.device
        return self

    def __call__(self, *args):
        self.args = args
        return torch.zeros((1, 1, 1), dtype=torch.float16)


class _CompatibilityLLM:
    def __init__(self):
        self.prefill_calls = 0
        self.decode_calls = 0
        self.input_sequence_lengths = []

    def set_prefill(self) -> None:
        self.prefill_calls += 1

    def set_decode(self) -> None:
        self.decode_calls += 1

    def set_input_sequence_length(self, length: int) -> None:
        self.input_sequence_lengths.append(length)

    def get_input_sequence_length(self) -> int:
        return 1


def test_macaron_model_type_loads_the_macaron_model_module() -> None:
    model_type = "MacaronV1TallForConditionalGeneration"

    assert MODEL_TYPE_MAPPING_MODULES[model_type] == "macaron_v1_tall"
    assert (
        get_model_class(
            {
                "chip_arch": "XH2a",
                "model_type": model_type,
                "model_name": "macaron_v1_tall",
            }
        )
        is XHMacaronV1TallModel
    )


def test_macaron_workflow_reuses_qwen_config_with_export_overrides() -> None:
    workflow = MacaronV1TallWorkflow.from_config(
        model_dir="/tmp/macaron-model",
        config_path=("configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/" "macaron_v1_tall_w8a8.yaml"),
    )
    export = workflow.workflow_config.with_overrides(workflow.export_overrides()).export["model"]

    assert type(workflow) is MacaronV1TallWorkflow
    assert export["model_type"] == "MacaronV1TallForConditionalGeneration"
    assert export["model_name"] == "macaron_v1_tall"
    assert export["fuse_gdr_ops"] is False
    assert export["fuse_gdr_block_recurrent_ops"] is False


def test_macaron_workflow_keeps_lora_graph_targets_unfused() -> None:
    overrides = MacaronV1TallWorkflow.export_overrides(
        {
            "export.model.fuse_gdr_ops": True,
            "export.model.fuse_gdr_block_recurrent_ops": True,
        }
    )

    assert overrides["export.model.fuse_gdr_ops"] is False
    assert overrides["export.model.fuse_gdr_block_recurrent_ops"] is False


def test_auto_workflow_uses_macaron_workflow_from_yaml() -> None:
    workflow = AutoLLMWorkflow.from_config(
        model_dir="/tmp/macaron-model",
        config_path=("configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/" "macaron_v1_tall_w8a8.yaml"),
    )

    assert type(workflow) is MacaronV1TallWorkflow


def test_macaron_quant_uses_the_qwen_moe_backbone_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = MacaronV1TallWorkflow.from_config(
        model_dir="/tmp/macaron-model",
        config_path=(
            "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/"
            "macaron_v1_tall_w8a8.yaml"
        ),
    )
    calls: dict[str, object] = {}

    def fake_qwen_quant(self, **kwargs):
        calls.update(kwargs)
        return QuantResult(raw_model_dir="/tmp/macaron-model", skipped=True)

    monkeypatch.setattr(Qwen35Workflow, "quant", fake_qwen_quant)

    workflow.quant(
        output_dir="/tmp/macaron-quant",
        device="cuda",
        config_overrides={"quant.iters": 1},
    )

    assert calls["config_overrides"] == {
        "quant.iters": 1,
        "export.model.model_type": "Qwen3_5MoeForConditionalGeneration",
    }


def test_macaron_workflow_injects_after_qwen_parent_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/" "macaron_v1_tall_w8a8.yaml"
    workflow = MacaronV1TallWorkflow.from_config(
        model_dir="/tmp/macaron-model",
        config_path=config_path,
    )
    base_export = tmp_path / "base"
    final_export = tmp_path / "final"
    calls: dict[str, object] = {}

    def fake_qwen_export(self, **kwargs):
        calls["qwen_export"] = kwargs
        base_export.mkdir()
        base_model_export = base_export / "hmquant_export"
        base_model_export.mkdir()
        (base_model_export / "golden_meta_info.json").write_text("{}")
        config_file = base_export / "macaron_v1_tall_w8a8.yaml"
        config_file.write_text("quant: null\nexport: {}\n")
        return ExportResult(
            work_dir=str(base_export),
            config_file=str(config_file),
        )

    def fake_injection(**kwargs):
        calls["injection"] = kwargs
        return final_export / "hmquant_export"

    monkeypatch.setattr(Qwen35Workflow, "export", fake_qwen_export)
    monkeypatch.setattr(
        macaron_workflow_module,
        "inject_macaron_lora_export",
        fake_injection,
    )

    result = workflow.export(
        QuantResult(raw_model_dir="/tmp/macaron-model", skipped=True),
        output_dir=str(final_export),
        device="cpu",
    )

    overrides = calls["qwen_export"]["config_overrides"]
    assert overrides["export.model.model_type"] == ("MacaronV1TallForConditionalGeneration")
    assert calls["injection"]["base_export_dir"] == str(base_export)
    assert calls["injection"]["base_model_export_dir"] == (base_export / "hmquant_export")
    assert calls["injection"]["final_export_dir"] == final_export
    assert result.work_dir == str(final_export)


def test_macaron_lora_yaml_is_qwen_baseline_plus_overlay() -> None:
    config_dir = Path("configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b")
    baseline = yaml.safe_load((config_dir / "qwen3_6_35b_a3b_full_w8a8.yaml").read_text())
    macaron = yaml.safe_load((config_dir / "macaron_v1_tall_w8a8.yaml").read_text())
    config = load_macaron_lora_export_config(macaron)
    macaron.pop("macaron_lora")
    macaron_model = macaron["export"]["model"]
    baseline_model = baseline["export"]["model"]
    assert macaron["quant"] == baseline["quant"]
    assert {key: value for key, value in macaron_model.items() if key not in {"model_type", "model_name"}} == {
        key: value for key, value in baseline_model.items() if key not in {"model_type", "model_name"}
    }
    assert macaron_model["model_type"] == ("MacaronV1TallForConditionalGeneration")
    assert macaron_model["model_name"] == "macaron_v1_tall"
    assert config.adapters == ("L0", "L1", "L2", "L3")
    assert config.non_routed_quantization.quant_type is None
    assert config.routed_expert_quantization.quant_type is None


def test_macaron_w4a8_yaml_uses_w8_lora() -> None:
    config_dir = Path(
        "configs_merak/workflows/xh2a/llm_models/"
        "qwen3_5_moe/35b_a3b"
    )
    baseline = yaml.safe_load(
        (config_dir / "qwen3_6_35b_a3b_full.yaml").read_text()
    )
    macaron = yaml.safe_load(
        (config_dir / "macaron_v1_tall_w4a8.yaml").read_text()
    )
    config = load_macaron_lora_export_config(macaron)

    assert macaron["quant"] == baseline["quant"]
    assert macaron["export"]["model"]["quant_scheme"] == {
        "quant_type": "w4a8h0_ssfp",
        "nodes": {"lm_head": {"quant_type": "w8a8h1_sefp"}},
        "ops": {},
    }
    assert config.non_routed_quantization.quant_type == "w8a8h0_ssfp"
    assert config.routed_expert_quantization.quant_type == "w8a8h0_ssfp"


def test_macaron_non_routed_lora_accepts_standard_quant_schemes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "macaron.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "macaron_lora": {
                    "adapters": {
                        "available": ["L0", "L1", "L2", "L3"],
                    },
                    "quantization": {
                        "non_routed": {
                            "quant_scheme": "w8a8h1_sefp",
                        },
                        "routed_expert": {
                            "quant_scheme": "inherit",
                        },
                    },
                }
            }
        )
    )

    config = load_macaron_lora_export_config(yaml.safe_load(path.read_text()))

    assert config.non_routed_quantization.quant_type == "w8a8h1_sefp"
    assert config.non_routed_quantization.hmfp_spec == {
        "mode": "sefp",
        "man_bit": 8,
        "hidden_bit": True,
    }
    assert config.non_routed_quantization.act_spec == {
        "mode": "sefp",
        "man_bit": 8,
        "hidden_bit": True,
    }
    assert config.routed_expert_quantization.quant_type is None


def test_macaron_routed_lora_accepts_weight_quantization_override() -> None:
    config = {
        "macaron_lora": {
            "adapters": {"available": ["L0", "L1", "L2", "L3"]},
            "quantization": {
                "non_routed": {"quant_scheme": "w8a8h0_ssfp"},
                "routed_expert": {
                    "quant_scheme": "w8a8h0_ssfp",
                },
            },
        }
    }

    parsed = load_macaron_lora_export_config(config)

    assert parsed.routed_expert_quantization.quant_type == "w8a8h0_ssfp"
    assert parsed.routed_expert_quantization.hmfp_spec == {
        "mode": "ssfp",
        "man_bit": 8,
        "hidden_bit": False,
    }
    assert parsed.routed_expert_quantization.act_spec == {
        "mode": "sefp",
        "man_bit": 8,
        "hidden_bit": True,
    }


def test_macaron_lora_config_uses_effective_workflow_overrides() -> None:
    workflow = MacaronV1TallWorkflow.from_config(
        model_dir="/tmp/macaron-model",
        config_path=("configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/" "macaron_v1_tall_w8a8.yaml"),
    )

    effective = workflow.workflow_config.with_overrides(
        {"macaron_lora.quantization.non_routed.quant_scheme": "w8a8h1_sefp"}
    )
    config = load_macaron_lora_export_config(effective.data)

    assert config.non_routed_quantization.quant_type == "w8a8h1_sefp"


def test_qwen_sample_forward_accepts_multimodal_inputs_as_kwargs() -> None:
    parameters = inspect.signature(_Qwen3_5HFCompatible._sample_forward).parameters

    assert parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD


def test_qwen_sample_forward_keeps_visual_inputs() -> None:
    model = _Qwen3_5HFCompatible.__new__(_Qwen3_5HFCompatible)
    model._llm_prefill = True
    model._llm_model = _CompatibilityLLM()
    model._past_seq_length = 0
    received = {}

    def original_forward(**kwargs):
        received.update(kwargs)
        return "ok"

    model._xh_orig_forward = original_forward
    image = torch.zeros((1, 3, 4, 4), dtype=torch.float16)

    assert model._sample_forward(input_ids=torch.tensor([[1, 2]]), pixel_values=image) == "ok"
    assert received["pixel_values"] is image
    assert model._past_seq_length == 2
    assert model._llm_model.prefill_calls == 1
    assert model._llm_model.decode_calls == 1


def test_runtime_lora_model_rejects_fast_mode() -> None:
    model = XHMacaronV1TallHMONNXModel.__new__(XHMacaronV1TallHMONNXModel)

    with pytest.raises(RuntimeError, match="compiler-bound LoRA"):
        model.to_fast()


def test_macaron_runtime_inherits_qwen_hybrid_cache_contract() -> None:
    assert XHMacaronV1TallHMONNXModel.forward is XHQwen3_5MoeHMONNXModel.forward
    assert not hasattr(XHMacaronV1TallHMONNXModel, "set_request_adapter")


def test_macaron_example_entrypoints_keep_stable_adapter_surface() -> None:
    from examples_merak.llm.macaron_v1_tall import hf_demo, hmonnx_demo, macaron_workflow

    assert hf_demo.ADAPTER_NAMES == ["L0", "L1", "L2", "L3"]
    assert set(hf_demo.ADAPTER_ROLES) == set(hf_demo.ADAPTER_NAMES)
    assert macaron_workflow.DEFAULT_CONFIG.endswith(
        "macaron_v1_tall_w8a8.yaml"
    )
    assert callable(hmonnx_demo.find_meta_file)
    assert callable(hmonnx_demo.run_generation)
