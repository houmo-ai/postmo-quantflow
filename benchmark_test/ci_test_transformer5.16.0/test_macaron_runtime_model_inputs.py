"""Macaron configuration selects the ordinary Qwen workflow and runtime."""
import inspect
from pathlib import Path
import torch
import yaml
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_model import _Qwen3_5HFCompatible
from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

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


def test_macaron_yaml_reuses_qwen_workflow(tmp_path):
    root = Path(__file__).resolve().parents[2]
    for name in ("macaron_v1_tall_w4a8.yaml", "macaron_v1_tall_w8a8.yaml", "macaron_v1_tall_lora.yaml"):
        path = root / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b" / name
        config = yaml.safe_load(path.read_text())
        assert config["export"]["model"]["model_type"] == "Qwen3_5MoeForConditionalGeneration"
        assert "macaron_lora" not in config
        workflow = AutoLLMWorkflow.from_config(model_dir=str(tmp_path), config_path=str(path))
        assert isinstance(workflow, Qwen35Workflow)
