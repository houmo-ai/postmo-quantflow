"""Public build, relocation and inference for native Qwen3.5 MoE text."""
import pytest
from _qwen35_moe_text_checkpoint import write_native_moe_text_checkpoint as _write_native_checkpoint
from _unified_cases import cases_for
from _unified_runner import check_case_result


MODEL_ID = "qwen3.5-moe-text"
CASE_SPEC = ("qwen3.5-moe-text", "qwen_moe_text", "test_unified_model_qwen3_5_moe_text", ("full",))


def prepare(case, root, monkeypatch):
    source, _ = _write_native_checkpoint(root)
    return {
        "model_id": case.model_id,
        "model_dir": str(source),
        "output_dir": str(root / "artifact"),
        "device": "cuda:0",
        "context_length": 64,
        "prefill_chunk_length": 8,
    }, {"prompt": "hello world", "max_new_tokens": 2}


def check_inference(moved, inputs, record):
    import torch

    from xhmodel_merak._environment import scoped_environment
    from xhmodel_merak.model_inference import inference_model

    with scoped_environment({"TORCH_COMPILE_DISABLE": "1"}), torch._dynamo.config.patch(disable=True):
        observed = inference_model(artifact_dir=str(moved), inputs=inputs, device="cuda:0")
    assert observed.model_id == MODEL_ID
    assert observed.components == ("prefill", "decode")
    assert observed.output_tokens == 2
    record.update(passed=True, stage="complete", inference=observed.to_dict())


@pytest.mark.parametrize("case", cases_for(__file__), ids=lambda case: case.id)
def test_public_build_relocate_and_inference(case, unified_runs):
    check_case_result(case, unified_runs)
