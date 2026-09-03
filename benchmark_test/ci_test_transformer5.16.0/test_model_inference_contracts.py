import numpy as np
import pytest

from xhmodel_merak.model_inference import ActionInferenceResult, AudioInferenceResult, TextInferenceResult


def _common(output_type: str):
    return {
        "model_id": "model-id",
        "task": "task",
        "output_type": output_type,
        "artifact_dir": "/artifact",
        "meta_file": "/artifact/meta.json",
        "latency_s": 0.5,
        "components": ("component",),
    }


def test_text_action_and_audio_results_have_task_specific_payloads() -> None:
    text = TextInferenceResult(
        **_common("text"),
        output_text="ready",
        output_tokens=1,
        tokens_per_second=2.0,
    )
    actions = ActionInferenceResult(
        **_common("actions"),
        actions=((0.1, 0.2),),
        horizon=1,
        action_dim=2,
        seed=7,
    )
    audio = AudioInferenceResult(
        **_common("audio"),
        waveform=np.asarray([0.1, -0.1], dtype=np.float32),
        sample_rate=24_000,
        num_samples=2,
        duration_s=2 / 24_000,
        real_time_factor=0.1,
        voice="zf_001",
        speed=1.0,
    )

    assert text.to_dict()["output_text"] == "ready"
    assert actions.to_dict()["actions"] == [[0.1, 0.2]]
    assert audio.to_dict()["waveform"] == pytest.approx([0.1, -0.1])


def test_result_serialization_rejects_non_finite_numbers() -> None:
    audio = AudioInferenceResult(
        **_common("audio"),
        waveform=np.asarray([float("nan")], dtype=np.float32),
        sample_rate=24_000,
        num_samples=1,
        duration_s=1 / 24_000,
        real_time_factor=1.0,
        voice="zf_001",
        speed=1.0,
    )

    with pytest.raises(ValueError, match="finite"):
        audio.to_dict()
