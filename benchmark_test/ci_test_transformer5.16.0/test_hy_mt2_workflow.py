"""HY-MT2 HMONNX workflow, evaluation, and runtime contracts."""

import json
from pathlib import Path

import pytest


CONFIG_PATH = Path("configs_merak/workflows/xh2a/other_models/hy_mt2/hy_mt2.yaml")
PACKAGE_PATH = Path("xhmodel_merak/xh_other_model/models/hy_mt2")


def test_hy_mt2_auto_workflow_is_discoverable() -> None:
    from xhmodel_merak.workflows import AutoWorkflow

    workflow = AutoWorkflow.from_config("/tmp/hy-mt2-model", str(CONFIG_PATH))

    assert type(workflow).__name__ == "HyMT2Workflow"
    assert workflow.workflow_config.export["model"]["type"] == "XHHyMT2Model"


def test_hy_mt2_input_names_follow_runtime_contract() -> None:
    from xhmodel_merak.xh_other_model.models.hy_mt2._export_utils import (
        xh1_hmonnx_compatible,
    )

    assert xh1_hmonnx_compatible(
        [
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "past_key_cache_3",
            "past_value_cache_3",
        ]
    ) == [
        "input_1",
        "valid_length",
        "current_length",
        "model_layers_3_self_attn_kcache_input",
        "model_layers_3_self_attn_vcache_input",
    ]


@pytest.mark.parametrize(
    ("source_code", "target_code", "expected_prefix", "target_name"),
    [
        ("zho_Hans", "eng_Latn", "将以下文本翻译为", "英语"),
        ("eng_Latn", "zho_Hans", "Translate the following text into", "Chinese"),
    ],
)
def test_hy_mt2_uses_official_translation_prompt(
    source_code: str,
    target_code: str,
    expected_prefix: str,
    target_name: str,
) -> None:
    from xhmodel_merak.xh_other_model.models.hy_mt2.evaluation import (
        build_translation_prompt,
    )

    prompt = build_translation_prompt("A short sentence.", source_code, target_code)

    assert prompt.startswith(expected_prefix)
    assert target_name in prompt
    assert prompt.endswith("\n\nA short sentence.")


def test_hy_mt2_representative_prompt_is_padded_with_real_token_length() -> None:
    import torch

    from xhmodel_merak.xh_other_model.models.hy_mt2.evaluation import (
        build_padded_chat_input_ids,
        build_representative_prompt,
    )

    class FakeTokenizer:
        pad_token_id = 99
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": build_representative_prompt()}]
            assert kwargs["add_generation_prompt"] is True
            return {"input_ids": torch.tensor([[1, 7, 8, 2]], dtype=torch.long)}

    input_ids, token_count = build_padded_chat_input_ids(FakeTokenizer(), build_representative_prompt(), 8)

    assert token_count == 4
    assert input_ids.tolist() == [[1, 7, 8, 2, 99, 99, 99, 99]]


def test_hy_mt2_export_unties_embeddings_without_legacy_torchscript_config() -> None:
    import torch

    from xhmodel_merak.xh_other_model.models.hy_mt2.hy_mt2_converter import (
        _untie_word_embeddings_for_export,
    )

    class Config:
        tie_word_embeddings = True

    class TinyTiedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = Config()
            self.input_embeddings = torch.nn.Embedding(4, 3)
            self.output_embeddings = torch.nn.Linear(3, 4, bias=False)

        def tie_weights(self) -> None:
            self.output_embeddings.weight = self.input_embeddings.weight

        def get_output_embeddings(self):
            return self.output_embeddings

    model = TinyTiedModel()
    _untie_word_embeddings_for_export(model)

    assert model.config.tie_word_embeddings is False
    assert torch.equal(model.output_embeddings.weight, model.input_embeddings.weight)
    assert model.output_embeddings.weight.data_ptr() != model.input_embeddings.weight.data_ptr()


def test_hy_mt2_translation_metrics_are_exact_for_identical_text() -> None:
    from xhmodel_merak.xh_other_model.models.hy_mt2.evaluation import (
        compute_translation_metrics,
    )

    metrics = compute_translation_metrics(
        ["Hello world.", "Machine translation."],
        ["Hello world.", "Machine translation."],
        target_code="eng_Latn",
    )

    assert metrics["chrf_pp"] == pytest.approx(100.0)
    assert metrics["bleu"] == pytest.approx(100.0)


def test_hy_mt2_rejects_incomparable_baseline_sample_counts(tmp_path: Path) -> None:
    from examples_merak.llm.hy_mt2.eval_flores import _metric_deltas

    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(
        json.dumps(
            {
                "metrics": {
                    "zho_Hans-eng_Latn": {
                        "samples": 100,
                        "chrf_pp": 50.0,
                        "bleu": 25.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="baseline sample count"):
        _metric_deltas(
            {"zho_Hans-eng_Latn": {"samples": 1, "chrf_pp": 40.0, "bleu": 20.0}},
            str(baseline_file),
        )


def test_hy_mt2_rejects_incomparable_baseline_context(tmp_path: Path) -> None:
    from examples_merak.llm.hy_mt2.eval_flores import _metric_deltas

    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(
        json.dumps(
            {
                "offset": 0,
                "metrics": {
                    "zho_Hans-eng_Latn": {
                        "samples": 1,
                        "chrf_pp": 50.0,
                        "bleu": 25.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="baseline offset"):
        _metric_deltas(
            {"zho_Hans-eng_Latn": {"samples": 1, "chrf_pp": 40.0, "bleu": 20.0}},
            str(baseline_file),
            evaluation_context={"offset": 1},
        )


def test_hy_mt2_rejects_stale_resumed_prediction() -> None:
    from examples_merak.llm.hy_mt2.eval_flores import _validate_existing_record
    from xhmodel_merak.xh_other_model.models.hy_mt2.evaluation import TranslationSample

    sample = TranslationSample(
        sample_id="devtest:0:zho_Hans-eng_Latn",
        source_code="zho_Hans",
        target_code="eng_Latn",
        source="新文本",
        reference="New text",
    )
    stale_record = {**sample.to_dict(), "source": "旧文本", "prompt": "prompt", "prediction": "Old text"}

    with pytest.raises(ValueError, match="incompatible source.*--overwrite"):
        _validate_existing_record(stale_record, sample, "prompt")


def test_hy_mt2_retained_prefill_session_restores_fixed_input_length() -> None:
    import torch

    from xhmodel_merak.xh_other_model.models.hy_mt2.inference import HyMT2Inference

    inference = HyMT2Inference.__new__(HyMT2Inference)
    torch.nn.Module.__init__(inference)
    inference.retain_sessions = True
    inference.prefill_session = torch.nn.Identity()
    inference.decode_session = torch.nn.Identity()
    inference.prefill_input_sequence_length = 256
    inference.input_sequence_length = 1

    inference.set_phase_prefill(True)

    assert inference.input_sequence_length == 256


def test_hy_mt2_eval_resolves_legacy_meta_from_export_root(tmp_path: Path) -> None:
    from xhmodel_merak.xh_other_model.models.hy_mt2.evaluation import (
        resolve_legacy_meta,
    )

    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "export_meta_info.json").write_text(
        json.dumps({"legacy_meta": "meta.json"}),
        encoding="utf-8",
    )

    assert resolve_legacy_meta(tmp_path) == tmp_path / "meta.json"


def test_hy_mt2_package_stays_inside_xh_other_model_boundary() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE_PATH.glob("*.py"))

    assert "xh_model_zoo" not in source
    assert "_legacy_base_converter" not in source
    assert "from ....utils" not in source
    assert "self.execution_device" not in source
    assert "self._prefill =" not in source
