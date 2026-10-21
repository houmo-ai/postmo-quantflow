"""Prefill-decode contracts for the LLM-family MOSS adaptation."""

import json
from pathlib import Path

import pytest
import torch
import yaml

from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.prefill_decode import (
    FLASH_ATTENTION_BIT_NAMES,
    MossLMHead,
    parse_flash_attention,
)

_WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / "configs_merak/workflows/xh2a/llm_models/moss_transcribe_diarize/0_9b/moss_transcribe_diarize.yaml"
)
_WORKFLOW_16K = _WORKFLOW.with_name("moss_transcribe_diarize_16k.yaml")
_WORKFLOW_64K = _WORKFLOW.with_name("moss_transcribe_diarize_64k.yaml")
_WORKFLOW_128K = _WORKFLOW.with_name("moss_transcribe_diarize_128k.yaml")


def test_parse_flash_attention_yaml_switch():
    assert parse_flash_attention(None) is None
    assert parse_flash_attention({"enable": False, "q_bits": 8}) is None
    assert parse_flash_attention({"enable": True}) == {name: 16 for name in FLASH_ATTENTION_BIT_NAMES}
    assert parse_flash_attention({"enable": True, "q_bits": 8, "k_bits": 8, "v_bits": 8, "p_bits": 8}) == {
        "q_bits": 8, "k_bits": 8, "v_bits": 8, "p_bits": 8,
    }
    with pytest.raises(ValueError, match="q_bits"):
        parse_flash_attention({"enable": True, "q_bits": 4})

def test_checked_in_prefill_decode_is_512_2048_xh2a_w8a8():
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    model = data["export"]["model"]
    wrap = model["wrap_cfg"]

    assert model["chip_arch"] == "XH2a"
    assert model["quant_type"] == "w8a8_sefp"
    assert model["prefill_chunk_length"] == 512
    assert model["context_max_length"] == 2048
    assert model["max_pe_length"] == 262144
    assert model["flash_attention"]["enable"] is False
    assert model["export_cfg"]["input_names"][0] == "inputs_embeds"
    assert "input_ids" not in model["export_cfg"]["input_names"]
    assert wrap["kv_cache"]["cache_axis"] == 2

def test_checked_in_16k_64k_and_128k_workflows_enable_flash_attention():
    short = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    fa16 = yaml.safe_load(_WORKFLOW_16K.read_text(encoding="utf-8"))
    fa64 = yaml.safe_load(_WORKFLOW_64K.read_text(encoding="utf-8"))
    fa128 = yaml.safe_load(_WORKFLOW_128K.read_text(encoding="utf-8"))

    assert short["export"]["model"]["flash_attention"]["enable"] is False
    assert fa16["export"]["model"]["flash_attention"]["enable"] is True
    assert fa64["export"]["model"]["flash_attention"]["enable"] is True
    assert fa128["export"]["model"]["flash_attention"]["enable"] is True
    assert fa16["export"]["model"]["context_max_length"] == 16384
    assert fa64["export"]["model"]["context_max_length"] == 65536
    assert fa64["export"]["model"]["prefill_chunk_length"] == 4096
    assert fa128["export"]["model"]["context_max_length"] == 131072
    for model in (short, fa16, fa64, fa128):
        assert model["export"]["model"]["max_pe_length"] == 262144
    for model in (fa16, fa128):
        bits = model["export"]["model"]["flash_attention"]
        assert {bits[name] for name in ("q_bits", "k_bits", "v_bits", "p_bits")} == {8}
    # 64k carries the 30-minute recipe, where 8-bit FlashAttention operands cost
    # accuracy: k_bits/v_bits also set the paged KV cache mantissa width, and the
    # error accumulates over ~18k decode steps until the timestamp anchor is lost.
    # Measured on AliMeeting far (20 x 30-min clips, paper protocol): 28.12 CER
    # with 2 runaway files at 8 bit, 23.52 with none at 16 bit, fp16 being 23.60.
    bits64 = fa64["export"]["model"]["flash_attention"]
    assert {bits64[name] for name in ("q_bits", "k_bits", "v_bits", "p_bits")} == {16}

def test_yaml_fused_embedding_abi_has_no_input_ids_or_audio():
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    names = data["export"]["model"]["export_cfg"]["input_names"]
    outputs = data["export"]["model"]["export_cfg"]["output_names"]

    assert names == [
        "inputs_embeds",
        "past_seq_length",
        "current_input_length",
        "past_key_cache",
        "past_value_cache",
    ]
    assert "input_ids" not in names
    assert "input_features" not in names
    # lm_head now runs in the decoder graph, so the first output carries
    # vocabulary logits rather than the final hidden state. This block of
    # export_cfg is inert either way (config.export_cfg is never read; the
    # names come from model.get_export_cfg), but it is kept truthful.
    assert "logits" in outputs
    assert "last_hidden_state" not in outputs

def test_workflow_context_override_reaches_model_config(tmp_path):
    from xhmodel_merak.workflows import AutoWorkflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    workflow = AutoWorkflow.from_config(model_dir=str(model_dir), config_path=str(_WORKFLOW))
    model = workflow.get_model(
        QuantResult(raw_model_dir=str(model_dir), skipped=True),
        config_overrides={
            "export.model.context_max_length": 64,
            "export.model.prefill_chunk_length": 8,
        },
    )
    assert model.config.context_max_length == 64
    assert model.config.prefill_chunk_length == 8

def test_export_rejects_invalid_context_before_native_load(tmp_path):
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.configuration import (
        XHMossTranscribeDiarizeConfig,
    )
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.model import XHMossTranscribeDiarizeModel

    cfg = XHMossTranscribeDiarizeConfig(
        model_name="moss_tiny",
        context_max_length=8,
        prefill_chunk_length=16,
        max_pe_length=32,
        convert_hmonnx=False,
    )
    model = XHMossTranscribeDiarizeModel(cfg)
    with pytest.raises(ValueError, match="prefill_chunk_length"):
        model.export_hmonnx(str(tmp_path / "out"))

def test_model_puts_yaml_flash_attention_on_wrap_cfg():
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.configuration import (
        XHMossTranscribeDiarizeConfig,
    )
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.model import XHMossTranscribeDiarizeModel

    cfg = XHMossTranscribeDiarizeConfig(
        model_name="moss_tiny",
        flash_attention={"enable": True, "q_bits": 16, "k_bits": 16, "v_bits": 16, "p_bits": 16},
        convert_hmonnx=False,
    )
    model = XHMossTranscribeDiarizeModel(cfg)
    assert model.wrap_cfg["flash_attention"]["enable"] is True

def test_moss_lm_head_projects_hidden_states_to_vocab():
    head = MossLMHead(hidden_size=4, vocab_size=11)
    logits = head(torch.zeros(1, 2, 4))

    assert logits.shape == (1, 2, 11)

def test_serving_aliases_require_published_graphs():
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.workflow import publish_moss_serving_aliases

    meta = {
        "components": {
            "prefill_decode": {
                "prefill": {"hmonnx_file": "prefill/a.onnx"},
                "decode": {"onnx_file": "decode/b.onnx"},
            }
        }
    }
    published = publish_moss_serving_aliases(meta)
    assert published["prefill_hmonnx"] == "prefill/a.onnx"
    assert published["decode_hmonnx"] == "decode/b.onnx"

def test_export_info_keeps_graphs_in_workflow_workdir(tmp_path):
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.configuration import (
        XHMossTranscribeDiarizeConfig,
    )
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.model import XHMossTranscribeDiarizeModel

    cfg = XHMossTranscribeDiarizeConfig(model_name="moss_tiny", convert_hmonnx=False)
    model = XHMossTranscribeDiarizeModel(cfg)
    info = model.get_export_info(str(tmp_path))

    assert Path(info.exported_dir) == tmp_path.resolve()
    assert (tmp_path / "hmquant_moss_tiny").is_dir()
    assert info.model_name == "hmquant_moss_tiny"

