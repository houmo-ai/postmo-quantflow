# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Real CPU integration of OCR calibration, workflow and chunked generation."""

import copy
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageFont
from test_paddleocr_vl_model import tiny_config
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    PaddleOCRVLForConditionalGeneration,
    PaddleOCRVLImageProcessor,
    PaddleOCRVLProcessor,
    PreTrainedTokenizerFast,
)

from examples_merak.llm.paddleocr_vl import paddleocr_vl_workflow, prepare_calibration
from xhmodel_merak.workflows import AutoWorkflow
from xhmodel_merak.xh_llm.models.paddleocr_vl.calibration import load_calibration_inputs
from xhmodel_merak.xh_llm.models.paddleocr_vl.processing import prepare_inputs


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "configs_merak/workflows/xh2a/llm_models/paddleocr_vl/paddleocr_vl_1_6.yaml"


def _save_processor(path):
    vocab = {"[UNK]": 0, "OCR:": 1, "[EOS]": 2, "42": 3, "<image>": 60}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]",
        extra_special_tokens={"image_token": "<image>"},
    )
    processor = PaddleOCRVLProcessor(
        image_processor=PaddleOCRVLImageProcessor(image_mean=[0.5] * 3, image_std=[0.5] * 3),
        tokenizer=tokenizer,
        chat_template="{% for message in messages %}{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}<image>{% else %}{{ item['text'] }}{% endif %}"
        "{% endfor %}{% endfor %}",
    )
    processor.save_pretrained(path)
    tiny_config().save_pretrained(path)
    return processor


def test_calibration_appends_answer_and_eos_after_visual_prompt(tmp_path):
    processor = _save_processor(tmp_path)
    image = Image.new("RGB", (56, 28), "red")
    image.save(tmp_path / "image.png")
    manifest = tmp_path / "calibration.jsonl"
    manifest.write_text(json.dumps({"image": "image.png", "prompt": "OCR:", "answer": "42"}) + "\n")
    prompt = prepare_inputs(processor, image, "OCR:", width=56, height=28)
    actual = load_calibration_inputs(tmp_path, manifest, nsamples=1, width=56, height=28)[0]
    count = prompt["input_ids"].shape[1]
    torch.testing.assert_close(actual["input_ids"][:, :count], prompt["input_ids"])
    assert actual["input_ids"][0, count:].tolist() == [3, 2]
    assert actual["mm_token_type_ids"][0, count:].tolist() == [0, 0]
    torch.testing.assert_close(actual["pixel_values"], prompt["pixel_values"])
    assert actual["attention_mask"].all()
    with pytest.raises(ValueError, match="nsamples"):
        load_calibration_inputs(tmp_path, manifest, nsamples=2)
    manifest.write_text(json.dumps({"image": "image.png", "prompt": "OCR:", "answer": ""}) + "\n")
    with pytest.raises(ValueError, match="answer"):
        load_calibration_inputs(tmp_path, manifest, width=56, height=28)


def test_rendered_calibration_is_reproducible_and_labels_have_consistent_balances(tmp_path, monkeypatch):
    font = ImageFont.truetype("DejaVuSans.ttf", 20).path
    manifests = []
    for name in ("first", "second"):
        output = tmp_path / name
        monkeypatch.setattr(sys, "argv", [
            "prepare_calibration", "--output-dir", str(output), "--samples", "2", "--seed", "73",
            "--font", str(font),
        ])
        prepare_calibration.main()
        records = [json.loads(line) for line in (output / "calibration.jsonl").read_text().splitlines()]
        assert len(records) == 2
        for record in records:
            image = output / record["image"]
            assert hashlib.sha256(image.read_bytes()).hexdigest() == record["sha256"]
            with Image.open(image) as pixels:
                assert pixels.mode == "RGB" and pixels.size == (896, 896)
            assert record["prompt"] == "Table Recognition:"
            assert record["answer_format"] == "otsl"
            assert record["answer"].endswith("<nl>")
            assert "<table>" not in record["answer"]
            rows = [row.split("<fcel>")[1:] for row in record["answer"].split("<nl>")[:-1]]
            assert len(rows) == 17
            assert all(len(row) == 5 for row in rows)
            assert rows[0] == ["Date", "Reference", "Debit", "Credit", "Balance"]
            previous = None
            for row in rows[1:]:
                debit, credit, balance = map(Decimal, row[2:])
                assert (debit == 0) != (credit == 0)
                if previous is not None:
                    assert balance == previous - debit + credit
                previous = balance
        manifests.append(records)
    assert manifests[0] == manifests[1]


def test_cli_rejects_calibration_with_a_base_export_config(tmp_path, monkeypatch, capsys):
    from xhquant.api import get_xhquant_logger

    with capsys.disabled():
        get_xhquant_logger()
    tiny_config().save_pretrained(tmp_path / "hf")
    monkeypatch.setattr(sys, "argv", [
        "paddleocr_vl_workflow", "--model-dir", str(tmp_path / "hf"), "--config", str(WORKFLOW),
        "--output-dir", str(tmp_path / "output"), "--golden-image", "image.png",
        "--calibration-jsonl", "calibration.jsonl",
    ])
    with pytest.raises(SystemExit) as error:
        paddleocr_vl_workflow.main()
    assert error.value.code == 2
    assert "enabled quant section" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_workflow_parent_quanted_aligned_then_visual_token_gear_export_survives_trace_pruning(
    tmp_path,
    monkeypatch,
):
    torch.manual_seed(92)
    monkeypatch.setattr("xhmodel_merak.xh_llm.base_model.XHBaseModel.check_transformer_version", lambda self: True)
    config = tiny_config()
    config.text_config.vocab_size = 128
    native = PaddleOCRVLForConditionalGeneration(config).eval().half()
    native.config._attn_implementation = "eager"
    native.config.text_config._attn_implementation = "eager"
    native.save_pretrained(tmp_path / "hf")
    workflow = AutoWorkflow.from_config(str(tmp_path / "hf"), str(WORKFLOW))
    model = workflow.get_model(
        workflow.quant(str(tmp_path / "unused"), "cpu"),
        {
            "export.model.dtype": "float16",
            "export.model.context_max_length": 32,
            "export.model.max_pe_length": 64,
            "export.model.prefill_chunk_length": 4,
            "export.model.visual_config.max_size_h": 56,
            "export.model.visual_config.max_size_w": 56,
            "export.model.visual_config.image_token_gears": [4, 6],
            "export.model.visual_config.image_token_capacity": 6,
        },
        work_dir=str(tmp_path / "work"),
    )
    model.to_quanted_aligned()
    assert not hasattr(model.visual._native_model_root, "visual")
    exports = []

    def export_without_hmonnx(self, output_dir):
        exports.append({
            "profile_dir": Path(output_dir).name,
            "dummy_shape": tuple(self.get_dummy_inputs()["pixel_values"].shape),
            "snapshot_has_visual": hasattr(self._native_visual_source, "visual"),
            "root_has_visual": hasattr(self._native_model_root, "visual"),
        })
        graph = Path(output_dir) / "graph.onnx"
        graph.parent.mkdir(parents=True, exist_ok=True)
        graph.write_text("graph\n")
        return graph

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.paddleocr_vl.visual.BaseVisionModel._export_hmonnx",
        export_without_hmonnx,
    )

    metadata = model.visual.export_hmonnx(str(tmp_path / "export" / "visual"))

    assert [item["profile_dir"] for item in exports] == ["visual_m4", "visual_m6"]
    assert [item["dummy_shape"] for item in exports] == [(1, 16, 3, 14, 14), (1, 24, 3, 14, 14)]
    assert all(item["snapshot_has_visual"] for item in exports)
    assert not any(item["root_has_visual"] for item in exports)
    assert [gear["image_token_capacity"] for gear in metadata.gears] == [4, 6]



@pytest.mark.parametrize("with_image", [False, True])
def test_workflow_wrap_generates_native_tokens_across_three_prefill_chunks(tmp_path, with_image, monkeypatch):
    torch.manual_seed(91)
    monkeypatch.setattr("xhmodel_merak.xh_llm.base_model.XHBaseModel.check_transformer_version", lambda self: True)
    config = tiny_config()
    config.text_config.eos_token_id = 2
    config.text_config.pad_token_id = 0
    native = PaddleOCRVLForConditionalGeneration(config).eval().half()
    native.config._attn_implementation = "eager"
    native.config.text_config._attn_implementation = "eager"
    native.save_pretrained(tmp_path / "hf")
    workflow = AutoWorkflow.from_config(str(tmp_path / "hf"), str(WORKFLOW))
    quant_result = workflow.quant(str(tmp_path / "unused"), "cpu")
    assert quant_result.skipped and not (tmp_path / "unused").exists()
    model = workflow.get_model(quant_result, {
        "export.model.dtype": "float16", "export.model.context_max_length": 32,
        "export.model.max_pe_length": 64, "export.model.prefill_chunk_length": 4,
        "export.model.visual_config.max_size_h": 56, "export.model.visual_config.max_size_w": 56,
        "export.model.visual_config.image_token_gears": [4],
        "export.model.visual_config.image_token_capacity": 4,
    }, work_dir=str(tmp_path / "wrap"))
    assert model.visual.dtype == torch.float16
    model.to_wrap(copy.deepcopy(native))
    input_ids = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13, 14]])
    mm_types = torch.zeros_like(input_ids)
    image_inputs = {}
    if with_image:
        input_ids[0, 1] = 61
        input_ids[0, 2:6] = 60
        input_ids[0, 6] = 62
        mm_types[0, 2:6] = 1
        image_inputs = {"pixel_values": torch.randn(16, 3, 14, 14, dtype=torch.float16),
                        "image_grid_thw": torch.tensor([[1, 4, 4]])}
    kwargs = dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                  mm_token_type_ids=mm_types, max_new_tokens=3, do_sample=False, eos_token_id=[63],
                  **image_inputs)
    with torch.inference_mode():
        expected = native.generate(**kwargs, use_cache=True)
        actual = model.generate(**kwargs)
        repeated = model.generate(**kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
    assert actual.shape[1] > input_ids.shape[1]
    assert model.get_export_cfg()["input_names"][:4] == [
        "inputs_embeds", "time_position_ids", "height_position_ids", "width_position_ids",
    ]
    model.to(dtype=torch.float32)
    with pytest.raises(ValueError, match="float16 KV-cache ABI"):
        model.generate(**kwargs)
