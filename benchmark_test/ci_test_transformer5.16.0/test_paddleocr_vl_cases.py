"""CPU image/processor and issue-report contracts; no weights or GPU inference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


processing = _load("paddleocr_case_processing", "xhmodel_merak/xh_llm/models/paddleocr_vl/processing.py")
evaluation = _load("paddleocr_case_evaluation", "examples_merak/llm/paddleocr_vl/evaluate_cases.py")


@pytest.fixture(scope="module")
def processor():
    # The real native image processor and tokenizer run on tiny pixels and an
    # in-memory vocabulary. There is no download or model-directory dependency.
    from tokenizers import Tokenizer, models
    from transformers import PaddleOCRVLImageProcessor, PaddleOCRVLProcessor, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0, "<image>": 1, "OCR:": 2}, unk_token="[UNK]")),
        unk_token="[UNK]",
        extra_special_tokens={"image_token": "<image>"},
    )
    return PaddleOCRVLProcessor(
        image_processor=PaddleOCRVLImageProcessor(image_mean=[0.5] * 3, image_std=[0.5] * 3),
        tokenizer=tokenizer,
        chat_template="{% for message in messages %}{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}<image>{% else %}{{ item['text'] }}{% endif %}"
        "{% endfor %}{% endfor %}",
    )


@pytest.mark.parametrize("mode", ["L", "RGBA", "RGB"])
def test_load_rgb_preserves_geometry_and_does_not_apply_exif(mode):
    image = Image.new(mode, (6, 2))
    image.getexif()[274] = 6
    actual = processing.load_rgb_image(image)
    assert actual.mode == "RGB"
    assert actual.size == (6, 2)


def test_letterbox_retains_pixels_and_explicit_archived_content_size():
    image = Image.new("RGB", (8, 2), "red")
    boxed = processing.prepare_image(image, "letterbox", 6, 6, content_size=(6, 2))
    assert boxed.getpixel((3, 0)) == (255, 255, 255)
    assert boxed.getpixel((3, 1)) == (255, 255, 255)
    assert boxed.getpixel((3, 2)) == (255, 0, 0)
    assert boxed.getpixel((3, 3)) == (255, 0, 0)
    assert boxed.getpixel((3, 4)) == (255, 255, 255)
    stretched = processing.prepare_image(image, "stretch", 6, 6)
    assert stretched.tobytes() == bytes([255, 0, 0]) * 36


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resize_mode": "unknown"},
        {"width": 0},
        {"resize_mode": "stretch", "content_size": (2, 2)},
        {"resize_mode": "letterbox", "width": 4, "height": 4, "content_size": (5, 2)},
    ],
)
def test_invalid_resize_settings_fail(kwargs):
    with pytest.raises(ValueError):
        processing.prepare_image(Image.new("RGB", (8, 8)), **kwargs)


@pytest.mark.parametrize(
    "mode,color,normalized",
    [
        ("L", 255, [1.0, 1.0, 1.0]),
        ("RGB", (255, 0, 128), [1.0, -1.0, 2 * 128 / 255 - 1]),
    ],
)
def test_native_processor_keeps_fixed_patch_grid_and_rgb_values(processor, mode, color, normalized):
    import torch

    image = Image.new(mode, (80, 20), color)
    inputs = processing.prepare_inputs(processor, image, "OCR:", width=56, height=28)
    assert inputs["image_grid_thw"].tolist() == [[1, 2, 4]]
    assert inputs["pixel_values"].shape == (8, 3, 14, 14)
    expected = torch.tensor(normalized).reshape(1, 3, 1, 1).expand(8, 3, 14, 14)
    torch.testing.assert_close(inputs["pixel_values"], expected)
    assert (inputs["input_ids"] == processor.tokenizer.image_token_id).sum().item() == 2
    assert inputs["input_ids"].dtype == torch.int64
    assert inputs["mm_token_type_ids"].shape == inputs["input_ids"].shape


def test_native_processor_rejects_patch_merge_incompatible_shape(processor):
    with pytest.raises(ValueError, match="divisible"):
        processing.prepare_inputs(processor, Image.new("RGB", (3, 3)), "OCR:", width=30, height=28)


@pytest.mark.parametrize("size,grid", [((112, 56), [1, 2, 4]), ((56, 112), [1, 4, 2])])
def test_native_resize_preserves_aspect_grid_and_matches_raw_processor(processor, size, grid):
    import torch

    native = type(processor)(
        image_processor=type(processor.image_processor)(
            size={"shortest_edge": 28 * 28, "longest_edge": 56 * 56},
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        ),
        tokenizer=processor.tokenizer,
        chat_template=processor.chat_template,
    )
    image = Image.frombytes("RGB", size, bytes(index % 251 for index in range(size[0] * size[1] * 3)))
    actual = processing.prepare_native_inputs(native, image, "OCR:")
    reference = native(text=["<image>OCR:"], images=[image], return_tensors="pt", do_resize=True)
    assert actual["image_grid_thw"].tolist() == [grid]
    assert actual["pixel_values"].shape == (8, 3, 14, 14)
    assert (actual["input_ids"] == native.tokenizer.image_token_id).sum().item() == 2
    assert actual.keys() == reference.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], reference[name], atol=0, rtol=0)


def test_native_resize_is_explicit_and_available_for_both_backends():
    arguments = ["--model-dir", "model", "--output-dir", "output"]
    assert evaluation.parse_args(arguments).native_resize is False
    assert evaluation.parse_args(arguments + ["--native-resize"]).native_resize is True
    deployed = evaluation.parse_args(arguments + ["--backend", "hmonnx", "--meta-file", "meta", "--native-resize"])
    assert deployed.native_resize is True


def _region(box=(1, 1, 7, 7), polygon=None):
    return {
        "polygon_points": polygon,
        "crop_recovery": {
            "crop_size": [box[2] - box[0], box[3] - box[1]],
            "source_crops": [{"crop_box": list(box), "coordinate": [2, 2, 6, 6], "polygon_mask_applied": True}],
        },
    }


def test_region_uses_archived_padding_and_polygon_mask():
    image = Image.new("RGB", (8, 8), "red")
    region = _region(polygon=[[2, 2], [6, 2], [2, 6]])
    actual = processing.recover_region_image(image, region)
    assert actual.size == (6, 6)
    assert actual.getpixel((0, 0)) == (255, 255, 255)
    assert actual.getpixel((1, 1)) == (255, 0, 0)
    assert actual.getpixel((5, 5)) == (255, 255, 255)


def test_region_recovers_constituent_polygons_and_left_aligned_merge():
    image = Image.new("RGB", (8, 8), "red")
    region = {
        "crop_recovery": {
            "merged_on_white_canvas": True,
            "merge_aligns": ["left"],
            "crop_size": [6, 5],
            "source_crops": [
                {"crop_box": [0, 0, 6, 2], "coordinate": [0, 0, 6, 2], "polygon_mask_applied": True},
                {"crop_box": [0, 4, 2, 7], "coordinate": [0, 4, 2, 7], "polygon_mask_applied": True},
            ],
        }
    }
    source = {
        "layout_detector_regions": [
            {"coordinate": [0, 0, 6, 2], "polygon_points": [[0, 0], [6, 0], [6, 2], [0, 2]]},
            {"coordinate": [0, 4, 2, 7], "polygon_points": [[0, 4], [2, 4], [2, 7], [0, 7]]},
        ]
    }
    actual = processing.recover_region_image(image, region, layout_source=source)
    assert actual.size == (6, 5)
    assert actual.getpixel((5, 0)) == (255, 0, 0)
    assert actual.getpixel((0, 4)) == (255, 0, 0)
    assert actual.getpixel((5, 4)) == (255, 255, 255)


@pytest.mark.parametrize(
    "region,match",
    [
        (_region(), "polygon_points"),
        (_region(box=(-1, 0, 2, 2)), "outside"),
        ({"crop_recovery": {}}, "source_crops"),
    ],
)
def test_region_missing_or_invalid_metadata_fails(region, match):
    with pytest.raises(ValueError, match=match):
        processing.recover_region_image(Image.new("RGB", (8, 8)), region)


def test_report_detects_empty_row_and_digit_degeneration():
    text = "<table><tr><td>value</td></tr>" + "<tr><td></td><td></td></tr>" * 139 + "</table>" + "0" * 1925
    result = evaluation.analyze_output(text)
    assert result["table_row_count"] == 140
    assert result["empty_table_rows"] == result["trailing_empty_table_rows"] == 139
    assert result["longest_identical_nonspace_run"] == 1925
    assert result["longest_identical_nonspace_character"] == "0"
    assert result["closed_table_count"] == 1
    assert result["table_format"] == "html"
    assert result["otsl"] is None


def test_native_otsl_keeps_empty_cells_and_counts_tail_rows():
    text = "<fcel>日期<fcel>金额<nl><fcel>20080828<ecel><nl>" + "<ecel><ecel><nl>" * 3
    result = evaluation.analyze_output(text)
    assert result["table_format"] == "otsl"
    assert result["table_count"] == 1
    assert result["table_row_count"] == 5
    assert result["empty_table_rows"] == result["trailing_empty_table_rows"] == 3
    assert result["tables"] == [[["日期", "金额"], ["20080828", ""], ["", ""], ["", ""], ["", ""]]]
    assert result["otsl"]["row_lengths"] == [2] * 5
    assert result["otsl"]["repairs_applied"] is False
    assert result["otsl"]["rows"][1][1] == {"marker": "ecel", "text": ""}
    assert result["closed_table_count"] == 0  # Still counts HTML closing tags only.


def test_native_otsl_scores_simple_bank_cells_by_row_and_column():
    headers = ["客户账号", "交易日期", "交易流水号", "存入金额", "转出金额", "账户余额", "摘要", "对方账号", "对方名称"]
    rows = [headers]
    assertions = {"table_headers": headers, "cell_assertions": []}
    for index in range(37):
        rows.append(
            [str(index + 1), "20080828", "140851234153", "200,000.00", "0.00", "30.78", "转入", "123456", "张三"]
        )
        assertions["cell_assertions"].append(
            {"row_1based_excluding_header": index + 1, "expected_cells": {"对方账号": "123456"}}
        )
    # Both values remain in the text, but one row swaps account/name columns.
    rows[18][-2:] = ["张三", "123456"]
    text = "".join("".join("<fcel>" + value for value in row) + "<nl>" for row in rows)
    result = evaluation.analyze_output(text, assertions)
    assert result["table_format"] == "otsl"
    assert result["otsl"]["row_lengths"] == [9] * 38
    assert result["bank_cells"]["status"] == "evaluated_partial_only"
    checks = result["bank_cells"]["checks"]
    assert len(checks) == 37
    assert sum(item["matches"] for item in checks) == 36
    assert checks[17]["actual"] == "张三"


@pytest.mark.parametrize(
    "tail,reason,widths",
    [
        ("<fcel>10<lcel><nl><ucel><xcel><nl>", "merged_cells_not_scored", [2, 2, 2]),
        ("<fcel>10<nl>", "ragged_rows_not_scored", [2, 1]),
        ("<fcel>10<ecel>unexpected text<nl>", "malformed_otsl_not_scored", [2, 2]),
    ],
)
def test_native_otsl_preserves_merge_and_ragged_rows_without_false_scores(tail, reason, widths):
    labels = {
        "table_headers": ["a", "b"],
        "cell_assertions": [{"row_1based_excluding_header": 1, "expected_cells": {"a": "10"}}],
    }
    result = evaluation.analyze_output("<fcel>a<fcel>b<nl>" + tail, labels)
    assert result["otsl"]["row_lengths"] == widths
    assert result["otsl"]["repairs_applied"] is False
    assert result["bank_cells"]["status"] == "not_evaluable"
    assert result["bank_cells"]["reason"] == reason
    assert result["bank_cells"]["checks"] == []
    assert result["empty_table_rows"] == result["trailing_empty_table_rows"] == 0


def test_native_otsl_keeps_unterminated_last_row_without_padding():
    result = evaluation.analyze_output("<fcel>a<fcel>b<nl><fcel>10")
    assert result["tables"] == [[["a", "b"], ["10"]]]
    assert result["otsl"]["row_lengths"] == [2, 1]
    assert result["otsl"]["ends_with_row_separator"] is False
    assert result["otsl"]["repairs_applied"] is False


def test_report_finds_menu_phrase_loop_missed_by_single_character_metric():
    prefix = "时间\n星期一\n星期二\n星期三\n星期四\n星期五\n星期六\n午餐\n"
    text = prefix + "菜名 配料 重量克 " * 509 + "菜名 "
    result = evaluation.analyze_output(text)
    assert result["table_format"] == "plain_text"
    assert result["longest_identical_nonspace_run"] == 1
    assert result["phrase_repetition"]["unit"] == "菜名 配料 重量克 "
    assert result["phrase_repetition"]["consecutive_complete_repetitions"] == 509
    assert result["phrase_repetition"]["start_char_offset_0based"] == len(prefix)
    assert result["phrase_repetition"]["end_char_offset_0based_exclusive"] == len(text) - len("菜名 ")
    assert result["characters"] == len(text)
    assert evaluation.analyze_output("7" * 3866)["phrase_repetition"] is None


def test_report_detects_missing_tail_and_does_not_call_substrings_cell_accuracy():
    assertions = {
        "text_assertions": [{"id": "tail", "expected": "窗口咨询"}],
        "table_headers": ["对方账号", "对方名称"],
        "cell_assertions": [
            {"row_1based_excluding_header": 1, "expected_cells": {"对方账号": "123456", "对方名称": "张三"}}
        ],
    }
    text = "<table><tr><td>对方账号</td><td>对方名称</td></tr><tr><td>张三</td><td>123456</td></tr></table>"
    result = evaluation.analyze_output(text, assertions)
    assert [item["expected"] for item in result["partial_visual_anchors"]["missing"]] == ["窗口咨询"]
    assert result["bank_cells"]["table_header_matched"] is True
    assert all(not item["matches"] for item in result["bank_cells"]["checks"])


def test_select_full_cases_and_only_primary_regions_with_native_budgets():
    manifest = {
        "cases": [
            {
                "id": "menu",
                "full_image": {"id": "menu_full", "replay_defaults": {"max_new_tokens": 1024}},
                "regions": [
                    {"primary_problem_region": False},
                    {
                        "id": "menu_table",
                        "primary_problem_region": True,
                        "task": "table",
                        "replay_prompt": "Table Recognition:",
                        "max_new_tokens": 4096,
                        "context_max_length": 16384,
                        "crop_recovery": {
                            "static_resize": {
                                "resize_mode": "letterbox",
                                "target_size": [896, 896],
                                "effective_content_size": [896, 500],
                            }
                        },
                    },
                ],
            }
        ]
    }
    assert [sample["id"] for sample in evaluation.select_samples(manifest)] == ["menu_full"]
    selected = evaluation.select_samples(manifest, ["menu"], include_regions=True)
    assert [sample["id"] for sample in selected] == ["menu_full", "menu_table"]
    assert selected[1]["settings"]["max_new_tokens"] == 4096
    assert selected[1]["settings"]["content_size"] == [896, 500]
    with pytest.raises(ValueError, match="Unknown"):
        evaluation.select_samples(manifest, ["missing"])


@pytest.mark.parametrize(
    "generated,eos,budget,reason",
    [
        ([3, 7], 7, 10, "eos"),
        ([3, 8], [7, 8], 2, "eos"),
        ([3, 4], 7, 2, "max_new_tokens"),
        ([3], 7, 2, "other"),
    ],
)
def test_generation_evidence_preserves_ids_and_termination(generated, eos, budget, reason):
    result = evaluation.generation_summary([1, 2], [1, 2, *generated], eos, budget)
    assert result["input_ids"] == [1, 2]
    assert result["output_ids"] == generated
    assert result["generated_tokens"] == len(generated)
    assert result["finish_reason"] == reason
    assert result["reached_token_limit"] == (len(generated) == budget)


def test_generation_rejects_missing_input_prefix():
    with pytest.raises(ValueError, match="original input IDs"):
        evaluation.generation_summary([1, 2], [3, 4], 7, 2)


def test_cli_requires_hmonnx_metadata_and_positive_budget():
    common = ["--model-dir", "model", "--output-dir", "output"]
    with pytest.raises(SystemExit):
        evaluation.parse_args([*common, "--backend", "hmonnx"])
    with pytest.raises(SystemExit):
        evaluation.parse_args([*common, "--max-new-tokens", "0"])
    args = evaluation.parse_args(
        [
            *common,
            "--backend",
            "hmonnx",
            "--meta-file",
            "meta.json",
            "--case",
            "bank_statement",
            "--include-regions",
            "--max-new-tokens",
            "4096",
        ]
    )
    assert args.meta_file == "meta.json"
    assert args.case == ["bank_statement"]
    assert args.include_regions is True
    assert args.max_new_tokens == 4096


@pytest.mark.parametrize("flags,cache", [([], None), (["--hf-cache"], True), (["--no-hf-cache"], False)])
def test_hf_cache_cli_preserves_optional_override(flags, cache):
    args = evaluation.parse_args(["--model-dir", "model", "--output-dir", "output", *flags])
    assert args.hf_cache is cache
    assert args.dtype == "bfloat16"


@pytest.mark.parametrize("flag", ["--hf-cache", "--no-hf-cache"])
def test_hmonnx_rejects_both_hf_cache_overrides(flag):
    with pytest.raises(SystemExit):
        evaluation.parse_args(
            ["--model-dir", "model", "--output-dir", "output", "--backend", "hmonnx", "--meta-file", "meta.json", flag]
        )


def test_backend_dtype_defaults_and_hmonnx_fp16_cache_abi(capsys):
    common = ["--model-dir", "model", "--output-dir", "output"]
    assert evaluation.parse_args(common).dtype == "bfloat16"
    assert evaluation.parse_args([*common, "--dtype", "float16"]).dtype == "float16"
    hmonnx = [*common, "--backend", "hmonnx", "--meta-file", "meta.json"]
    assert evaluation.parse_args(hmonnx).dtype == "float16"
    with pytest.raises(SystemExit):
        evaluation.parse_args([*hmonnx, "--dtype", "bfloat16"])
    assert "FP16 KV-cache ABI" in capsys.readouterr().err


@pytest.mark.parametrize(
    "backend,cache_flag,expected_cache,quantized",
    [
        ("hf", None, False, False),
        ("hf", "--hf-cache", True, False),
        ("hf", "--no-hf-cache", False, False),
        ("hmonnx", None, True, False),
        ("hf", "--hf-cache", True, True),
    ],
)
def test_case_runner_records_cache_and_forwards_only_requested_options(
    tmp_path, monkeypatch, processor, backend, cache_flag, expected_cache, quantized
):
    """Run real CPU preprocessing/reporting, stubbing only model execution and SDK logging."""
    import torch

    import xhmodel_merak.xh_llm as xh_llm
    import xhquant.api as xhquant_api
    from xhmodel_merak.xh_llm.models.paddleocr_vl import hf
    from xhmodel_merak.xh_llm.models.paddleocr_vl.model import XHPaddleOCRVLModel

    image = tmp_path / "input.png"
    Image.new("RGB", (28, 28), "red").save(image)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"cases": {}}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "visual_assertions": str(labels),
                "cases": [
                    {
                        "id": "tiny",
                        "input_image": str(image),
                        "image_size": [28, 28],
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "regions": [],
                        "full_image": {
                            "id": "tiny_full",
                            "replay_defaults": {
                                "prompt": "OCR:",
                                "resize_mode": "stretch",
                                "target_size": [28, 28],
                                "max_new_tokens": 2,
                                "context_max_length": 16384,
                            },
                        },
                    }
                ],
            }
        )
    )
    calls = {}

    class InferenceBoundary:
        # Checkpoint generation defaults can differ from a model config; the
        # omitted CLI flag must follow generation_config and omit the kwarg.
        generation_config = SimpleNamespace(use_cache=False, eos_token_id=0)
        config = SimpleNamespace(use_cache=True)

        def set_attn_implementation(self, implementation):
            calls["attention"] = implementation

        def to(self, device):
            if quantized:
                assert calls["attention"] == "eager"
            calls["device"] = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls["generation_kwargs"] = kwargs
            return torch.cat([kwargs["input_ids"], torch.tensor([[2, 0]])], dim=1)

    def load_runtime(meta_file, **kwargs):
        calls["runtime_load"] = (meta_file, kwargs)
        return InferenceBoundary()

    def load_dequantized(model_dir, **kwargs):
        calls["dequantized_load"] = (model_dir, kwargs)
        return InferenceBoundary()

    monkeypatch.setattr(hf, "load_paddleocr_processor", lambda model_dir: processor)
    monkeypatch.setattr(hf, "load_paddleocr_model", lambda *args, **kwargs: InferenceBoundary())
    monkeypatch.setattr(xh_llm, "AutoLLMHONNXModel", SimpleNamespace(from_pretrained=load_runtime))
    monkeypatch.setattr(xhquant_api, "xhquant_init", lambda *args: None)
    monkeypatch.setattr(XHPaddleOCRVLModel, "get_hf_model", load_dequantized)
    model_dir = tmp_path / "model"
    config = hf.PaddleOCRVLConfig()
    if quantized:
        config.quantization_config = {"quant_method": "gptq", "bits": 8, "group_size": 64}
    config.save_pretrained(model_dir)
    output = tmp_path / "output"
    output.mkdir()
    argv = [
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(output),
        "--manifest",
        str(manifest),
        "--backend",
        backend,
        "--device",
        "cpu",
    ]
    if backend == "hmonnx":
        argv += ["--meta-file", "metadata.json"]
    if cache_flag:
        argv += [cache_flag]
    evaluation.evaluate(evaluation.parse_args(argv))
    result = json.loads((output / "tiny_full/result.json").read_text())
    assert result["status"] == "completed"
    assert result["use_cache"] is expected_cache
    assert result["generated_tokens"] == 2
    assert result["finish_reason"] == "eos"
    assert (output / "tiny_full/prepared.png").is_file()
    assert (output / "tiny_full/raw_output.txt").is_file()
    assert calls["device"] == "cpu"
    if quantized:
        assert calls["dequantized_load"] == (
            str(model_dir),
            {"dtype": torch.bfloat16, "device_map": "cpu", "gptq_weight_mode": "dequantize"},
        )
        assert calls["attention"] == "eager"
    if cache_flag is None:
        assert "use_cache" not in calls["generation_kwargs"]
    else:
        assert calls["generation_kwargs"]["use_cache"] is expected_cache
    if backend == "hmonnx":
        assert calls["runtime_load"] == ("metadata.json", {"device_map": ["cpu"]})
        assert result["dtype"] == "float16"
        assert calls["generation_kwargs"]["pixel_values"].dtype == torch.float16
        assert result["use_cache_source"] == "fixed_hmonnx_kv_cache_abi"
    elif cache_flag is None:
        assert result["use_cache_source"] == "checkpoint_generation_config"
    else:
        assert result["use_cache_source"] == "explicit_hf_cache_override"
