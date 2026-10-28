# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Small real-model semantics and public contracts; full checkpoints run in demos."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.model_build import describe_model, list_models
from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_inference.inputs import validate_inputs
from xhmodel_merak.xh_llm.ocr_runtime import last_nonpad_index, prefill_logits
from xhmodel_merak.xh_llm.models.ocr_rm.processing import build_prompt, load_image


@pytest.mark.parametrize(
    "model_id,model_type,hidden",
    [
        ("minicpm-ocr-1.0", "MiniCPMOcrForConditionalGeneration", 1024),
        ("ocr-rm-2b-value", "OcrRewardForSequenceClassification", 2048),
    ],
)
def test_public_recipe_and_real_workflow(model_id, model_type, hidden, tmp_path):
    import yaml

    assert model_id in {item["model_id"] for item in list_models()}
    recipe = get_recipe(model_id)
    assert dict(recipe.source_constraints)["text_config.hidden_size"] == hidden
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / recipe.variants[0].workflow_config).read_text())
    assert workflow["export"]["model"]["model_type"] == model_type
    assert workflow["quant"] is None
    model = workflow["export"]["model"]
    assert model["fuse_gdr_ops"] is True
    assert model["fuse_gdr_block_recurrent_ops"] is True
    assert model["normalize_force_fp32"] is True
    assert model["cumsum_matmul_quant_config"]["w_schema"]["man_bit"] == 8
    assert model["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
    assert model["visual_config"]["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
    assert recipe.components == ("visual", "prefill", "decode")
    assert describe_model(model_id)
    # Reject the other model's source identity before loading any weights.
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "wrong"}))
    with pytest.raises(Exception, match="Registry identity"):
        resolve_build_plan(model_id=model_id, model_dir=str(source), output_dir=str(tmp_path / "out"))


def test_reward_prompt_and_image_order_contract(tmp_path):
    from PIL import Image

    prompt = build_prompt("a\nb")
    assert prompt.startswith("<|im_start|>user\n<|vision_start|>")
    assert "<|vision_end|>OCR文本：\na\nb\n<|vision_start|>" in prompt
    assert prompt.endswith("一致标为 1，不一致标为 0。<|im_end|>\n")
    assert "assistant" not in prompt and "system" not in prompt
    path = tmp_path / "wide.png"
    Image.new("RGB", (400, 1), "black").save(path)
    padded = load_image(path)
    assert padded.size == (400, 3)
    assert padded.getpixel((0, 0)) == (255, 255, 255)
    assert padded.getpixel((0, 1)) == (0, 0, 0)


@pytest.mark.parametrize("quant_type", ["w8a8h0_sefp", "w16a16h0_sefp"])
def test_reward_rejects_forbidden_h0_deployment(quant_type):
    from xhmodel_merak.model_catalog.precision import resolve_deployment_precision

    recipe = get_recipe("ocr-rm-2b-value")
    with pytest.raises(ValueError, match="require a quant_type"):
        resolve_deployment_precision(recipe, {"prefill": quant_type})


@pytest.mark.parametrize("location", ["base", "visual", "node", "cumsum"])
def test_reward_workflow_rejects_hidden_h0_overrides(location):
    from xhmodel_merak.xh_llm.models.ocr_rm.precision import validate_sefp_h1_config

    scheme = {"quant_type": "w8a8h1_sefp"}
    model = {"quant_scheme": dict(scheme), "visual_config": {"quant_scheme": dict(scheme)}}
    if location == "base":
        model["quant_scheme"]["quant_type"] = "w16a16h0_sefp"
    elif location == "visual":
        model["visual_config"]["quant_scheme"]["quant_type"] = "w8a8h0_sefp"
    elif location == "node":
        model["quant_scheme"]["nodes"] = {"lm_head": "w8a8h0_sefp"}
    else:
        model["cumsum_matmul_quant_config"] = {"w_schema": {"fp_mode": "sefp", "hidden_bit": False}}
    with pytest.raises(ValueError, match="forbids SEFP h0"):
        validate_sefp_h1_config(model)


def test_input_contracts_reject_generation_on_reward_model():
    from xhmodel_merak.errors import ModelError

    recipe = get_recipe("ocr-rm-2b-value")
    assert validate_inputs(recipe, {"orig_image": "a", "render_image": "b", "ocr_text": ""})["ocr_text"] == ""
    with pytest.raises(ModelError):
        validate_inputs(recipe, {"orig_image": "a", "render_image": "b", "ocr_text": "x", "max_new_tokens": 1})
    with pytest.raises(ModelError):
        validate_inputs(get_recipe("minicpm-ocr-1.0"), {"image": "a", "max_new_tokens": 0})


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_token_gears", [96]),
        ("image_token_capacity", True),
        ("max_size_w", 224),
    ],
)
def test_reward_visual_planning_rejects_gear_drift(field, value):
    import yaml
    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.model_integrations.ocr_rm.planning import resolve_config

    recipe = get_recipe("ocr-rm-2b-value")
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / recipe.variants[0].workflow_config).read_text())
    assert resolve_config(recipe=recipe, workflow=workflow) == {}
    workflow["export"]["model"]["visual_config"][field] = value
    with pytest.raises(ModelError, match="registered gears"):
        resolve_config(recipe=recipe, workflow=workflow)


@pytest.mark.parametrize("ids,pad,expected", [([[1, 2, 0, 0]], 0, 1), ([[1, 0, 3, 0]], 0, 2)])
def test_last_nonpad_pooling(ids, pad, expected):
    assert last_nonpad_index(torch.tensor(ids), pad) == expected
    with pytest.raises(ValueError):
        last_nonpad_index(torch.zeros((1, 4), dtype=torch.long), 0)


def test_real_hybrid_reward_head_matches_native(restore_model_initialization_state, monkeypatch):
    from inspect import unwrap

    from transformers import Qwen3_5Config, Qwen3_5ForSequenceClassification
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from xhmodel_merak.xh_llm.models.ocr_rm.model import classification_shell

    # This is a CPU reference check. Optional FLA/causal-conv kernels target
    # CUDA, so use Transformers' own PyTorch implementations for this test.
    for name in (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        monkeypatch.setattr(modeling_qwen3_5, name, unwrap(getattr(modeling_qwen3_5, name)))

    torch.manual_seed(17)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "partial_rotary_factor": 0.5,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 0],
            },
            "pad_token_id": 0,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "out_hidden_size": 32,
            "num_heads": 4,
            "depth": 1,
            "num_position_embeddings": 16,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        pad_token_id=0,
        num_labels=2,
        tie_word_embeddings=False,
    )
    with restore_model_initialization_state():
        native = Qwen3_5ForSequenceClassification(config).float().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        expected = native(input_ids=ids, use_cache=False).logits
        head = native.score
        embedding = native.model.language_model.embed_tokens
        shell = classification_shell(native)
        actual = shell(input_ids=ids, use_cache=False).logits[:, -1]
    assert shell.lm_head is head
    assert shell.model.language_model.embed_tokens is embedding
    assert shell.lm_head.weight.shape == (2, 32)
    assert shell.config.text_config.vocab_size == 64
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_ocr_4x_native_vision_matches_static_capacity(restore_model_initialization_state):
    from xhmodel_merak.xh_llm.models.minicpm_ocr.native import (
        MiniCPMOcrConfig,
        MiniCPMOcrNativeModel,
        MiniCPMOcrImageProcessor,
    )
    from xhmodel_merak.xh_llm.minicpm_v_series.token_capacity_vision import (
        build_token_capacity_vision_export_model,
        prepare_token_capacity_vision_inputs,
        crop_token_capacity_image_embeds,
    )

    raw = json.loads((Path(__file__).parent / "fixtures/unified/minicpm_ocr.json").read_text())["config"]
    with restore_model_initialization_state():
        torch.manual_seed(41)
        native = MiniCPMOcrNativeModel(MiniCPMOcrConfig(**raw)).float().eval()
    pixels = torch.randn(1, 3, 14, 16 * 14)
    target = torch.tensor([[4, 4]], dtype=torch.int32)
    wrapper = build_token_capacity_vision_export_model(native, 32, "4x")
    inputs = prepare_token_capacity_vision_inputs(pixels, (4, 4), 32, "4x", positions_per_side=4)
    with torch.inference_mode():
        expected = torch.cat(list(native.get_image_features(pixels, target, downsample_mode="4x").pooler_output))
        actual = crop_token_capacity_image_embeds(wrapper(*inputs[:3]), (4, 4), 32, "4x").squeeze(0)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    processor = MiniCPMOcrImageProcessor(downsample_mode="4x")
    # A two-patch-wide edge must remain two patches, rather than being rounded
    # to the older processor's four-patch grid. Preserve rectangular slicing.
    assert processor.find_best_resize((28, 28), 448, 14) == (28, 28)
    assert processor.get_sliced_grid((448, 448 * 4), 9, 448) == [1, 4]


def test_chunked_prefill_preserves_full_multimodal_positions():
    # Only the expensive graph/session boundary is substituted. The behavior
    # under test is host chunk scheduling, image scattering and position slicing.
    calls = []
    embedding = torch.nn.Embedding(20, 3)

    class Processor:
        def __call__(self, data):
            return (
                torch.nn.functional.pad(data["inputs_embeds"], (0, 0, 0, 4 - data["inputs_embeds"].shape[1])),
                torch.zeros(4),
                torch.zeros(4),
                torch.zeros(4),
                data["past_seq_length"],
            )

    class Model:
        device = torch.device("cpu")
        embed_tokens = embedding
        meta_info = SimpleNamespace(
            model_config=SimpleNamespace(context_max_length=16, prefill_chunk_length=4, image_token_id=19)
        )
        prefill_model = SimpleNamespace(
            hmonnx_session=SimpleNamespace(get_input=lambda name: SimpleNamespace(shape=(4,)))
        )

        def set_prefill(self):
            pass

        def get_data_preprocessor(self):
            return Processor()

        def forward(self, *args):
            calls.append(args)
            return (args[0][:, -1:],)

    ids = torch.tensor([[1, 19, 19, 19, 19, 19, 2]])
    positions = torch.tensor([[[0, 1, 1, 1, 1, 1, 4]], [[0, 1, 1, 2, 2, 3, 4]], [[0, 1, 2, 1, 2, 1, 4]]])
    features = torch.arange(15).reshape(5, 3).float()
    prefill_logits(Model(), ids, image_embeds=features, position_ids=positions)
    assert len(calls) == 2 and calls[1][-1] == 4
    torch.testing.assert_close(calls[0][0][0, 1:4], features[:3])
    torch.testing.assert_close(calls[1][0][0, :2], features[3:])
    for axis in range(3):
        torch.testing.assert_close(calls[1][axis + 1][:3], positions[axis, 0, 4:])
    with pytest.raises(ValueError, match="placeholder"):
        prefill_logits(Model(), ids, image_embeds=features[:1])


def test_ocr_artifact_moves_and_retains_only_runtime_files(tmp_path):
    import shutil
    import onnx
    from onnx import helper, TensorProto
    from xhmodel_merak.model_artifact.files import confined_file
    from xhmodel_merak.model_integrations.minicpm_ocr.build import describe_artifact, sanitize_artifact

    root = tmp_path / "stage"
    root.mkdir()
    for component in ("visual_m384", "prefill", "decode"):
        folder = root / component
        folder.mkdir()
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            component,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        )
        onnx.save(helper.make_model(graph), folder / "model.onnx")
    (root / "hf_config").mkdir()
    (root / "hf_config/config.json").write_text('{"model_type": "minicpm_ocr"}')
    (root / "hf_config/tokenizer.json").write_text("{}")
    (root / "quant_embedding.pt").write_bytes(b"fixture")
    (root / "effective_workflow.yaml").write_text("quant: null\n")
    (root / "export.log").write_text("not a runtime dependency")
    child = {
        "model_config": {"hf_model": "/deleted/source"},
        "hf_config": "hf_config",
        "prefill_hmonnx": "prefill/model.onnx",
        "decode_hmonnx": "decode/model.onnx",
        "quant_embedding": "quant_embedding.pt",
    }
    (root / "golden_meta_info.json").write_text(json.dumps(child))
    meta = {
        "schema_version": 1,
        "model_type": "minicpm_ocr",
        "downsample_mode": "4x",
        "components": ["visual", "prefill", "decode"],
        "text_metadata": "golden_meta_info.json",
        "hf_config": "hf_config",
        "visual": {"hmonnx": "visual_m384/model.onnx"},
    }
    (root / "runtime_meta_info.json").write_text(json.dumps(meta))
    plan = SimpleNamespace(components=("visual", "prefill", "decode"))
    sanitize_artifact(root, plan)
    assert not (root / "export.log").exists()
    assert json.loads((root / "golden_meta_info.json").read_text())["model_config"]["hf_model"] == "hf_config"
    moved = tmp_path / "relocated"
    shutil.move(root, moved)
    entries = describe_artifact(moved, lambda value, field: confined_file(moved, value, field), plan.components)
    assert entries["visual"] == {"path": "visual_m384/model.onnx", "type": "hmonnx"}
    meta["visual"]["hmonnx"] = "../outside.onnx"
    (moved / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(Exception):
        describe_artifact(moved, lambda value, field: confined_file(moved, value, field), plan.components)
