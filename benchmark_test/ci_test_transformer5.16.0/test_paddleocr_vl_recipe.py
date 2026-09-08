from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest
import torch

from recipes.paddleocr_vl import (
    PaddleOCRVLCombinedQModel,
    PaddleOCRVLQModel,
    PaddleOCRVLVisionQModel,
    _copy_processor_assets,
    _merge_quantized_stages,
    _pad_symmetric_qzeros,
    preserve_paddleocr_vl_tie_word_embeddings,
    quantize_paddleocr_vl,
    register_paddleocr_vl_gptqmodel,
    rotate_model_paddleocr_vl,
    rotate_vision_attention_vo,
)


ROOT = Path(__file__).resolve().parents[2]
FULL_PADDLEOCR_VL = ROOT / "/data01/datasets/PaddleOCR-VL-1.6"


@pytest.mark.transformer516
def test_paddleocr_vl_pads_symmetric_qzeros_for_partial_output_word() -> None:
    class Packed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sym = True
            self.pack_factor = 4
            self.register_buffer("qzeros", torch.full((18, 1072), 0x7F7F7F7F, dtype=torch.int32))
            self.register_buffer("scales", torch.ones((18, 4304)))

    model = torch.nn.Sequential(Packed())
    assert _pad_symmetric_qzeros(model) == 1
    assert model[0].qzeros.shape == (18, 1076)
    assert torch.equal(model[0].qzeros[:, -1], model[0].qzeros[:, -2])


@pytest.mark.transformer516
def test_paddleocr_vl_recipe_signature_and_dry_run(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "paddleocr_vl", "tie_word_embeddings": False}),
        encoding="utf-8",
    )
    cases = [
        {
            "input_ids": [[1, 100295, 2]],
            "attention_mask": [[1, 1, 1]],
            "pixel_values": [[[0.0]]],
            "image_grid_thw": [[1, 1, 1]],
        }
    ]

    signature = inspect.signature(quantize_paddleocr_vl)
    signature.bind(
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "out"),
        calibration_data=cases,
        method="autoround",
        bits=8,
        group_size=64,
        iters=4,
        rotation=None,
        device="cuda:0",
        seed=42,
        dry_run=True,
    )
    result = quantize_paddleocr_vl(
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "out"),
        calibration_data=cases,
        method="auto_round",
        iters=4,
        dry_run=True,
    )
    rotated_result = quantize_paddleocr_vl(
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "out_rotated"),
        calibration_data=cases,
        method="autoround",
        rotation="hadamard",
        dry_run=True,
    )
    combined_result = quantize_paddleocr_vl(
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "out_combined"),
        calibration_data=cases,
        method="autoround",
        rotation="hadamard",
        quantize_vision=True,
        vision_rotation="hadamard",
        dry_run=True,
    )

    assert result.output_dir == str((tmp_path / "out").resolve())
    assert result.method == "autoround"
    assert result.artifact_format == "gptqmodel_hf"
    assert result.provenance["explicit_untied_embeddings"] is True
    assert result.provenance["offload_to_disk"] is False
    assert result.provenance["offload_to_disk_path"] is None
    assert result.precision["scope"] == "language_model"
    assert result.precision["visual_modules"] == "float"
    assert result.precision["vision_patch_embedding"] == "float"
    assert result.precision["projector_modules"] == "float"
    assert rotated_result.precision["language_rotation"] == "hadamard"
    assert rotated_result.provenance["rotation_contract"]["scope"] == "offline_r1_hidden_basis"
    assert rotated_result.provenance["rotation_contract"]["runtime_wrapper_required"] is False
    assert combined_result.precision["scope"] == "language_model+vision_encoder"
    assert combined_result.precision["visual_modules"]["bits"] == 8
    assert combined_result.precision["visual_modules"]["rotation"] == "hadamard"
    assert (
        combined_result.provenance["vision_rotation_contract"]["scope"]
        == "offline_per_head_attention_v_o_basis"
    )


@pytest.mark.transformer516
def test_paddleocr_vl_rejects_text_only_and_unknown_rotation(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "paddleocr_vl"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pure text calibration"):
        quantize_paddleocr_vl(
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "out"),
            calibration_data=[{"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}],
            dry_run=True,
        )

    multimodal_cases = [
        {
            "input_ids": [[1, 100295, 2]],
            "pixel_values": [[[0.0]]],
            "image_grid_thw": [[1, 1, 1]],
        }
    ]

    with pytest.raises(ValueError, match="rotation must be None or 'hadamard'"):
        quantize_paddleocr_vl(
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "out"),
            calibration_data=multimodal_cases,
            rotation="random",
            dry_run=True,
        )

    with pytest.raises(ValueError, match="offline R1 rotation currently requires offload_to_disk=False"):
        quantize_paddleocr_vl(
            model_dir=str(model_dir),
            output_dir=str(tmp_path / "out"),
            calibration_data=multimodal_cases,
            rotation="hadamard",
            offload_to_disk=True,
            dry_run=True,
        )


@pytest.mark.transformer516
def test_paddleocr_vl_registers_combined_language_and_vision_module_trees() -> None:
    registered = register_paddleocr_vl_gptqmodel()

    import gptqmodel.models.auto as auto

    assert auto.MODEL_MAP["paddleocr_vl"] is registered
    assert "paddleocr_vl" in auto.SUPPORTED_MODELS
    assert registered is PaddleOCRVLCombinedQModel
    assert registered.extract_layers_node() == [
        "model.language_model.layers",
        "model.visual.vision_model.encoder.layers",
    ]
    from gptqmodel.quantization.rotation.model import MODEL_ROTATION_REGISTRY

    rotation_entry = next(
        entry for entry in MODEL_ROTATION_REGISTRY if entry.get("cls") is PaddleOCRVLQModel
    )
    assert rotation_entry["vision"] is False
    assert rotation_entry["rotate_fn"] is rotate_model_paddleocr_vl

    modules = registered.simple_layer_modules(
        _DummyConfig(),
        _DummyQuantizeConfig(),
    )
    flattened = [name for group in modules for name in group]
    assert "self_attn.q_proj" in flattened
    assert "self_attn.o_proj" in flattened
    assert "mlp.gate_proj" in flattened
    assert "mlp.down_proj" in flattened
    assert "self_attn.out_proj" in flattened
    assert "mlp.fc1" in flattened
    assert "mlp.fc2" in flattened

    assert register_paddleocr_vl_gptqmodel(stage="language") is PaddleOCRVLQModel
    assert register_paddleocr_vl_gptqmodel(stage="vision") is PaddleOCRVLVisionQModel
    assert PaddleOCRVLVisionQModel.module_tree[:6] == [
        "model", "visual", "vision_model", "encoder", "layers", "#",
    ]


@pytest.mark.transformer516
def test_paddleocr_vl_native_minimal_model_shape_and_tie_patch() -> None:
    pytest.importorskip("transformers.models.paddleocr_vl")
    from transformers.models.paddleocr_vl.configuration_paddleocr_vl import (
        PaddleOCRVLConfig,
    )
    from transformers.models.paddleocr_vl.modeling_paddleocr_vl import (
        PaddleOCRVLForConditionalGeneration,
    )

    with preserve_paddleocr_vl_tie_word_embeddings(False):
        config = PaddleOCRVLConfig(
            text_config={
                "vocab_size": 128,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "max_position_embeddings": 128,
                "tie_word_embeddings": True,
            },
            vision_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_channels": 3,
                "image_size": 14,
                "patch_size": 14,
                "spatial_merge_size": 1,
            },
            tie_word_embeddings=True,
        )
    model = PaddleOCRVLForConditionalGeneration(config)
    loader_model = PaddleOCRVLQModel.loader.from_config(
        config,
        trust_remote_code=False,
        dtype=torch.float32,
    )
    PaddleOCRVLQModel._select_module_tree(model)

    assert config.tie_word_embeddings is False
    assert config.text_config.tie_word_embeddings is False
    assert PaddleOCRVLQModel.module_tree[:4] == ["model", "language_model", "layers", "#"]
    assert model.model.language_model.layers[0].self_attn.q_proj.weight.shape == torch.Size([32, 16])
    assert model.model.language_model.layers[0].self_attn.o_proj.weight.shape == torch.Size([16, 32])
    assert loader_model.model.language_model.layers[0].self_attn.q_proj.weight.dtype is torch.float32
    assert hasattr(model.model, "visual")
    assert hasattr(model.model, "projector")


@pytest.mark.transformer516
def test_paddleocr_vl_offline_r1_rotation_save_load_preserves_multimodal_logits(tmp_path: Path) -> None:
    pytest.importorskip("transformers.models.paddleocr_vl")
    from transformers.models.paddleocr_vl.modeling_paddleocr_vl import (
        PaddleOCRVLForConditionalGeneration,
    )

    from gptqmodel.quantization.rotation.rotation import fuse_layer_norms, get_orthogonal_matrix

    torch.manual_seed(7)
    base_model = PaddleOCRVLForConditionalGeneration(_minimal_native_config()).eval()
    _make_language_norms_nontrivial(base_model)
    with torch.no_grad():
        base_model.model.projector.linear_2.bias.copy_(torch.linspace(-0.2, 0.2, 16))
        base_model.lm_head.weight.add_(torch.linspace(-0.05, 0.05, 128).unsqueeze(1))
    rotated_model = copy.deepcopy(base_model).eval()

    assert _untied_lm_head(rotated_model)
    assert not torch.allclose(
        base_model.model.language_model.layers[0].input_layernorm.weight,
        torch.ones_like(base_model.model.language_model.layers[0].input_layernorm.weight),
    )
    assert not torch.allclose(
        base_model.model.language_model.norm.weight,
        torch.ones_like(base_model.model.language_model.norm.weight),
    )

    input_ids = torch.tensor([[1, 42, 2]])
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": torch.randn(1, 3, 14, 14),
        "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 1, 0]], dtype=torch.long),
        "use_cache": False,
    }

    with torch.inference_mode():
        expected = base_model(**inputs).logits

    projector_bias_before = rotated_model.model.projector.linear_2.bias.detach().clone()
    q_block = get_orthogonal_matrix(16, "hadamard", torch.device("cpu"))
    fuse_layer_norms(
        rotated_model,
        pre_lm_head_norm_module_name="model.language_model.norm",
        layers_node="model.language_model.layers",
        lm_head_name="lm_head",
    )
    _, returned_q = rotate_model_paddleocr_vl(
        rotated_model,
        rotate_mode="hadamard",
        device=torch.device("cpu"),
        lm_head_name="lm_head",
        layers_node="model.language_model.layers",
        Q=q_block,
    )

    with torch.inference_mode():
        actual = rotated_model(**inputs).logits

    assert returned_q is q_block
    assert rotated_model.model.language_model.layers[0].self_attn.q_proj.weight.shape == torch.Size([32, 16])
    assert rotated_model.model.language_model.layers[0].self_attn.o_proj.weight.shape == torch.Size([16, 32])
    assert not torch.equal(rotated_model.model.projector.linear_2.bias, projector_bias_before)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    save_dir = tmp_path / "rotated_native"
    rotated_model.save_pretrained(save_dir)
    reloaded = PaddleOCRVLForConditionalGeneration.from_pretrained(
        save_dir,
        local_files_only=True,
    ).eval()

    with torch.inference_mode():
        reloaded_logits = reloaded(**inputs).logits

    assert reloaded.config.tie_word_embeddings is False
    assert reloaded.config.text_config.tie_word_embeddings is False
    assert _untied_lm_head(reloaded)
    torch.testing.assert_close(reloaded_logits, actual, atol=0, rtol=0)


@pytest.mark.transformer516
def test_paddleocr_vl_vision_vo_rotation_preserves_logits_and_qk_rope() -> None:
    pytest.importorskip("transformers.models.paddleocr_vl")
    from transformers.models.paddleocr_vl.modeling_paddleocr_vl import (
        PaddleOCRVLForConditionalGeneration,
    )

    torch.manual_seed(11)
    base_model = PaddleOCRVLForConditionalGeneration(_minimal_native_config()).eval()
    rotated_model = copy.deepcopy(base_model).eval()
    attention = rotated_model.model.visual.vision_model.encoder.layers[0].self_attn
    q_before = attention.q_proj.weight.detach().clone()
    k_before = attention.k_proj.weight.detach().clone()
    v_before = attention.v_proj.weight.detach().clone()
    out_before = attention.out_proj.weight.detach().clone()
    rotary_before = {
        name: value.detach().clone()
        for name, value in rotated_model.model.visual.vision_model.encoder.rotary_pos_emb.state_dict().items()
    }

    input_ids = torch.tensor([[1, 42, 2]])
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": torch.randn(1, 3, 14, 14),
        "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 1, 0]], dtype=torch.long),
        "use_cache": False,
    }
    with torch.inference_mode():
        expected = base_model(**inputs).logits

    matrices = rotate_vision_attention_vo(rotated_model, "hadamard", torch.device("cpu"))
    with torch.inference_mode():
        actual = rotated_model(**inputs).logits

    assert matrices[0].shape == torch.Size([8, 8])
    assert torch.equal(attention.q_proj.weight, q_before)
    assert torch.equal(attention.k_proj.weight, k_before)
    assert not torch.equal(attention.v_proj.weight, v_before)
    assert not torch.equal(attention.out_proj.weight, out_before)
    for name, value in rotated_model.model.visual.vision_model.encoder.rotary_pos_emb.state_dict().items():
        assert torch.equal(value, rotary_before[name])
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.transformer516
def test_paddleocr_vl_stage_merge_replaces_float_vision_weights(tmp_path: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    language_dir = tmp_path / "language"
    vision_dir = tmp_path / "vision"
    output_dir = tmp_path / "merged"
    language_dir.mkdir()
    vision_dir.mkdir()
    prefix = "model.visual.vision_model.encoder.layers.0.self_attn.v_proj"
    save_file(
        {
            "model.language_model.layers.0.self_attn.q_proj.qweight": torch.ones(2, 2),
            f"{prefix}.weight": torch.full((2, 2), 3.0),
            "model.projector.linear_1.weight": torch.full((2, 2), 4.0),
        },
        language_dir / "model.safetensors",
    )
    save_file(
        {
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 5.0),
            f"{prefix}.qweight": torch.full((2, 2), 6.0),
            f"{prefix}.scales": torch.full((2, 1), 7.0),
            "model.projector.linear_1.weight": torch.full((2, 2), 8.0),
        },
        vision_dir / "model.safetensors",
    )
    (language_dir / "config.json").write_text('{"model_type":"paddleocr_vl"}\n', encoding="utf-8")
    (language_dir / "quantize_config.json").write_text('{"bits":8}\n', encoding="utf-8")

    _merge_quantized_stages(language_dir, vision_dir, output_dir, max_shard_size=32)

    index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    keys = set(index["weight_map"])
    assert f"{prefix}.weight" not in keys
    assert f"{prefix}.qweight" in keys
    assert "model.language_model.layers.0.self_attn.q_proj.qweight" in keys
    assert "model.language_model.layers.0.self_attn.q_proj.weight" not in keys
    assert (output_dir / "quantize_config.json").read_text(encoding="utf-8") == '{"bits":8}\n'
    projector_file = output_dir / index["weight_map"]["model.projector.linear_1.weight"]
    with safe_open(projector_file, framework="pt", device="cpu") as stream:
        assert torch.equal(stream.get_tensor("model.projector.linear_1.weight"), torch.full((2, 2), 4.0))


@pytest.mark.transformer516
def test_paddleocr_vl_quantize_config_rotation_contract_round_trips(tmp_path: Path) -> None:
    from gptqmodel import QuantizeConfig
    from gptqmodel.quantization.config import FORMAT, METHOD

    result = quantize_paddleocr_vl(
        model_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        calibration_data=[
            {
                "input_ids": [[1, 42, 2]],
                "pixel_values": [[[[0.0]]]],
                "image_grid_thw": [[1, 1, 1]],
            }
        ],
        rotation="hadamard",
        dry_run=True,
    )
    contract = result.provenance["rotation_contract"]
    qcfg = QuantizeConfig(
        bits=8,
        group_size=64,
        quant_method=METHOD.AUTO_ROUND,
        format=FORMAT.GPTQ,
        rotation="hadamard",
        offload_to_disk=False,
        meta={"rotation_contract": contract},
    )

    save_dir = tmp_path / "qcfg"
    save_dir.mkdir()
    qcfg.save_pretrained(save_dir)
    restored = QuantizeConfig.from_pretrained(save_dir)

    assert qcfg.to_dict()["meta"]["rotation_contract"] == contract
    assert restored.meta["rotation_contract"] == contract
    assert restored.meta["rotation_contract"]["runtime_wrapper_required"] is False


@pytest.mark.transformer516
def test_paddleocr_vl_register_does_not_leak_tie_config_patch() -> None:
    pytest.importorskip("transformers.models.paddleocr_vl")
    from transformers.models.paddleocr_vl.configuration_paddleocr_vl import (
        PaddleOCRVLConfig,
    )

    before = PaddleOCRVLConfig(
        text_config={"tie_word_embeddings": True},
        tie_word_embeddings=True,
    )
    register_paddleocr_vl_gptqmodel()
    after_register = PaddleOCRVLConfig(
        text_config={"tie_word_embeddings": True},
        tie_word_embeddings=True,
    )
    with preserve_paddleocr_vl_tie_word_embeddings(False):
        inside_patch = PaddleOCRVLConfig(
            text_config={"tie_word_embeddings": True},
            tie_word_embeddings=True,
        )
    after_patch = PaddleOCRVLConfig(
        text_config={"tie_word_embeddings": True},
        tie_word_embeddings=True,
    )

    assert before.tie_word_embeddings is True
    assert before.text_config.tie_word_embeddings is True
    assert after_register.tie_word_embeddings is True
    assert after_register.text_config.tie_word_embeddings is True
    assert inside_patch.tie_word_embeddings is False
    assert inside_patch.text_config.tie_word_embeddings is False
    assert after_patch.tie_word_embeddings is True
    assert after_patch.text_config.tie_word_embeddings is True


@pytest.mark.transformer516
def test_paddleocr_vl_processor_asset_copy_preserves_quantized_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "tiny_source"
    output = tmp_path / "out_gptq"
    source.mkdir()
    output.mkdir()
    (source / "config.json").write_text('{"model_type": "paddleocr_vl"}\n', encoding="utf-8")
    (output / "config.json").write_text("quantized config\n", encoding="utf-8")
    (output / "model.safetensors").write_bytes(b"quantized weights")

    provenance = _copy_processor_assets(source, output)

    assert provenance["status"] == "source_missing"
    assert provenance["files"] == []
    assert (output / "config.json").read_text(encoding="utf-8") == "quantized config\n"
    assert (output / "model.safetensors").read_bytes() == b"quantized weights"


@pytest.mark.transformer516
@pytest.mark.skipif(
    not FULL_PADDLEOCR_VL.is_dir(),
    reason="local PaddleOCR-VL processor assets are not available",
)
def test_paddleocr_vl_copied_processor_assets_load_real_multimodal_inputs(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image
    from transformers import AutoProcessor

    output = tmp_path / "out_gptq"
    output.mkdir()
    (output / "config.json").write_text(
        json.dumps({"model_type": "paddleocr_vl", "sentinel": "keep"}) + "\n",
        encoding="utf-8",
    )
    (output / "quantize_config.json").write_text('{"bits": 8}\n', encoding="utf-8")
    (output / "model.safetensors").write_bytes(b"packed weights")

    provenance = _copy_processor_assets(FULL_PADDLEOCR_VL, output)

    copied = {item["name"]: item for item in provenance["files"]}
    assert provenance["status"] == "copied"
    assert provenance["source_fingerprint_sha256"]
    assert provenance["source_revision"]
    assert "preprocessor_config.json" in copied
    assert "processor_config.json" in copied
    assert "tokenizer.model" in copied
    assert not any(name.endswith(".py") for name in copied)
    assert json.loads((output / "config.json").read_text(encoding="utf-8"))["sentinel"] == "keep"
    assert (output / "quantize_config.json").read_text(encoding="utf-8") == '{"bits": 8}\n'
    assert (output / "model.safetensors").read_bytes() == b"packed weights"

    processor = AutoProcessor.from_pretrained(output, trust_remote_code=False)
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "OCR:"}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[Image.new("RGB", (28, 28), "white")],
        return_tensors="pt",
    )

    assert inputs["input_ids"].ndim == 2
    assert inputs["pixel_values"].shape[-3:] == (3, 14, 14)
    assert inputs["image_grid_thw"].shape == torch.Size([1, 3])
    assert inputs["mm_token_type_ids"].shape == inputs["input_ids"].shape


class _DummyConfig:
    pass


class _DummyQuantizeConfig:
    dynamic = None


def _make_language_norms_nontrivial(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for layer in model.model.language_model.layers:
            layer.input_layernorm.weight.copy_(torch.linspace(0.5, 1.5, 16))
            layer.post_attention_layernorm.weight.copy_(torch.linspace(1.5, 0.5, 16))
        model.model.language_model.norm.weight.copy_(torch.linspace(0.75, 1.25, 16))


def _untied_lm_head(model: torch.nn.Module) -> bool:
    return (
        model.lm_head.weight.data_ptr()
        != model.model.language_model.embed_tokens.weight.data_ptr()
    )


def _minimal_native_config():
    from transformers.models.paddleocr_vl.configuration_paddleocr_vl import (
        PaddleOCRVLConfig,
    )

    return PaddleOCRVLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "max_position_embeddings": 128,
            "tie_word_embeddings": False,
            "rope_parameters": {"mrope_section": [2, 3, 3]},
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_channels": 3,
            "image_size": 14,
            "patch_size": 14,
            "spatial_merge_size": 1,
        },
        tie_word_embeddings=False,
        image_token_id=42,
    )
