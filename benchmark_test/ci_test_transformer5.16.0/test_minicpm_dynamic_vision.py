"""Native MiniCPM ViTs with Qwen-style padding masks through real HMONNX."""

from copy import deepcopy
import importlib
from types import SimpleNamespace

from _minicpm_v47_fixture import minicpmv47_modeling  # noqa: F401

import pytest
import torch
from PIL import Image

from xhmodel_merak.xh_llm.minicpm_v_series.token_capacity_vision import (
    MiniCPMTokenCapacityVisionExportModel,
    crop_token_capacity_image_embeds,
    prepare_token_capacity_vision_inputs,
)
from xhmodel_merak.xh_llm.minicpm_v_series.token_capacity_vision_export import (
    _convert_hmonnx,
    _hmonnx_inputs,
    vision_input_names,
)


def _native_parts(family):
    module = importlib.import_module(f"transformers.models.minicpmv{family}.modeling_minicpmv{family}")
    configuration = importlib.import_module(f"transformers.models.minicpmv{family}.configuration_minicpmv{family}")
    config_cls = getattr(configuration, f"MiniCPMV{family}VisionConfig")
    config = config_cls(
        hidden_size=144,
        intermediate_size=288,
        num_hidden_layers=2,
        num_attention_heads=2,  # Preserve the real MiniCPM ViT's 72-wide heads.
        image_size=112,
        patch_size=14,
        insert_layer_id=0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(47)
    vision = getattr(module, f"MiniCPMV{family}VisionModel")(config).eval()
    merger = getattr(module, f"MiniCPMV{family}Merger")(
        SimpleNamespace(
            vision_config=config, text_config=SimpleNamespace(hidden_size=64), merge_kernel_size=(2, 2), merger_times=1
        )
    ).eval()
    return vision, merger


def _native_features(vision, merger, pixels, grid, mode):
    sizes = torch.tensor([grid], dtype=torch.int32)
    hidden = vision(pixels.float(), target_sizes=sizes, use_vit_merger=mode == "16x").last_hidden_state
    return torch.cat(merger(hidden, sizes // 2 if mode == "16x" else sizes)).unsqueeze(0)


def _run_vision(session, inputs):
    output = session(*inputs)
    return output[0] if isinstance(output, (tuple, list)) else output


def _assert_padding_probabilities_zero(captured, count, mode):
    stages = set()
    for probabilities in captured:
        capacity = probabilities.shape[-1]
        if capacity == 64:
            valid = count
        elif mode == "16x" and capacity == 16:
            valid = count // 4
        else:
            continue  # local 2x2 merger windows are wholly valid or padded
        stages.add(capacity)
        assert torch.count_nonzero(probabilities[..., valid:]) == 0
    assert stages == ({64, 16} if mode == "16x" else {64})


@pytest.mark.parametrize("mode", ("4x", "16x"))
@pytest.mark.parametrize("grid", ((4, 4), (4, 8), (8, 4), (8, 8)))
def test_dynamic_vision_padding_formula_and_finite_fp16_mask(mode, grid):
    capacity = 64
    count = grid[0] * grid[1]
    pixels = torch.randn(1, 3, 14, count * 14)
    inputs = prepare_token_capacity_vision_inputs(pixels, grid, capacity, mode, 8)
    padded = inputs[0].reshape(1, 3, 14, capacity, 14)
    assert padded[:, :, :, count:].count_nonzero() == 0
    assert inputs[1].shape == (1, capacity)
    for bias, valid, length in ((inputs[2], count, capacity), (inputs[3], count // 4, capacity // 4)):
        assert bias.shape == (1, 1, 1, length)
        assert torch.all(bias[..., :valid] == 0)
        assert torch.all(bias[..., valid:] == torch.finfo(torch.float32).min)
    names = vision_input_names(mode)
    coerced = _hmonnx_inputs(inputs[: len(names)], names)
    for name, value in zip(names, coerced, strict=True):
        if name.endswith("attention_bias"):
            assert torch.isfinite(value).all()
            assert torch.all((value == 0) | (value == -65504))


@pytest.mark.parametrize("family", ("4_6", "4_7"))
@pytest.mark.parametrize("mode", ("4x", "16x"))
@pytest.mark.parametrize("grid", ((4, 4), (4, 8), (8, 4)))
def test_dynamic_vision_matches_native_and_ignores_padded_patches(family, mode, grid):
    torch.set_num_threads(2)
    vision, merger = _native_parts(family)
    wrapper = MiniCPMTokenCapacityVisionExportModel(vision, merger, 64, mode).eval()
    count = grid[0] * grid[1]
    pixels = torch.randn(1, 3, 14, count * 14)
    args = prepare_token_capacity_vision_inputs(pixels, grid, 64, mode, 8)
    names = vision_input_names(mode)
    with torch.inference_mode():
        native = _native_features(vision, merger, pixels, grid, mode)
        actual = crop_token_capacity_image_embeds(wrapper(*args[: len(names)]), grid, 64, mode)
        torch.testing.assert_close(actual, native, rtol=1e-5, atol=1e-5)
        perturbed = [value.clone() for value in args]
        tail = perturbed[0].reshape(1, 3, 14, 64, 14)[:, :, :, count:]
        tail.copy_(torch.randn_like(tail) * 3)
        perturbed[1][:, count:] = 63
        masked = crop_token_capacity_image_embeds(wrapper(*perturbed[: len(names)]), grid, 64, mode)
        torch.testing.assert_close(masked, actual, rtol=0, atol=1e-6)


@pytest.mark.parametrize("family", ("4_6", "4_7"))
@pytest.mark.parametrize("mode", ("4x", "16x"))
def test_dynamic_vision_hmonnx_from_images_masks_both_stages(tmp_path, family, mode):
    import onnx
    from transformers.models.minicpmv4_6.image_processing_pil_minicpmv4_6 import MiniCPMV4_6ImageProcessorPil
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.inference import MiniCPMV47HMONNXRuntime

    torch.set_num_threads(2)
    vision, merger = _native_parts(family)
    wrapper = MiniCPMTokenCapacityVisionExportModel(deepcopy(vision), deepcopy(merger), 64, mode).eval()
    dummy = prepare_token_capacity_vision_inputs(torch.randn(1, 3, 14, 16 * 14), (4, 4), 64, mode, 8)
    names = vision_input_names(mode)
    path = tmp_path / f"vision_{mode}.onnx"
    _convert_hmonnx(wrapper, path, dummy[: len(names)], names, "XH2a", "w8a8h1_sefp")
    graph = onnx.load(path, load_external_data=False)
    assert [value.name for value in graph.graph.input] == list(names)
    assert "Softmax" in {node.op_type for node in graph.graph.node}
    producers = {output: node for node in graph.graph.node for output in node.output}
    mask_stages = set()
    for node in graph.graph.node:
        if node.op_type != "Softmax":
            continue
        second = producers[node.input[0]]
        masks = set(second.input) & {"pre_attention_bias", "post_attention_bias"}
        if masks:
            first = producers[second.input[0]]
            assert first.op_type == second.op_type == "Add"
            mask = masks.pop()
            assert mask in first.input
            assert all(
                onnx.helper.get_attribute_value(next(a for a in add.attribute if a.name == "saturate")) == 1
                for add in (first, second)
            )
            mask_stages.add(mask)
    assert mask_stages == set(names[2:])

    runtime = MiniCPMV47HMONNXRuntime.__new__(MiniCPMV47HMONNXRuntime)
    runtime.meta = {"vision": {mode: {
        "downsample_mode": mode, "hmonnx": path.name, "token_capacity": 64, "positions_per_side": 8
    }}}
    runtime.work_dir = tmp_path
    runtime.device = torch.device("cpu")
    runtime.enable_golden = False
    runtime._vision_sessions = {}
    runtime._vision_golden_steps = {}
    session = runtime._get_vision_session(mode)
    captured = []

    def capture_probabilities(module, args, output):
        captured.append(next(iter(output.values())).detach().clone())

    for module in session.node_modules:
        if type(module).__name__ == "Softmax":
            module.register_forward_hook(capture_probabilities)
    processor = MiniCPMV4_6ImageProcessorPil(scale_resolution=112, max_slice_nums=1)
    grids = set()
    with torch.inference_mode():
        for size in ((57, 113), (113, 57), (112, 112)):
            image = Image.new("RGB", size, (60, 120, 200))
            processed = processor(image, downsample_mode=mode, return_tensors="pt")
            pixels, sizes = processed["pixel_values"], processed["target_sizes"]
            grid_size = tuple(sizes[0].tolist())
            grids.add(grid_size)
            expected = _native_features(vision, merger, pixels, grid_size, mode).squeeze(0)
            captured.clear()
            actual = runtime._extract_features(pixels, sizes, mode)
            assert actual.shape == expected.shape
            cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.flatten(), dim=0)
            assert cosine > 0.98
            count = grid_size[0] * grid_size[1]
            _assert_padding_probabilities_zero(captured, count, mode)

            args = prepare_token_capacity_vision_inputs(pixels.half(), grid_size, 64, mode, 8)
            if count < 64:
                perturbed = [value.clone() for value in args]
                tail = perturbed[0].reshape(1, 3, 14, 64, 14)[:, :, :, count:]
                tail.copy_(torch.randn_like(tail) * 3)
                perturbed[1][:, count:] = 63
                captured.clear()
                masked = _run_vision(session, perturbed[: len(names)])
                _assert_padding_probabilities_zero(captured, count, mode)
                length = count // (16 if mode == "16x" else 4)
                # SEFP groups can round valid QK scores differently when
                # padded keys change. Check masking directly and retain the
                # native-output accuracy check rather than demand bit equality.
                cosine = torch.nn.functional.cosine_similarity(
                    masked[:, :length].float().flatten(), expected.flatten(), dim=0
                )
                assert cosine > 0.98
                unmasked = list(perturbed)
                index = 3 if mode == "16x" else 2
                unmasked[index] = torch.zeros_like(unmasked[index])
                captured.clear()
                _run_vision(session, unmasked[: len(names)])
                stage_length = 16 if mode == "16x" else 64
                valid = count // 4 if mode == "16x" else count
                assert any(
                    torch.count_nonzero(value[..., valid:]) > 0 for value in captured if value.shape[-1] == stage_length
                )
    assert len(grids) == 3
