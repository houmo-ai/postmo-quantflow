"""Real residual VAE boundaries under one canvas and changing request geometry."""

import pytest
import torch
from torch.nn import functional as F

from xhmodel_merak.xh_other_model.models.qwen_image21.canvas_vae import (
    CanvasAttention,
    CanvasVaeDecode,
    CanvasVaeEncode,
    CanvasVaeSession,
    spatial_inputs,
)
from xhmodel_merak.xh_other_model.models.qwen_image21.capacity import vae_token_capacity


@pytest.fixture(scope="module")
def canvas():
    height, width = 96, 96
    return (height, width), vae_token_capacity((height, width))


@pytest.fixture(scope="module")
def source():
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        return AutoencoderKLQwenImage21(
            base_dim=8, decoder_base_dim=8, z_dim=4, num_res_blocks=2, latents_mean=[0.0] * 4, latents_std=[1.0] * 4
        ).eval()


@pytest.mark.parametrize("actual_hw", [(32, 32), (32, 64), (64, 32), (64, 96)])
@pytest.mark.parametrize("native_attention", [False, True])
def test_canvas_encode_decode_match_native_and_exclude_dirty_padding(source, canvas, actual_hw, native_attention):
    torch.manual_seed(11)
    height, width = actual_hw
    canvas_hw, capacity = canvas
    image = torch.randn(1, 4, height, width)
    latent = torch.randn(1, 4, height // 16, width // 16)
    masks = spatial_inputs(actual_hw, canvas_hw, dtype=torch.float32)
    encode = CanvasVaeEncode(source, capacity, use_noncausal_flash_attention=native_attention).eval()
    decode = CanvasVaeDecode(source, capacity, use_noncausal_flash_attention=native_attention).eval()
    with torch.inference_mode():
        native_latent = source.encode(image.unsqueeze(2), return_dict=False)[0].mode().squeeze(2)
        native_image = source.decode(latent.unsqueeze(2), return_dict=False)[0].clamp(-1, 1).squeeze(2)
        padded_image = F.pad(image, (0, canvas_hw[1] - width, 0, canvas_hw[0] - height))
        padded_latent = F.pad(latent, (0, (canvas_hw[1] - width) // 16, 0, (canvas_hw[0] - height) // 16))
        encoded, decoded = encode(padded_image, *masks), decode(padded_latent, *masks)
        torch.testing.assert_close(encoded[..., : height // 16, : width // 16], native_latent, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(decoded[..., :height, :width], native_image, atol=1e-5, rtol=3e-5)
        assert torch.count_nonzero(encoded * (1 - masks[4])) == 0
        assert torch.count_nonzero(decoded * (1 - masks[0])) == 0
        torch.testing.assert_close(encode(padded_image + 1000 * (1 - masks[0]), *masks), encoded, atol=0, rtol=0)
        torch.testing.assert_close(decode(padded_latent + 1000 * (1 - masks[4]), *masks), decoded, atol=0, rtol=0)


def test_canvas_session_reuses_graph_across_sizes_without_request_residue(source, canvas):
    canvas_hw, capacity = canvas
    module = CanvasVaeDecode(source, capacity).eval()
    session = CanvasVaeSession(module, canvas_hw, encode=False, token_capacity=capacity)
    small = torch.randn(1, 4, 2, 4)
    large = torch.randn(1, 4, 4, 6)
    with torch.inference_mode():
        first = session(small)
        assert session(large).shape == (1, 4, 64, 96)
        torch.testing.assert_close(session(small), first, atol=0, rtol=0)
    for actual in ((48, 64), (128, 32), (0, 32)):
        with pytest.raises(ValueError):
            spatial_inputs(actual, canvas_hw)


def test_canvas_session_allows_effective_limit_below_physical_capacity(source, canvas):
    canvas_hw, capacity = canvas
    module = CanvasVaeDecode(source, capacity).eval()
    latent = torch.randn(1, 4, 2, 4)
    limit = latent.shape[-2] * latent.shape[-1]
    session = CanvasVaeSession(module, canvas_hw, encode=False, token_capacity=limit)
    with torch.inference_mode():
        first = session(latent)
        assert first.shape == (1, 4, 32, 64)
        assert torch.isfinite(first).all()
        # This input fits the canvas but exceeds the effective request budget.
        with pytest.raises(ValueError, match="published token capacity"):
            session(torch.randn(1, 4, 4, 4))
        torch.testing.assert_close(session(latent), first, atol=0, rtol=0)


def test_canvas_attention_permutation_has_row_holes_and_restores_exactly(canvas):
    canvas_hw, capacity = canvas
    masks, order, restore, length = spatial_inputs((32, 64), canvas_hw)[-4:]
    assert length.item() == 8
    assert order[:8].tolist() == [0, 1, 2, 3, 6, 7, 8, 9]
    values = torch.arange(capacity)
    torch.testing.assert_close(values.index_select(0, order).index_select(0, restore), values)


@pytest.mark.parametrize("capacity", [0, -1, 36.5, True])
def test_canvas_attention_rejects_invalid_physical_capacity(source, capacity):
    with pytest.raises(ValueError, match="positive integer derived from the canvas"):
        CanvasAttention(source.encoder.mid_block.attentions[0], capacity)


@pytest.mark.parametrize("module_type", [CanvasVaeEncode, CanvasVaeDecode])
@pytest.mark.parametrize("capacity", [1, 35, 37])
def test_canvas_vae_rejects_mismatched_physical_capacity(source, canvas, module_type, capacity):
    canvas_hw, _ = canvas
    factor = 1 if module_type is CanvasVaeEncode else 16
    value = torch.zeros(1, 4, *(side // factor for side in canvas_hw))
    inputs = spatial_inputs((32, 64), canvas_hw, dtype=torch.float32)
    module = module_type(source, capacity).eval()
    # Capacity 1 used to broadcast silently and include padded KV positions.
    with torch.inference_mode(), pytest.raises(
        ValueError, match=f"VAE canvas 96x96 requires physical capacity 36, got {capacity}"
    ):
        module(value, *inputs)


@pytest.mark.parametrize("canvas_hw", [(32, 33), (33, 32), (32,), (0, 32)])
def test_canvas_geometry_is_checked_before_constructing_masks_or_session(canvas_hw):
    with pytest.raises(ValueError, match="positive multiples of 32"):
        spatial_inputs((32, 32), canvas_hw)
    with pytest.raises(ValueError, match="positive multiples of 32"):
        CanvasVaeSession(None, canvas_hw, encode=False, token_capacity=8)


@pytest.fixture(scope="module", params=["encoder", "decoder"])
def exported_canvas_attention(source, canvas, tmp_path_factory, request):
    from xhquant.api import (
        ConfigDict,
        DeviceType,
        QuantScheme,
        create_quant_config,
        convert_dynamo_model_to_quanted_model,
        convert_quanted_model_to_hmonnx,
    )
    torch.manual_seed(23)
    canvas_hw, capacity = canvas
    attention = getattr(source, request.param).mid_block.attentions[0]
    module = CanvasAttention(attention, capacity).eval()
    mask, order, restore, length = spatial_inputs((32, 64), canvas_hw, dtype=torch.float32)[-4:]
    inputs = (torch.randn(1, attention.dim, *(side // 16 for side in canvas_hw)), mask, order, restore, length)
    config = ConfigDict(create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type="w8a8_sefp")))
    quanted = convert_dynamo_model_to_quanted_model(module, inputs, DeviceType.XH2a, config)
    path = tmp_path_factory.mktemp(request.param) / "attention.onnx"
    convert_quanted_model_to_hmonnx(
        quanted, inputs, str(path), ["image", "mask", "order", "restore", "length"], ["output"]
    )
    return path, quanted, inputs


def test_canvas_qkv_split_exports_explicit_sections(exported_canvas_attention, canvas, tmp_path):
    import onnx
    from _qwen_image21_split_probe import assert_split_probe

    path, _, inputs = exported_canvas_attention
    _, capacity = canvas
    channels = inputs[0].shape[1]
    assert_split_probe(
        onnx.load(path), [channels] * 3, [1, 1, capacity, channels * 3], tmp_path / "split.onnx",
        split_output="split",
    )


def test_split_probe_selects_each_target_in_a_graph_with_two_valid_splits(exported_canvas_attention, canvas, tmp_path):
    import onnx
    from _qwen_image21_split_probe import assert_split_probe

    path, _, inputs = exported_canvas_attention
    model = onnx.load(path)
    original = next(node for node in model.graph.node if list(node.output) == ["split"])
    channels = inputs[0].shape[1]
    _, capacity = canvas
    shape = [1, 1, capacity, channels * 3]
    # A second legitimate partition of the same QKV tensor must not prevent
    # checking the actual QKV Split, or hide a failure in either target.
    sections = (channels, channels * 2)
    infos = {value.name: value for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    element_type = infos[original.input[0]].type.tensor_type.elem_type
    model.graph.node.append(
        onnx.helper.make_node(
            "Split", [original.input[0]], ["aux_parts"], name="aux_split", domain=original.domain,
            axis=3, indices_or_sections=sections,
        )
    )
    for index, width in enumerate(sections):
        output = f"aux_part_{index}"
        model.graph.node.append(
            onnx.helper.make_node(
                "TupleGetItem", ["aux_parts"], [output], name=output, domain=original.domain, index=index
            )
        )
        model.graph.output.append(onnx.helper.make_tensor_value_info(output, element_type, [*shape[:-1], width]))
    assert_split_probe(model, (channels,) * 3, shape, tmp_path / "qkv.onnx", split_output="split")
    assert_split_probe(model, sections, shape, tmp_path / "aux.onnx", split_output="aux_parts")
    with pytest.raises(AssertionError, match="expected one Split producing"):
        assert_split_probe(model, sections, shape, tmp_path / "missing.onnx", split_output="missing")


@pytest.mark.parametrize("attribute,value", [
    ("num_outputs", 3), ("indices_or_sections", [1, 1, 1]), ("indices_or_sections", [1.0, 1.0, 1.0]),
    ("axis", 2), ("axis", -1.5),
])
def test_split_probe_rejects_invalid_target_attributes(exported_canvas_attention, canvas, tmp_path, attribute, value):
    import onnx
    from _qwen_image21_split_probe import assert_split_probe

    path, _, inputs = exported_canvas_attention
    model = onnx.load(path)
    split = next(node for node in model.graph.node if list(node.output) == ["split"])
    kept = [entry for entry in split.attribute if entry.name != attribute]
    del split.attribute[:]
    split.attribute.extend([*kept, onnx.helper.make_attribute(attribute, value)])
    channels = inputs[0].shape[1]
    _, capacity = canvas
    with pytest.raises(AssertionError):
        assert_split_probe(
            model, [channels] * 3, [1, 1, capacity, channels * 3], tmp_path / "invalid.onnx", split_output="split"
        )


@pytest.mark.parametrize("fault", ["no_input", "empty_input", "missing_metadata", "extra_input"])
def test_split_probe_rejects_missing_or_additional_inputs(exported_canvas_attention, canvas, tmp_path, fault):
    import onnx
    from _qwen_image21_split_probe import assert_split_probe

    path, _, inputs = exported_canvas_attention
    model = onnx.load(path)
    split = next(node for node in model.graph.node if list(node.output) == ["split"])
    if fault == "no_input":
        del split.input[:]
    elif fault == "empty_input":
        split.input[0] = ""
    elif fault == "missing_metadata":
        split.input[0] = "missing_tensor"
    else:
        split.input.append("sections_tensor")
        model.graph.initializer.append(
            onnx.helper.make_tensor("sections_tensor", onnx.TensorProto.INT64, [3], [inputs[0].shape[1]] * 3)
        )
    channels = inputs[0].shape[1]
    _, capacity = canvas
    message = "tensor metadata is missing" if fault == "missing_metadata" else "exactly one non-empty tensor input"
    with pytest.raises(AssertionError, match=message):
        assert_split_probe(
            model, [channels] * 3, [1, 1, capacity, channels * 3], tmp_path / "invalid.onnx", split_output="split"
        )


def test_split_probe_rejects_duplicate_tuple_indices(exported_canvas_attention, canvas, tmp_path):
    import onnx
    from _qwen_image21_split_probe import assert_split_probe

    path, _, inputs = exported_canvas_attention
    model = onnx.load(path)
    parts = [node for node in model.graph.node if node.op_type == "TupleGetItem" and node.input[0] == "split"]
    next(entry for entry in parts[1].attribute if entry.name == "index").i = 0
    channels = inputs[0].shape[1]
    _, capacity = canvas
    with pytest.raises(AssertionError, match="TupleGetItem indices must be unique"):
        assert_split_probe(
            model, [channels] * 3, [1, 1, capacity, channels * 3], tmp_path / "invalid.onnx", split_output="split"
        )


@pytest.mark.parametrize("dtype_name", ["INT32", "DOUBLE"])
def test_split_probe_rejects_unsupported_input_dtype(exported_canvas_attention, canvas, tmp_path, dtype_name):
    import onnx
    from _qwen_image21_split_probe import assert_split_probe

    path, _, inputs = exported_canvas_attention
    model = onnx.load(path)
    split = next(node for node in model.graph.node if list(node.output) == ["split"])
    info = next(value for value in (*model.graph.input, *model.graph.value_info) if value.name == split.input[0])
    info.type.tensor_type.elem_type = getattr(onnx.TensorProto, dtype_name)
    channels = inputs[0].shape[1]
    _, capacity = canvas
    with pytest.raises(AssertionError, match="only supports FLOAT or FLOAT16"):
        assert_split_probe(
            model, [channels] * 3, [1, 1, capacity, channels * 3], tmp_path / "invalid.onnx", split_output="split"
        )


def test_split_probe_preserves_supported_float_dtypes(exported_canvas_attention, canvas, tmp_path):
    import onnx
    from _qwen_image21_split_probe import assert_split_probe

    path, _, inputs = exported_canvas_attention
    channels = inputs[0].shape[1]
    _, capacity = canvas
    sections, shape = [channels] * 3, [1, 1, capacity, channels * 3]
    baseline = tmp_path / "baseline.onnx"
    assert_split_probe(onnx.load(path), sections, shape, baseline, split_output="split")
    for elem_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
        model = onnx.load(baseline)
        for info in (*model.graph.input, *model.graph.output):
            info.type.tensor_type.elem_type = elem_type
        assert_split_probe(model, sections, shape, tmp_path / f"dtype_{elem_type}.onnx", split_output="split")


def test_canvas_attention_w8a8_export_matches_quantized_golden(exported_canvas_attention):
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    assert torch.cuda.is_available(), "W8A8 golden comparison requires a visible CUDA GPU"
    path, quanted, inputs = exported_canvas_attention
    runtime = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    quanted.cuda().half()
    gpu_inputs = tuple(value.cuda().half() if value.is_floating_point() else value.cuda() for value in inputs)
    with torch.inference_mode():
        expected, = quanted(*gpu_inputs)
        expected = expected.clone()
        actual = runtime(*gpu_inputs).clone()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
