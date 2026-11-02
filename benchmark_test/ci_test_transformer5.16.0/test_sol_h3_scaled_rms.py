# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Sol normalization policy and the actual text-refiner graph boundary."""

import pytest
import torch
from torch import nn

from xhmodel_merak.xh_other_model.models.sol_h3.export import export_graph


def test_sol_norm_and_reference_yaml_flags_default_on_and_validate_types():
    from copy import deepcopy
    from pathlib import Path

    import yaml

    from xhmodel_merak.xh_other_model.models.sol_h3.build import validate_build_config

    root = Path(__file__).resolve().parents[2]
    path = root / "configs_merak/workflows/xh2a/other_models/sol_h3/sol_h3_all_384p_768p_w4.yaml"
    raw = yaml.safe_load(path.read_text())["export"]
    assert raw["norm_fp16"] is True and raw["ltx_reference_conditioning"]["enabled"] is True
    assert raw.pop("upscaler_conv3d_mode") == "slice"
    assert raw.pop("upscaler_conv3d_fastpath") is False
    raw.pop("norm_fp16")
    raw.pop("ltx_reference_conditioning")
    default = validate_build_config(raw)
    assert default["upscaler_conv3d_mode"] == "slice"
    assert default["upscaler_conv3d_fastpath"] is False
    assert validate_build_config({**raw, "upscaler_conv3d_fastpath": True})["upscaler_conv3d_fastpath"] is True
    for fastpath in (0, "false", None):
        with pytest.raises(ValueError, match="upscaler_conv3d_fastpath"):
            validate_build_config({**raw, "upscaler_conv3d_fastpath": fastpath})
    assert validate_build_config({**raw, "upscaler_conv3d_mode": "gather"})["upscaler_conv3d_mode"] == "gather"
    for mode in (False, "auto"):
        with pytest.raises(ValueError, match="upscaler_conv3d_mode"):
            validate_build_config({**raw, "upscaler_conv3d_mode": mode})
    assert default["norm_fp16"] is True and default["ltx_reference_conditioning"] == {
        "enabled": True,
        "encoder_norm_eps": 2**-24,
    }
    explicit = validate_build_config({**raw, "norm_fp16": False, "ltx_reference_conditioning": {"enabled": False}})
    assert explicit["norm_fp16"] is False and explicit["ltx_reference_conditioning"] == {
        "enabled": False,
        "encoder_norm_eps": 2**-24,
    }
    for value in (0, "true"):
        with pytest.raises(ValueError, match="norm_fp16"):
            validate_build_config({**raw, "norm_fp16": value})
        with pytest.raises(ValueError, match="ltx_reference_conditioning"):
            validate_build_config({**raw, "ltx_reference_conditioning": {"enabled": value}})
    before = deepcopy(raw)
    validate_build_config(raw)
    assert raw == before


@torch.no_grad()
@pytest.mark.parametrize("capacity", [5, 1792])
def test_refiner_div4_exports_plain_rms_and_matches_native(tmp_path, capacity):
    import onnx
    from test_sol_h3_static_graph import text_key_mask, tiny_source

    from xhmodel_merak.xh_other_model.models.sol_h3.h3_graph import H3TextGraph
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    source = tiny_source()
    # Exercise the actual CUDA FlashAttention kernel, whose minimum head width is 64.
    source = (
        type(source)
        .from_config(source.config, attention_head_dim=64, hidden_size=128, ffn_dim=256)
        .eval()
        .requires_grad_(False)
    )
    for module in source.modules():
        if isinstance(module, nn.RMSNorm):
            module.eps = 1e-5
    values = torch.randn(1, capacity, 24) / 16
    values[:, 3:] = 0
    mask = text_key_mask(capacity, 3)
    expected = source.token_refiner(source.context_embedder(values[:, :3] * 16))
    path = tmp_path / "refiner.onnx"
    export_graph(
        H3TextGraph(source, input_divisor=16, sdk_ops=True),
        (values, mask),
        path,
        ["features", "key_mask"],
        ["refined"],
        quant_scheme={"target_device": "XH2a", "quant_type": "w16a16_sefp"},
    )
    nodes = onnx.load(path, load_external_data=False).graph.node
    graph = onnx.load(path, load_external_data=False).graph
    assert [value.name for value in graph.input] == ["features", "key_mask"]
    assert [dim.dim_value for dim in graph.input[1].type.tensor_type.shape.dim] == [1, 1, 1, capacity]
    flashes = [node for node in nodes if node.op_type == "FlashAttention"]
    assert len(flashes) == 2 * int(capacity >= 512)
    assert all(len(node.input) == 5 and node.input[4] == "key_mask" for node in flashes)
    assert not any(node.op_type in {"Less", "LessOrEqual"} for node in nodes)
    assert not any(node.op_type in {"Abs", "ReduceMax", "Max"} for node in nodes)
    norms = [node for node in nodes if node.op_type == "RMSNorm"]
    attrs = [{a.name: onnx.helper.get_attribute_value(a) for a in node.attribute} for node in norms]
    assert len(norms) == 9
    assert sum(abs(a["eps"] - 1e-5 / 16) < 1e-12 for a in attrs) == 5
    assert all(a["force_fp32"] == 1 for a in attrs)
    session = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    output = session.run({"features": values.cuda().half(), "key_mask": mask.cuda().half()})
    actual = output[0] if isinstance(output, (tuple, list)) else output
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[:, :3].cpu().float(), expected, rtol=0.03, atol=0.03)


@pytest.mark.parametrize(
    "options",
    [
        {"rms_norm_eps": 1e-10},
        {"rms_norm_mode": "stable", "rms_norm_eps": 2.0**-24},
    ],
)
def test_residual_precision_rejects_unsafe_or_ignored_configuration(options):
    from xhmodel_merak.xh_other_model.models.sol_h3.precision import H3PrecisionPolicy

    with pytest.raises(ValueError):
        H3PrecisionPolicy(**options)
