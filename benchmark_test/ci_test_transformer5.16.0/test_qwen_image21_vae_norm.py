"""Existing RMSNorm fusion, zero padding and real FX/Dynamo HMONNX contracts."""

from types import SimpleNamespace

import onnx
import pytest
import torch

from xhmodel_merak.xh_other_model.models.qwen_image21.export import export_cpu_graph
from xhmodel_merak.xh_other_model.models.qwen_image21.vae_graph import SpatialChannelNormalize


def make_vae_norm(gamma):
    return SpatialChannelNormalize(
        SimpleNamespace(channel_first=True, bias=0, gamma=gamma, scale=gamma.numel() ** 0.5)
    )


@pytest.mark.parametrize("channels", [64, 768, 1152])
def test_vae_rmsnorm_has_small_epsilon_and_finite_zero_padding(channels):
    torch.manual_seed(5)
    values = torch.randn(1, channels, 2, 4)
    values[:, :, :, 1] = 0
    values[:, :, :, 2] *= 1e-14
    gamma = torch.randn(channels)
    norm = make_vae_norm(gamma)
    expected = values * torch.rsqrt(values.square().mean(1, keepdim=True) + 1e-8) * gamma[None, :, None, None]
    actual = norm(values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual[:, :, :, 1]) == 0
    # The accepted epsilon approximation remains close on ordinary activations.
    native = torch.nn.functional.normalize(values, dim=1) * channels**0.5 * gamma[None, :, None, None]
    torch.testing.assert_close(actual[:, :, :, [0, 3]], native[:, :, :, [0, 3]], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("dynamo", [False, True])
@pytest.mark.parametrize("quant_type", ["w16a16_sefp", "w8a16_sefp", "w8a8_sefp"])
def test_vae_rmsnorm_exports_as_one_existing_operator_and_loads(tmp_path, dynamo, quant_type):
    from xhquant.api import DeviceType, QuantScheme, create_quant_config
    from xhquant.backend.xh2a.ir.operators.rmsnorm import rmsnorm_xh2a_default
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(13)
    module = make_vae_norm(torch.empty(64).uniform_(0.5, 1.5)).eval()
    values = torch.randn(1, 64, 4, 4)
    values[:, :, 1] *= 1e-6
    values[:, :, 2:] = 0
    path = tmp_path / "norm.onnx"
    config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type))
    export_cpu_graph(
        module, (values,), path, ["x"], ["y"], target_device=DeviceType.XH2a, quant_config=config, dynamo=dynamo
    )
    graph = onnx.load(path).graph
    norms = [node for node in graph.node if node.op_type == "RMSNorm"]
    assert len(norms) == 1
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in norms[0].attribute}
    assert "epsilon_mode" not in attrs and attrs["eps"] == pytest.approx(1e-8)
    assert not any(node.op_type in {"Normalize", "Mul", "Add"} for node in graph.node)
    session = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cpu"]))
    with torch.inference_mode():
        actual = session(values.half())
        expected = rmsnorm_xh2a_default(
            values.half().permute(0, 2, 3, 1), module.norm.weight.half(), 1e-8, force_fp32=True
        )
        torch.testing.assert_close(actual, expected.permute(0, 3, 1, 2), atol=0, rtol=0)
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual[:, :, 2:]) == 0
