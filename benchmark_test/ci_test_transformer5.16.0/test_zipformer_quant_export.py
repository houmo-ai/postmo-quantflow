"""Small real Cast/Conv/Pow export reproducing HMSW-4851 and HMSW-4852.

Keep the integer cache cast and unfixed Pow: together they trigger PTQ's
metadata propagation, which must use the lowered floating execution dtype.
No full encoder checkpoint or compiler installation is needed.
"""

import importlib
from contextlib import nullcontext

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from xhmodel_merak.xh_other_model.models.zipformer.workflow import _convert_hmonnx
from xhquant.api import (
    DeviceType,
    HMONNXInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
)
from xhquant.core.graph import FrontendGraph
from xhquant.frontend.onnx.parsers.cast import Cast as ONNXCastParser
from xhquant.quantization.quantize_wrap import to_quant_graph


def _cast_graph(dtype, target="XH2a"):
    from types import SimpleNamespace

    root = torch.nn.Module()
    root.cast = ONNXCastParser().parse(None, SimpleNamespace(attrs={"to": dtype}))
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    graph.output(graph.call_module("cast", (value,)))
    frontend = FrontendGraph.convert(torch.fx.GraphModule(root, graph), {})
    return to_quant_graph(frontend, target, {}).disable_quant()


@pytest.mark.parametrize("dtype", [TensorProto.FLOAT, TensorProto.DOUBLE])
def test_xh2a_cast_reference_preserves_explicit_float_target(dtype):
    graph = _cast_graph(dtype)
    values = torch.tensor([1, 2049, 65519], dtype=torch.int32)
    original_dtype = torch.float32 if dtype == TensorProto.FLOAT else torch.float64
    assert graph(values).dtype == original_dtype
    graph.half()
    torch.testing.assert_close(graph(values), values.to(original_dtype))
    assert graph.cast._dtype == torch.float16
    graph.float()
    assert graph(values).dtype == original_dtype


@pytest.mark.parametrize("dtype", [TensorProto.INT32, TensorProto.INT64])
def test_xh2a_cast_reference_preserves_integer_width(dtype):
    graph = _cast_graph(dtype).half()
    values = torch.tensor([1.9, -2.9], dtype=torch.float16)
    expected_dtype = torch.int32 if dtype == TensorProto.INT32 else torch.int64
    torch.testing.assert_close(graph(values), torch.tensor([1, -2], dtype=expected_dtype))


def test_xh3_cast_keeps_explicit_fp32_boundary_after_half():
    graph = _cast_graph(TensorProto.FLOAT, target="XH3").half()
    values = torch.tensor([1, 17], dtype=torch.int32)
    torch.testing.assert_close(graph(values), values.float())


def _cache_model(dtype=TensorProto.FLOAT, mixed=False, with_pow=True):
    shape = [1, 2, 2, 2]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Cast", ["cached_len"], ["length_float"], to=TensorProto.FLOAT),
                helper.make_node("Add", ["x", "length_float"], ["offset"]),
                helper.make_node(
                    "Conv",
                    ["offset", "weight", "bias"],
                    ["conv"],
                    kernel_shape=[1, 1],
                    strides=[1, 1],
                    pads=[0, 0, 0, 0],
                    dilations=[1, 1],
                    group=1,
                ),
                helper.make_node("Pow", ["conv", "exponent"], ["output"]),
                helper.make_node("Add", ["cached_len", "step"], ["new_cached_len"]),
            ],
            "zipformer_cast_conv_pow",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
                helper.make_tensor_value_info("cached_len", TensorProto.INT32, [1, 1, 1, 1]),
            ],
            [
                helper.make_tensor_value_info("output", TensorProto.FLOAT, shape),
                helper.make_tensor_value_info("new_cached_len", TensorProto.INT32, [1, 1, 1, 1]),
            ],
            [
                numpy_helper.from_array(np.eye(2, dtype=np.float32).reshape(2, 2, 1, 1), "weight"),
                numpy_helper.from_array(np.array([0.25, -0.25], dtype=np.float32), "bias"),
                numpy_helper.from_array(np.array(2, dtype=np.float32), "exponent"),
                numpy_helper.from_array(np.ones([1, 1, 1, 1], dtype=np.int32), "step"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=9,
    )
    if dtype == TensorProto.FLOAT16:
        for value in list(model.graph.input) + list(model.graph.output):
            if value.type.tensor_type.elem_type == TensorProto.FLOAT:
                value.type.tensor_type.elem_type = dtype
        for index, tensor in enumerate(model.graph.initializer):
            if tensor.data_type == TensorProto.FLOAT:
                model.graph.initializer[index].CopyFrom(
                    numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float16), tensor.name)
                )
        model.graph.node[0].attribute[0].i = dtype
    if mixed:
        assert dtype == TensorProto.FLOAT16
        model.graph.node[1].input[0] = "x_half"
        nodes = [
            helper.make_node("Cast", ["x"], ["x_float"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["x_float"], ["x_half"], to=TensorProto.FLOAT16),
        ] + list(model.graph.node)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    if not with_pow:
        for index, node in enumerate(model.graph.node):
            if node.op_type == "Pow":
                del model.graph.node[index]
                break
        model.graph.output[0].name = "conv"
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize(
    ("dtype", "mixed"),
    [(TensorProto.FLOAT, False), (TensorProto.FLOAT16, False), (TensorProto.FLOAT16, True)],
    ids=["fp32", "fp16", "mixed-casts"],
)
def test_zipformer_cache_cast_conv_pow_exports_and_runs(tmp_path, dtype, mixed):
    model = _cache_model(dtype, mixed)
    source = tmp_path / "source.onnx"
    output = tmp_path / "export.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, source)
    if dtype == TensorProto.FLOAT:
        _convert_hmonnx(source, output, "XH2a", "w16a16_sefp", debug=False)
    else:
        config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type="w16a16_sefp"))
        convert_onnx_to_hmonnx(
            str(source),
            [torch.zeros(1, 2, 2, 2, dtype=torch.float16), torch.zeros(1, 1, 1, 1, dtype=torch.int32)],
            DeviceType.XH2a,
            str(output),
            quant_config=config,
            input_names=["x", "cached_len"],
            output_names=["output", "new_cached_len"],
            simplify=False,
        )
    exported = onnx.load(output)
    assert [v.name for v in exported.graph.input] == ["x", "cached_len"]
    assert [v.name for v in exported.graph.output] == ["output", "new_cached_len"]
    assert [v.type.tensor_type.elem_type for v in exported.graph.input] == [TensorProto.FLOAT16, TensorProto.INT32]
    session = HMONNXInference(str(output))
    session.to("cpu")
    values = torch.arange(8, dtype=torch.float16).reshape(1, 2, 2, 2) / 8
    length = torch.zeros([1, 1, 1, 1], dtype=torch.int32)
    bias = torch.tensor([0.25, -0.25], dtype=torch.float16).reshape(1, 2, 1, 1)
    with torch.no_grad():
        for step in range(3):
            actual, length = session(values, length)
            assert actual.dtype == torch.float16
            assert length.dtype == torch.int32
            torch.testing.assert_close(actual.float(), ((values + step + bias) ** 2).float(), rtol=0, atol=0.02)
            torch.testing.assert_close(length, torch.full_like(length, step + 1))


@pytest.mark.parametrize(
    ("with_pow", "infer_shape", "use_calibration", "preserve_dtype"),
    [(True, True, False, False), (True, False, True, False),
     (False, True, True, False), (True, True, True, True)],
    ids=["metadata-inputs", "inference-disabled", "weights-only", "preserve-dtype"],
)
@pytest.mark.parametrize("graph_v2", [False, True], ids=["graph-v1", "graph-v2"])
def test_ptq_metadata_inference_conditions(
    tmp_path, monkeypatch, with_pow, infer_shape, use_calibration, preserve_dtype, graph_v2
):
    from xhquant.api.ptq_export_hmonnx import _convert_model_to_quanted_model
    from xhquant.common.types import FrontendType, PrecisionMode
    from xhquant.core import QBaseModule
    from xhquant.core.graph import QuantGraphv2
    from xhquant.quantization.xh2a import ptq_quantizer_impl

    ptq = importlib.import_module("xhquant.quantization.xh2a.ptq_quantizer_impl")
    source = tmp_path / "source.onnx"
    onnx.save(_cache_model(with_pow=with_pow), source)
    inputs = [torch.zeros(1, 2, 2, 2), torch.zeros(1, 1, 1, 1, dtype=torch.int32)]
    config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type="w16a16_sefp"))
    graph = _convert_model_to_quanted_model(
        str(source), FrontendType.ONNX, inputs, DeviceType.XH2a, config, use_ptq=False, simplify=False
    )
    if graph_v2:
        graph = QuantGraphv2.convert(graph)
    calls = []
    original = ptq.MetaInfoPro.propagate

    expected_dtype = torch.float32 if preserve_dtype else torch.float16

    def record_metadata(self, *args):
        calls.append(tuple(arg.dtype for arg in args))
        assert all(param.dtype == expected_dtype for param in self.module.parameters())
        return original(self, *args)

    monkeypatch.setattr(ptq.MetaInfoPro, "propagate", record_metadata)
    ptq_quantizer_impl(
        graph, [inputs] if use_calibration else None, PrecisionMode.ALIGNED, torch.device("cpu"),
        infer_shape=infer_shape, preserve_model_dtype=preserve_dtype,
    )
    assert graph.is_fixed()
    assert all(module.precision_mode == PrecisionMode.ALIGNED
               for module in graph.modules() if isinstance(module, QBaseModule))
    assert all(param.dtype == expected_dtype for param in graph.parameters())
    assert calls == ([(expected_dtype, torch.int32)] if with_pow and infer_shape else [])
    # The temporary metadata scope must not round ordinary reference values.
    from xhquant.quantization.xh2a.qmodules.cast import QCast

    graph.disable_quant()
    casts = [module for module in graph.modules() if isinstance(module, QCast)]
    assert casts
    for module in casts:
        torch.testing.assert_close(module(torch.tensor([2049], dtype=torch.int32)), torch.tensor([2049.0]))


@pytest.mark.parametrize(
    ("target", "dtype"),
    [("XH2a", TensorProto.FLOAT), ("XH2a", TensorProto.INT64), ("XH3", TensorProto.FLOAT)],
)
@pytest.mark.parametrize("graph_v2", [False, True], ids=["graph-v1", "graph-v2"])
@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_ptq_metadata_restores_cast_and_quant_states(target, dtype, graph_v2, fail):
    from xhquant.common import PrecisionMode
    from xhquant.core import QBaseModule
    from xhquant.core.graph import QuantGraphv2
    from xhquant.quantization.xh2a.ptq_quantizer_impl import _ptq_metadata_context

    graph = _cast_graph(dtype, target).half()
    if graph_v2:
        graph = QuantGraphv2.convert(graph)
    graph.enable_quant()
    graph.cast.precision_mode = PrecisionMode.FAST
    states = [(module, module.quant_state, module.precision_mode)
              for module in graph.modules() if isinstance(module, QBaseModule)]
    dtypes = graph.cast._old_dtype, graph.cast._dtype
    with pytest.raises(AttributeError) if fail else nullcontext():
        with _ptq_metadata_context(graph, lower_float_casts=True):
            assert not graph.cast.is_quanting
            expected_dtype = torch.int64 if dtype == TensorProto.INT64 else (
                torch.float32 if target == "XH3" else torch.float16
            )
            assert graph(torch.tensor([2049], dtype=torch.int32)).dtype == expected_dtype
            # Fail inside the real Cast forward after it selects its reference
            # dtype, before QCast can restore its lowered _dtype itself.
            if fail:
                graph(None)
    assert (graph.cast._old_dtype, graph.cast._dtype) == dtypes
    assert all((module.quant_state, module.precision_mode) == (state, mode) for module, state, mode in states)
    graph.disable_quant()
    expected = torch.tensor([2049], dtype=torch.int64 if dtype == TensorProto.INT64 else torch.float32)
    torch.testing.assert_close(graph(torch.tensor([2049], dtype=torch.int32)), expected)


def test_ptq_half_layernorm_with_float_bias_exports_and_runs(tmp_path, monkeypatch):
    """Keep CI's mixed bias and LayerNorm2RMSNorm -> Add -> Conv failure path."""
    from xhquant.api import convert_quanted_model_to_hmonnx
    from xhquant.api.ptq_export_hmonnx import _convert_model_to_quanted_model
    from xhquant.common.types import FrontendType, PrecisionMode
    from xhquant.quantization.xh2a import ptq_quantizer_impl
    from xhquant.quantization.xh2a.qmodules.rmsnorm import QRMSNorm

    monkeypatch.setenv("ENABLE_LAYERNORM2RMSNORM", "1")
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["norm"], axis=-1),
                helper.make_node("Transpose", ["norm"], ["nchw"], perm=[0, 3, 1, 2]),
                helper.make_node(
                    "Conv", ["nchw", "weight", "conv_bias"], ["output"], kernel_shape=[1, 1],
                    strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
                ),
            ],
            "half_layernorm_float_bias",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 2, 2, 8])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 8, 2, 2])],
            [
                numpy_helper.from_array(np.ones(8, dtype=np.float16), "scale"),
                numpy_helper.from_array(np.linspace(-0.25, 0.25, 8, dtype=np.float32), "bias"),
                numpy_helper.from_array(np.eye(8, dtype=np.float16).reshape(8, 8, 1, 1), "weight"),
                numpy_helper.from_array(np.zeros(8, dtype=np.float16), "conv_bias"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=9,
    )
    source, output = tmp_path / "mixed_bias.onnx", tmp_path / "export.onnx"
    onnx.save(model, source)
    values = torch.arange(32, dtype=torch.float16).reshape(1, 2, 2, 8) / 8
    config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type="w16a16_sefp"))
    graph = _convert_model_to_quanted_model(
        str(source), FrontendType.ONNX, [values], DeviceType.XH2a, config, use_ptq=False, simplify=False
    )
    # Prove that the mixed dtype survived the actual ONNX frontend and that
    # an unfixed RMSNorm forces PTQ metadata propagation.
    assert {param.dtype for param in graph.parameters()} == {torch.float16, torch.float32}
    assert any(isinstance(module, QRMSNorm) and not module.is_quant_fixed for module in graph.modules())
    ptq_quantizer_impl(graph, [[values]], PrecisionMode.ALIGNED, torch.device("cpu"))
    assert graph.is_fixed()
    assert all(param.dtype == torch.float16 for param in graph.parameters())
    convert_quanted_model_to_hmonnx(graph, [values], str(output))
    session = HMONNXInference(str(output))
    session.to("cpu")
    with torch.no_grad():
        actual = session(values)
    if isinstance(actual, (tuple, list)):
        (actual,) = actual
    assert actual.dtype == torch.float16
    # Compare the deployed half-parameter model, whose weight normalization is
    # also what allowed the existing mixed-initializer CI model to execute.
    bias = torch.linspace(-0.25, 0.25, 8).half()
    expected = torch.nn.functional.layer_norm(values, [8], torch.ones(8).half(), bias, eps=1e-5)
    torch.testing.assert_close(actual, expected.permute(0, 3, 1, 2), rtol=0.01, atol=0.01)
