import runpy
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper


_TOOL = runpy.run_path(
    Path(__file__).resolve().parents[1] / "tools/check_hmonnx_attention_abi.py"
)
PreflightError = _TOOL["PreflightError"]
_check_attention_graph = _TOOL["_check_attention_graph"]


def _write_attention_graph(path: Path, *, extra_consumer: bool) -> None:
    attention = helper.make_node(
        "FlashAttention",
        ["input"],
        ["attention_output"],
        domain="ai.houmo.xh2a",
        num_heads=4,
    )
    transpose = helper.make_node(
        "Transpose",
        ["attention_output"],
        ["output"],
        perm=[0, 2, 1, 3],
    )
    nodes = [attention, transpose]
    outputs = [
        helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT16,
            [1, 8, 4, 16],
        )
    ]
    if extra_consumer:
        nodes.append(
            helper.make_node(
                "Identity",
                ["attention_output"],
                ["extra_output"],
            )
        )
        outputs.append(
            helper.make_tensor_value_info(
                "extra_output",
                TensorProto.FLOAT16,
                [1, 4, 8, 16],
            )
        )
    graph = helper.make_graph(
        nodes,
        "attention_abi",
        [
            helper.make_tensor_value_info(
                "input",
                TensorProto.FLOAT16,
                [1, 8, 64],
            )
        ],
        outputs,
        value_info=[
            helper.make_tensor_value_info(
                "attention_output",
                TensorProto.FLOAT16,
                [1, 4, 8, 16],
            )
        ],
    )
    onnx.save_model(helper.make_model(graph), path)


def test_attention_abi_accepts_one_immediate_layout_transpose(tmp_path) -> None:
    path = tmp_path / "model.onnx"
    _write_attention_graph(path, extra_consumer=False)

    facts = _check_attention_graph(path)

    assert facts["attention_nodes"] == 1
    assert facts["bhsd_to_bshd_transposes"] == 1


def test_attention_abi_rejects_extra_unsafe_consumer(tmp_path) -> None:
    path = tmp_path / "model.onnx"
    _write_attention_graph(path, extra_consumer=True)

    with pytest.raises(
        PreflightError,
        match="exactly one immediate Transpose",
    ):
        _check_attention_graph(path)
