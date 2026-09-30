import json
import runpy
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

_TOOL = runpy.run_path(
    Path(__file__).resolve().parents[1] / "tools/benchmark_w4_merak_abba.py"
)
_abba_pair_speedups = _TOOL["_abba_pair_speedups"]
_artifact_manifest = _TOOL["_artifact_manifest"]
_assert_logprobs_equal = _TOOL["_assert_logprobs_equal"]
_validate_runtime_w4_summary = _TOOL["_validate_runtime_w4_summary"]
_verify_artifact_manifest = _TOOL["_verify_artifact_manifest"]
_w4_artifact_facts = _TOOL["_w4_artifact_facts"]


def _write_external_w4_model(model_dir: Path) -> Path:
    (model_dir / "hf_config").mkdir(parents=True)
    (model_dir / "golden_meta_info.json").write_text("{}", encoding="utf-8")
    (model_dir / "merak_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "hf_config" / "config.json").write_text(
        "{}",
        encoding="utf-8",
    )
    graph_dir = model_dir / "prefill"
    graph_dir.mkdir()
    weight = np.zeros((1, 64, 8), dtype=np.int8)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Linear",
                ["input", "weight"],
                ["output"],
                domain="ai.houmo.xh2a",
                hmfp_weight_man_bit=4,
                hmfp_act_man_bit=8,
            )
        ],
        "w4",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [1, 64])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 8])],
        [numpy_helper.from_array(weight, name="weight")],
    )
    graph_path = graph_dir / "model.onnx"
    onnx.save_model(
        helper.make_model(graph),
        graph_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    return graph_path


def test_artifact_manifest_hashes_referenced_external_data(tmp_path) -> None:
    model_dir = tmp_path / "model"
    graph_path = _write_external_w4_model(model_dir)
    derived = graph_path.with_name(
        f"{graph_path.stem}_page_attention_0123456789abcdef.onnx"
    )
    derived.write_bytes(graph_path.read_bytes())

    manifest = _artifact_manifest(model_dir)

    assert manifest["external_data_files"] == 1
    external = [
        row for row in manifest["files"] if row["kind"] == "onnx_external_data"
    ]
    assert [Path(row["absolute_path"]).name for row in external] == [
        "weights.bin"
    ]
    assert str(derived) not in {
        row["absolute_path"] for row in manifest["files"]
    }
    derived.write_bytes(derived.read_bytes() + b"runtime-cache-update")
    _verify_artifact_manifest(manifest)
    assert _w4_artifact_facts(model_dir)["w4_nodes"] == 1


def test_artifact_manifest_detects_external_data_change(tmp_path) -> None:
    model_dir = tmp_path / "model"
    graph_path = _write_external_w4_model(model_dir)
    manifest = _artifact_manifest(model_dir)
    external_path = graph_path.parent / "weights.bin"
    external_path.write_bytes(external_path.read_bytes() + b"x")

    with pytest.raises(RuntimeError, match="artifact changed"):
        _verify_artifact_manifest(manifest)

    changed = _artifact_manifest(model_dir)
    assert changed["sha256"] != manifest["sha256"]


def _run(mode: str, values: list[float]) -> dict:
    return {
        "mode": mode,
        "rows": [
            {
                "max_tokens": 1,
                "latency": value,
            }
            for value in values
        ],
    }


def test_abba_pair_speedups_keep_process_order() -> None:
    runs = [
        _run("packed", [1.0, 1.1]),
        _run("expanded", [1.3, 1.4]),
        _run("expanded", [1.5, 1.6]),
        _run("packed", [1.2, 1.3]),
    ]

    speedups = _abba_pair_speedups(runs, "latency", 1)

    assert speedups == pytest.approx([1.35 / 1.05, 1.55 / 1.25])


def test_logprob_contract_detects_hidden_numeric_mismatch() -> None:
    expected = [
        [
            {
                "token_id": 7,
                "rank": 1,
                "decoded_token": "x",
                "logprob": -0.1,
            }
        ]
    ]
    actual = json.loads(json.dumps(expected))
    actual[0][0]["logprob"] = -0.2

    with pytest.raises(RuntimeError, match="top-logprob value mismatch"):
        _assert_logprobs_equal(actual, expected)


def test_runtime_w4_summary_requires_all_references_to_be_shared_and_packed() -> None:
    summary = {
        "runtime_w4_planned_references": 4,
        "runtime_w4_packed_references": 3,
        "runtime_w4_packed_initializers": 2,
        "runtime_w4_shared_references": 2,
        "runtime_w4_packed_saved_nbytes": 1024,
    }

    with pytest.raises(RuntimeError, match="not every planned"):
        _validate_runtime_w4_summary(summary, expected_packed=True)

    summary["runtime_w4_packed_references"] = 4
    _validate_runtime_w4_summary(summary, expected_packed=True)
