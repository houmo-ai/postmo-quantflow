import hashlib
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import pytest

from xhmodel_merak.model_integrations.glm53.shared_weights import (
    compact_glm53_language_weights,
    share_language_weights,
)


def write_graph(root, stage, bias):
    directory = root / stage
    directory.mkdir(parents=True)
    weight = np.arange(4, dtype=np.float32).reshape(2, 2)
    name = stage + "_weight"
    tensors = [numpy_helper.from_array(weight, name), numpy_helper.from_array(np.array(bias, np.float32), "bias")]
    if stage == "decode":
        tensors += [numpy_helper.from_array(weight.flatten(), "other_shape"),
                    numpy_helper.from_array(weight.view(np.int32), "other_dtype")]
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", name], ["product"]),
         helper.make_node("Add", ["product", "bias"], ["sum"]),
         helper.make_node("Constant", [], ["constant"], value=numpy_helper.from_array(np.ones(2, np.float32))),
         helper.make_node("Add", ["sum", "constant"], ["y"])],
        stage, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])], tensors,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    path = directory / "model.onnx"
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="original.bin", size_threshold=0, convert_attribute=True)
    return path.relative_to(root).as_posix()


def evaluate(path):
    return ReferenceEvaluator(onnx.load(path)).run(None, {"x": np.array([[2, 3]], np.float32)})[0]


def test_shared_external_data_preserves_outputs_after_move(tmp_path):
    root = tmp_path / "artifact"
    graphs = {stage: write_graph(root, stage, bias) for stage, bias in [("prefill", [1, 2]), ("decode", [3, 4])]}
    expected = {stage: evaluate(root / path) for stage, path in graphs.items()}
    (root / "golden_meta_info.json").write_text(json.dumps({stage + "_hmonnx": path for stage, path in graphs.items()}))
    report = compact_glm53_language_weights(root)
    assert report["reused_tensors"] >= 2  # Renamed weights and external Constant attributes.
    metadata = json.loads((root / "golden_meta_info.json").read_text())
    assert all(report["graphs"][stage].startswith(stage + "/") for stage in graphs)
    assert metadata["glm53_shared_external_data"]["files"] == {
        stage: f"{stage}/weights.bin" for stage in graphs
    }
    assert all(not (root / path).exists() for path in report["source_external_files"])
    decode = onnx.load(root / report["graphs"]["decode"], load_external_data=False)
    spans = {t.name: {e.key: e.value for e in t.external_data} for t in decode.graph.initializer}
    assert spans["decode_weight"]["offset"] != spans["other_shape"]["offset"]
    assert spans["decode_weight"]["offset"] != spans["other_dtype"]["offset"]
    relocated = tmp_path / "relocated"
    root.rename(relocated)
    for stage, relative in report["graphs"].items():
        path = relocated / relative
        assert metadata[stage + "_hmonnx_md5"] == hashlib.md5(path.read_bytes()).hexdigest()
        onnx.checker.check_model(path)
        np.testing.assert_array_equal(evaluate(path), expected[stage])
    weight_paths = [relocated / stage / "weights.bin" for stage in graphs]
    assert len(list(relocated.rglob("*.bin"))) == 2
    assert len({path.stat().st_ino for path in weight_paths}) == 1


@pytest.mark.parametrize("field,value", [("location", "../escape.bin"), ("length", "999999"), ("offset", "-1")])
def test_invalid_external_data_preserves_originals(tmp_path, field, value):
    graphs = {stage: write_graph(tmp_path, stage, [1, 2]) for stage in ("prefill", "decode")}
    path = tmp_path / graphs["decode"]
    model = onnx.load(path, load_external_data=False)
    tensor = model.graph.initializer[0]
    next(e for e in tensor.external_data if e.key == field).value = value
    path.write_bytes(model.SerializeToString())
    originals = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(Exception, match="Artifact path|range"):
        share_language_weights(tmp_path, graphs, chunk_size=3)
    for path, content in originals.items():
        assert path.read_bytes() == content


def test_shared_conversion_rejects_overwrite(tmp_path):
    graphs = {stage: write_graph(tmp_path, stage, [1, 2]) for stage in ("prefill", "decode")}
    share_language_weights(tmp_path, graphs, chunk_size=3)
    before = (tmp_path / "prefill/weights.bin").read_bytes()
    with pytest.raises(FileExistsError):
        share_language_weights(tmp_path, graphs)
    assert (tmp_path / "prefill/weights.bin").read_bytes() == before
