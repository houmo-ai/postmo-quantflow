"""ONNX metadata removal, backup and recovery; no model weights required."""
import sys

import onnx
from onnx import helper
import pytest

from examples_merak.llm.qwen3.debug_scripts import remove_staure as script

pytestmark = pytest.mark.transformer516


def make_model():
    nested = helper.make_graph([helper.make_node("Cast", ["x"], ["y"], to=1, saturate=1)], "nested", [], [])
    parent = helper.make_node("Custom", ["x"], ["z"], saturate=0, keep=7)
    parent.attribute.extend([helper.make_attribute("body", nested), helper.make_attribute("branches", [nested])])
    return helper.make_model(helper.make_graph([parent], "main", [], []))


def test_remove_saturate_recurses_without_changing_other_attributes():
    model = make_model()
    assert script.remove_saturate(model.graph.node) == 3
    node = model.graph.node[0]
    assert {attr.name for attr in node.attribute} == {"keep", "body", "branches"}
    assert next(attr.i for attr in node.attribute if attr.name == "keep") == 7
    assert script.remove_saturate(model.graph.node) == 0


def test_cli_preserves_original_and_refuses_existing_backup(monkeypatch, tmp_path):
    source = tmp_path / "model.onnx"
    onnx.save(make_model(), source)
    original = source.read_bytes()
    monkeypatch.setattr(sys, "argv", ["remove_staure.py", str(source)])
    script.main()
    backup = source.with_name("model.onnx.bak")
    assert backup.read_bytes() == original
    assert script.remove_saturate(onnx.load(source).graph.node) == 0
    cleaned = source.read_bytes()
    with pytest.raises(SystemExit) as caught:
        script.main()
    assert caught.value.code == 2
    assert source.read_bytes() == cleaned
    assert backup.read_bytes() == original


def test_failed_save_restores_original(monkeypatch, tmp_path):
    source = tmp_path / "model.onnx"
    onnx.save(make_model(), source)
    original = source.read_bytes()
    monkeypatch.setattr(sys, "argv", ["remove_staure.py", str(source)])
    def fail_save(model, path):
        path.write_bytes(b"partial write")
        raise OSError("disk full")
    monkeypatch.setattr(script.onnx, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        script.main()
    assert source.read_bytes() == original
    assert not source.with_name("model.onnx.bak").exists()
