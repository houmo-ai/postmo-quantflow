from pathlib import Path

import pytest

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build.artifacts import flatten_single_export_root


def _release(tmp_path: Path) -> Path:
    release = tmp_path / "hmquant_model"
    (release / "audio").mkdir(parents=True)
    (release / "audio/model.onnx").write_bytes(b"published graph")
    return release


def test_empty_component_workspace_does_not_collide(tmp_path):
    release = _release(tmp_path)
    (tmp_path / "audio").mkdir()
    (tmp_path / "effective_workflow.yaml").write_text("export: {}")

    assert flatten_single_export_root(tmp_path) == release.name
    assert (tmp_path / "audio/model.onnx").read_bytes() == b"published graph"
    assert (tmp_path / "effective_workflow.yaml").read_text() == "export: {}"
    assert not release.exists()


@pytest.mark.parametrize("collision", ["file", "nonempty", "nested_empty", "symlink", "dangling_symlink"])
def test_nonempty_or_link_workspace_is_not_overwritten(tmp_path, collision):
    release = _release(tmp_path)
    destination = tmp_path / "audio"
    if collision == "file":
        destination.write_bytes(b"existing")
    elif collision in {"nonempty", "nested_empty"}:
        destination.mkdir()
        if collision == "nonempty":
            (destination / "existing").write_bytes(b"existing")
        else:
            (destination / "empty").mkdir()
    else:
        target = tmp_path / "other"
        if collision == "symlink":
            target.mkdir()
        destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(ModelError, match="names collide"):
        flatten_single_export_root(tmp_path)
    assert (release / "audio/model.onnx").read_bytes() == b"published graph"
    assert destination.exists() or destination.is_symlink()


def test_all_collisions_are_checked_before_removing_empty_directories(tmp_path):
    release = _release(tmp_path)
    (tmp_path / "audio").mkdir()
    (release / "metadata.json").write_text("new")
    (tmp_path / "metadata.json").write_text("existing")
    with pytest.raises(ModelError, match="metadata.json"):
        flatten_single_export_root(tmp_path)
    assert (tmp_path / "audio").is_dir()
    assert (tmp_path / "metadata.json").read_text() == "existing"
    assert (release / "metadata.json").read_text() == "new"


def test_exported_file_cannot_replace_empty_directory(tmp_path):
    release = _release(tmp_path)
    (release / "metadata.json").write_text("new")
    (tmp_path / "metadata.json").mkdir()
    with pytest.raises(ModelError, match="metadata.json"):
        flatten_single_export_root(tmp_path)
