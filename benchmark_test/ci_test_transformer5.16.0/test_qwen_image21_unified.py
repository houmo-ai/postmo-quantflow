"""Image CLI forwarding and dependency publication use the real public contracts.

Only expensive build/inference execution is replaced in forwarding checks.
Real W8A8/native-attention six-graph export and CLI inference run in test_qwen_image21_cpu_export.
"""

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from examples_merak.aigc.qwen_image21 import unified
from examples_merak.unified_api import dependencies, generate_image
from xhmodel_merak import model_build, model_inference
from xhmodel_merak.model_build import describe_model
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.model_inference.inputs import validate_inputs


MODEL_IDS = ("qwen-image-2.1", "qwen-image-2.1-turbo")
DIFFUSERS = "diffusers==0.41.0"


@pytest.mark.parametrize("module", [unified, generate_image], ids=["model", "public"])
@pytest.mark.parametrize("editing", [False, True], ids=["generate", "edit_cfg"])
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_image_cli_forwards_inputs_and_preserves_rgba(model_id, module, editing, monkeypatch, tmp_path, capsys):
    pixels = np.arange(64 * 32 * 4, dtype=np.uint8).reshape(64, 32, 4)
    signature = inspect.signature(model_inference.inference_model)
    calls = []

    def run(**kwargs):
        signature.bind(**kwargs)
        validate_inputs(get_recipe(model_id), kwargs["inputs"])
        calls.append(kwargs)
        return SimpleNamespace(image=pixels, width=32, height=64, seed=9, latency_s=0.125)

    monkeypatch.setattr(model_inference, "inference_model", run)
    monkeypatch.setattr(generate_image, "inference_model", run)
    destination = tmp_path / "images" / "output.png"
    argv = ["--artifact-dir", "package", "--prompt", "a teapot", "--steps", "3", "--seed", "9",
            "--device", "cpu", "--output", str(destination)]
    expected = dict(prompt="a teapot", images=[], steps=3, seed=9, guidance_scale=1.0)
    if editing:
        argv += ["--image", "ref.png", "--height", "64", "--width", "32",
                 "--negative-prompt", "", "--guidance-scale", "2"]
        expected.update(images=["ref.png"], height=64, width=32, negative_prompt="", guidance_scale=2.0)
    if module is unified:
        argv += ["--model-id", model_id]
    module.main(argv)
    assert len(calls) == 1 and calls[0]["inputs"] == expected
    assert calls[0]["device"] == "cpu"
    assert calls[0]["artifact_dir"] == (str(Path("package").resolve()) if module is unified else "package")
    if module is unified:
        assert calls[0]["expected_model_id"] == model_id
    with Image.open(destination) as saved:
        assert saved.mode == "RGBA" and saved.size == (32, 64)
        np.testing.assert_array_equal(saved, pixels)
    assert str(destination) in capsys.readouterr().out


@pytest.mark.parametrize("precision", [None, "w16a16_sefp", "w8a16_sefp", "w8a8_sefp"])
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_model_cli_builds_requested_precision_then_uses_returned_path(model_id, precision, monkeypatch, tmp_path):
    build_signature = inspect.signature(model_build.build_model)
    inference_signature = inspect.signature(model_inference.inference_model)
    calls = []
    published = tmp_path / "published"

    def build(**kwargs):
        build_signature.bind(**kwargs)
        calls.append(("build", kwargs))
        return SimpleNamespace(output_dir=str(published))

    def infer(**kwargs):
        inference_signature.bind(**kwargs)
        validate_inputs(get_recipe(model_id), kwargs["inputs"])
        calls.append(("inference", kwargs))
        return SimpleNamespace(image=np.zeros((32, 32, 4), dtype=np.uint8), width=32, height=32,
                               seed=42, latency_s=0.1)

    monkeypatch.setattr(model_build, "build_model", build)
    monkeypatch.setattr(model_inference, "inference_model", infer)
    argv = ["--model-dir", "source", "--artifact-dir", str(tmp_path / "requested"), "--prompt", "a teapot",
            "--output", str(tmp_path / "out.png"), "--device", "cuda:0"]
    if precision is not None:
        argv += ["--quant-type", precision]
    argv += ["--model-id", model_id]
    unified.main(argv)
    assert [phase for phase, _ in calls] == ["build", "inference"]
    assert calls[0][1]["model_id"] == model_id
    assert calls[0][1]["deployment_precision"] == {"image": precision or "w16a16_sefp"}
    assert "attention" not in calls[0][1]
    assert calls[1][1]["artifact_dir"] == str(published)


@pytest.mark.parametrize("module,extra", [
    (unified, ["--quant-type", "w8a8_sefp"]),
    (unified, ["--model-dir", "source", "--attention", "masked"]),
    (unified, ["--guidance-scale", "2"]),
    (generate_image, ["--guidance-scale", "2"]),
    (generate_image, ["--golden"]),
])
def test_cli_rejects_unusable_options_before_execution(module, extra, tmp_path):
    with pytest.raises(SystemExit) as error:
        module.main(["--artifact-dir", "package", "--prompt", "a teapot",
                     "--output", str(tmp_path / "out.png"), *extra])
    assert error.value.code == 2
    assert not (tmp_path / "out.png").exists()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_public_dependency_cli_exports_pinned_versions_and_phase_hashes(model_id, tmp_path, capsys):
    root = tmp_path / "environment"
    dependencies.main(["--model-id", model_id, "--variant", "full", "--output-dir", str(root)])
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((root / "environment.json").read_text())
    expected = describe_model(model_id, full=True)["dependencies"]["profiles"]["full"]
    for phase in ("build", "inference", "checkpoint_build"):
        spec = printed[phase]
        assert spec["pip"] == expected[phase]["pip"]
        assert DIFFUSERS in spec["pip"] and "huggingface-hub==1.33.0" in spec["pip"]
        requirement_file = root / spec["requirements_file"]
        assert hashlib.sha256(requirement_file.read_bytes()).hexdigest() == spec["sha256"]
        assert DIFFUSERS in requirement_file.read_text()
        assert "work_dirs/" not in requirement_file.read_text()
    assert printed["status"] == "blocked_distribution"
    assert printed["validation"]["clean_install"] is False
    before = (root / "environment.json").read_bytes()
    with pytest.raises(FileExistsError):
        dependencies.main(["--model-id", model_id, "--output-dir", str(root)])
    assert (root / "environment.json").read_bytes() == before


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_installed_dependencies_provide_registered_qwen_capabilities(model_id):
    from xhmodel_merak._requirements import validate_worker_capabilities, validate_worker_requirements

    recipe = get_recipe(model_id)
    validate_worker_requirements((DIFFUSERS, "huggingface-hub==1.33.0"), model=model_id, operation="build")
    validate_worker_capabilities(recipe.worker_capabilities, model=model_id, operation="build")


@pytest.mark.parametrize("phase", ["build", "inference"])
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_ci_preparation_reads_only_selected_packages_from_recipe(model_id, phase, capsys):
    import prepare_dependencies

    prepare_dependencies.main(["--model-id", model_id, "--phase", phase,
                               "--packages", "diffusers", "huggingface_hub"])
    result = json.loads(capsys.readouterr().out)
    assert set(result["requirements"]) == {DIFFUSERS, "huggingface-hub==1.33.0"}
    assert result["command"][:4] == [sys.executable, "-m", "pip", "install"]
    assert set(result["command"][4:]) == set(result["requirements"])
    with pytest.raises(ValueError, match="not declared"):
        prepare_dependencies.select_requirements(model_id, phase, ["undeclared-package"])


def test_model_help_does_not_import_worker_packages():
    code = (
        "import runpy, sys\n"
        "sys.argv = ['example', '--help']\n"
        "try:\n"
        "    runpy.run_module('examples_merak.aigc.qwen_image21.unified', run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0\n"
        "assert not {'torch', 'transformers', 'xhquant', 'diffusers'} & sys.modules.keys()\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, check=True, timeout=20)
    assert "--negative-prompt" in result.stdout and "--guidance-scale" in result.stdout
