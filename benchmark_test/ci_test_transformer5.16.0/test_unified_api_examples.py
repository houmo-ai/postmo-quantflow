"""Exercise example CLIs and the real catalog without weights or runtime sessions.

Build/inference forwarding tests stub only expensive execution, bind arguments
against the real public signatures, and validate inference inputs with the real
schema. They do not claim model export or numerical inference coverage.
"""

import inspect
import json
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples_merak.unified_api import build, catalog, generate, synthesize
from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build import describe_model, list_models
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.model_inference.inputs import validate_inputs


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "qwen3.5-4b"


def test_synthesize_forwards_public_inputs_and_writes_pcm_wav(monkeypatch, tmp_path, capsys):
    import numpy as np

    signature = inspect.signature(synthesize.inference_model)
    calls = []

    def run(**kwargs):
        signature.bind(**kwargs)
        assert validate_inputs(get_recipe("kokoro-82m-v1.1-zh"), kwargs["inputs"])["speed"] == 1.2
        calls.append(kwargs)
        return SimpleNamespace(
            waveform=np.array([-2, -0.5, 0, 0.5, 2], np.float32),
            sample_rate=24000,
            duration_s=5 / 24000,
            segments=({"route": "t0032_f0160"},),
        )

    monkeypatch.setattr(synthesize, "inference_model", run)
    path = tmp_path / "audio/test.wav"
    synthesize.main(
        ["--artifact-dir", "artifact", "--text", "你好", "--output-wav", str(path), "--speed", "1.2", "--device", "cpu"]
    )
    assert calls == [
        {
            "artifact_dir": "artifact",
            "expected_model_id": "kokoro-82m-v1.1-zh",
            "inputs": {"text": "你好", "speed": 1.2},
            "device": "cpu",
        }
    ]
    with wave.open(str(path), "rb") as saved:
        assert (saved.getnchannels(), saved.getsampwidth(), saved.getframerate()) == (1, 2, 24000)
        np.testing.assert_array_equal(
            np.frombuffer(saved.readframes(5), dtype="<i2"), [-32767, -16384, 0, 16384, 32767]
        )
    summary = json.loads(capsys.readouterr().out)
    assert summary["output_wav"] == str(path) and summary["sample_rate"] == 24000


@pytest.mark.parametrize("name", ["catalog", "build", "generate", "synthesize"])
def test_examples_help_runs_from_repo_root_without_worker_imports(name):
    code = (
        "import runpy, sys\n"
        "sys.argv = ['example', '--help']\n"
        "try:\n"
        f"    runpy.run_module('examples_merak.unified_api.{name}', run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0\n"
        "assert not {'torch', 'transformers', 'xhquant'} & sys.modules.keys()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert "--help" in result.stdout


@pytest.mark.parametrize("args", [[], [MODEL_ID], [MODEL_ID, "--full"]])
def test_catalog_prints_real_public_result(args, capsys):
    catalog.main(args)
    result = json.loads(capsys.readouterr().out)
    expected = describe_model(MODEL_ID, full="--full" in args) if args else list_models()
    assert result == expected
    if args == [MODEL_ID]:
        assert "model_id" not in result
        assert result["quant_method"]["default"] == "autoround"
    elif args:
        assert result["model_id"] == MODEL_ID
        assert "input_schema" in result


@pytest.mark.parametrize(
    "module,args",
    [
        (catalog, ["--full"]),
        (build, []),
        (generate, []),
        (synthesize, []),
        (generate, ["--artifact-dir", "artifact", "--prompt", "hello", "--golden"]),
    ],
)
def test_invalid_cli_arguments_exit_before_execution(module, args):
    with pytest.raises(SystemExit) as error:
        module.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("reuse_checkpoint", [False, True])
def test_build_forwards_defaults_and_overrides(reuse_checkpoint, monkeypatch, capsys, tmp_path):
    signature = inspect.signature(build.build_model)
    calls = []
    model_dir = str(tmp_path / "model")
    output_dir = str(tmp_path / "artifact")
    checkpoint_dir = str(tmp_path / "quantized")

    def execute(**kwargs):
        signature.bind(**kwargs)
        calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {"output_dir": kwargs["output_dir"], "loadable": True})

    monkeypatch.setattr(build, "build_model", execute)
    args = ["--model-id", MODEL_ID, "--model-dir", model_dir, "--output-dir", output_dir]
    if reuse_checkpoint:
        args += [
            "--quant-checkpoint",
            checkpoint_dir,
            "--context-length",
            "4096",
            "--prefill-chunk-length",
            "256",
            "--device",
            "cuda:1",
        ]
    build.main(args)

    assert calls == [
        {
            "model_id": MODEL_ID,
            "model_dir": model_dir,
            "output_dir": output_dir,
            "device": "cuda:1" if reuse_checkpoint else None,
            "quant_checkpoint": checkpoint_dir if reuse_checkpoint else None,
            "context_length": 4096 if reuse_checkpoint else None,
            "prefill_chunk_length": 256 if reuse_checkpoint else None,
        }
    ]
    assert json.loads(capsys.readouterr().out) == {"output_dir": output_dir, "loadable": True}
    assert not (tmp_path / "artifact").exists()


@pytest.mark.parametrize("with_image", [False, True])
def test_generate_forwards_inputs_and_explicit_golden(with_image, monkeypatch, capsys, tmp_path):
    signature = inspect.signature(generate.inference_model)
    calls = []
    artifact_dir = str(tmp_path / "artifact")
    image_path = str(tmp_path / "image.jpg")
    inputs = {"prompt": "图里有什么？", "max_new_tokens": 64}

    def execute(**kwargs):
        signature.bind(**kwargs)
        assert validate_inputs(get_recipe(MODEL_ID), kwargs["inputs"]) == kwargs["inputs"]
        calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {"output_text": "示例回答", "golden_components": kwargs["golden"] or []})

    monkeypatch.setattr(generate, "inference_model", execute)
    args = ["--artifact-dir", artifact_dir, "--prompt", inputs["prompt"]]
    if with_image:
        inputs.update(image=image_path, max_new_tokens=128)
        args += [
            "--image",
            image_path,
            "--max-new-tokens",
            "128",
            "--device",
            "cuda:1",
            "--golden",
            "visual",
            "prefill",
            "decode",
        ]
    generate.main(args)

    golden = ["visual", "prefill", "decode"] if with_image else None
    assert calls == [
        {
            "artifact_dir": artifact_dir,
            "inputs": inputs,
            "device": "cuda:1" if with_image else None,
            "golden": golden,
        }
    ]
    assert json.loads(capsys.readouterr().out) == {"output_text": "示例回答", "golden_components": golden or []}


def test_catalog_rejects_unknown_model():
    with pytest.raises(ModelError):
        catalog.main(["not-a-registered-model"])


def test_build_rejects_missing_model_with_real_api(tmp_path):
    with pytest.raises(ModelError):
        build.main(
            [
                "--model-id",
                MODEL_ID,
                "--model-dir",
                str(tmp_path / "missing-model"),
                "--output-dir",
                str(tmp_path / "artifact"),
            ]
        )
    assert not (tmp_path / "artifact").exists()


def test_generate_rejects_missing_artifact_with_real_api(tmp_path):
    with pytest.raises(ModelError):
        generate.main(["--artifact-dir", str(tmp_path / "missing-artifact"), "--prompt", "hello"])
