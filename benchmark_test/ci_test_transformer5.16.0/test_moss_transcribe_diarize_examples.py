"""CLI contracts for MOSS example scripts. No HMONNX sessions or HF weights."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples_merak.asr.moss_transcribe_diarize import compare_pytorch
from examples_merak.asr.moss_transcribe_diarize import moss_transcribe_diarize_hmonnx_infer as infer


REPO_ROOT = Path(__file__).resolve().parents[2]


def _meta() -> dict:
    geometry = {
        "hmonnx_file": "prefill/model.onnx",
        "onnx_file": None,
        "cache_mode": "inplace",
        "prefill_length": 4096,
        "cache_length": 65536,
        "rope_mode": "fp32_table_integer_index",
        "enable_page_attention": False,
        "quant_type": "w8a8_sefp",
    }
    return {
        "quant_type": "w8a8_sefp",
        "lm_head_file": "lm_head.pt",
        "components": {
            "encode": {
                "hmonnx_file": "encode/hmquant_moss_transcribe_diarize_09b_encode_with_act.onnx",
                "quant_type": "w8a8_sefp",
            },
            "prefill_decode": {
                "prefill": dict(geometry),
                "decode": dict(geometry, hmonnx_file="decode/model.onnx"),
            },
        },
    }


def test_hmonnx_infer_inspects_metadata_without_opening_sessions(tmp_path, capsys):
    (tmp_path / "export_meta_info.json").write_text(json.dumps(_meta()), encoding="utf-8")
    infer.main(["--artifact-dir", str(tmp_path)])
    output = capsys.readouterr().out
    assert "published MOSS HMONNX plan" in output
    assert "lm_head.pt" in output
    assert "CacheTensor" in output
    assert "No --audio given" in output


def test_hmonnx_infer_nested_metadata_resolves_package_root(tmp_path, capsys):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "export_meta_info.json").write_text(json.dumps(_meta()), encoding="utf-8")
    infer.main(["--artifact-dir", str(tmp_path)])
    assert f"artifact root: {package}" in capsys.readouterr().out


def test_hmonnx_infer_forwards_audio_to_transcribe_artifact(monkeypatch, tmp_path, capsys):
    (tmp_path / "export_meta_info.json").write_text(json.dumps(_meta()), encoding="utf-8")
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake")
    calls = []

    def fake_run(root, meta, audio_path, device, prompt, max_new_tokens, enable_page_attention):
        calls.append(
            {
                "root": root,
                "audio": audio_path,
                "device": device,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "enable_page_attention": enable_page_attention,
            }
        )
        return "[S01] demo"

    monkeypatch.setattr(infer, "_run_hmonnx", fake_run)
    infer.main(
        [
            "--artifact-dir",
            str(tmp_path),
            "--audio",
            str(audio),
            "--device",
            "cuda:1",
            "--prompt",
            "转写",
            "--max-new-tokens",
            "32",
            "--enable-page-attention",
        ]
    )
    assert calls == [
        {
            "root": tmp_path.resolve(),
            "audio": audio,
            "device": "cuda:1",
            "prompt": "转写",
            "max_new_tokens": 32,
            "enable_page_attention": True,
        }
    ]
    assert "=== transcript ===" in capsys.readouterr().out


def test_hmonnx_infer_rejects_conflicting_page_attention_flags(tmp_path):
    (tmp_path / "export_meta_info.json").write_text(json.dumps(_meta()), encoding="utf-8")
    with pytest.raises(SystemExit, match="page-attention"):
        infer.main(
            [
                "--artifact-dir",
                str(tmp_path),
                "--enable-page-attention",
                "--disable-page-attention",
            ]
        )


@pytest.mark.parametrize(
    "module",
    [
        "examples_merak.asr.moss_transcribe_diarize.moss_transcribe_diarize_hmonnx_infer",
        "examples_merak.asr.moss_transcribe_diarize.compare_pytorch",
    ],
)
def test_moss_example_help_runs_without_worker_imports(module):
    code = (
        "import runpy, sys\n"
        "sys.argv = ['example', '--help']\n"
        "try:\n"
        f"    runpy.run_module({module!r}, run_name='__main__')\n"
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


def test_compare_and_infer_close_borrowed_runtimes():
    infer_src = Path(infer.__file__).read_text(encoding="utf-8")
    compare_src = Path(compare_pytorch.__file__).read_text(encoding="utf-8")
    assert "finally:" in infer_src and "runtime.close()" in infer_src
    assert "finally:" in compare_src and "runtime.close()" in compare_src


def test_compare_texts_reports_prefix_and_agreement():
    summary = compare_pytorch.compare_texts("abcd", "abxy")
    assert summary["prefix_chars"] == 2
    assert summary["agree_chars"] == 2
    assert summary["pytorch_chars"] == 4
    assert summary["hmonnx_chars"] == 4


def test_compare_pytorch_forwards_stubbed_backends(monkeypatch, capsys, tmp_path):
    calls = []

    def pytorch(audio, model_dir, device, prompt, max_new_tokens):
        calls.append(("pt", str(audio), str(model_dir), device, prompt, max_new_tokens))
        return "abcd"

    def hmonnx(audio, artifact_dir, device, prompt, enable_page_attention, max_new_tokens=None):
        calls.append(
            ("hm", str(audio), str(artifact_dir), device, prompt, enable_page_attention, max_new_tokens)
        )
        return "abxy"

    monkeypatch.setattr(compare_pytorch, "transcribe_pytorch", pytorch)
    monkeypatch.setattr(compare_pytorch, "transcribe_hmonnx", hmonnx)
    compare_pytorch.main(
        [
            "--audio",
            "clip.wav",
            "--artifact-dir",
            "artifact",
            "--model-dir",
            "hf",
            "--device",
            "cuda:0",
            "--disable-page-attention",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert calls == [
        ("pt", "clip.wav", "hf", "cuda:0", None, None),
        ("hm", "clip.wav", "artifact", "cuda:0", None, False, None),
    ]
    assert payload["prefix_chars"] == 2
    assert payload["pytorch_text"] == "abcd"
