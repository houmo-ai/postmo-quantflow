import json
from pathlib import Path

from xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter import (
    _resolve_repo_resource,
    _restore_unified_audio_config,
)


def test_repo_calibration_resource_cannot_be_shadowed_by_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("XH2MODELZOO_ROOT", raising=False)
    monkeypatch.delenv("XH2MODELZOO_DATA_ROOT", raising=False)
    shadow = tmp_path / "data/calib_data/NeelNanda-pile-10k.jsonl"
    shadow.parent.mkdir(parents=True)
    shadow.write_text('{"text":"shadow"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    resolved = Path(_resolve_repo_resource("xh2modelzoo://data/calib_data/NeelNanda-pile-10k.jsonl"))

    assert resolved != shadow.resolve()
    assert resolved == (Path(__file__).resolve().parents[2] / "data/calib_data/NeelNanda-pile-10k.jsonl").resolve()


def test_gptq_unified_audio_dimensions_survive_config_save(tmp_path):
    source = tmp_path / "source"
    quantized = tmp_path / "quantized"
    source.mkdir()
    quantized.mkdir()
    (source / "config.json").write_text(json.dumps({
        "audio_config": {"hidden_size": 640, "output_proj_dims": 640, "audio_samples_per_token": 640},
    }), encoding="utf-8")
    (quantized / "config.json").write_text(json.dumps({
        "audio_config": {"audio_samples_per_token": 640, "hidden_size": 512},
        "quantization_config": {"bits": 4},
    }), encoding="utf-8")

    _restore_unified_audio_config(str(source), str(quantized))

    output = json.loads((quantized / "config.json").read_text(encoding="utf-8"))
    assert output["audio_config"] == {
        "audio_samples_per_token": 640,
        "hidden_size": 512,
        "output_proj_dims": 640,
    }
    assert output["quantization_config"] == {"bits": 4}
