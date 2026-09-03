from pathlib import Path

from xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter import _resolve_repo_resource


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
