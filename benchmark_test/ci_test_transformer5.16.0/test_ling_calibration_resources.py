"""Exercise calibration URI translation at the ModelZoo producer boundary."""
from types import SimpleNamespace

import pytest

from xhmodel_merak.xh_llm.models.ling_3_flash.quant_adapter import (
    Ling3FlashQuantSpec,
    autoround_calibration_source,
    quantize_ling3_flash,
)


def test_autoround_resolves_packaged_calibration_uri(monkeypatch, tmp_path):
    calibration = tmp_path / "data/calib_data/NeelNanda-pile-10k.jsonl"
    calibration.parent.mkdir(parents=True)
    calibration.write_text('{"text":"sample"}\n', encoding="utf-8")
    monkeypatch.setenv("XH2MODELZOO_ROOT", str(tmp_path))
    monkeypatch.delenv("XH2MODELZOO_DATA_ROOT", raising=False)
    spec = Ling3FlashQuantSpec(
        model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "output"),
        method="autoround",
        calibration_jsonl="xh2modelzoo://data/calib_data/NeelNanda-pile-10k.jsonl",
    )

    assert autoround_calibration_source(spec, None) == str(calibration.resolve())


@pytest.mark.parametrize("method", ["gptq", "autoround"])
def test_recipe_resolves_packaged_calibration_uri(monkeypatch, tmp_path, method):
    import gptqmodel.recipes.ling3_flash as recipe

    calibration = tmp_path / "data/calib_data/NeelNanda-pile-10k.jsonl"
    calibration.parent.mkdir(parents=True)
    calibration.write_text('{"text":"sample"}\n', encoding="utf-8")
    monkeypatch.setenv("XH2MODELZOO_ROOT", str(tmp_path))
    monkeypatch.delenv("XH2MODELZOO_DATA_ROOT", raising=False)
    captured = {}

    def record_recipe(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_dir=kwargs["output_dir"])

    monkeypatch.setattr(
        recipe,
        "quantize_ling3_flash",
        record_recipe,
    )
    spec = Ling3FlashQuantSpec(
        model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "output"),
        method=method,
        calibration_jsonl="xh2modelzoo://data/calib_data/NeelNanda-pile-10k.jsonl",
    )

    quantize_ling3_flash(spec)

    assert captured["calibration_jsonl"] == str(calibration.resolve())
    assert captured["mla_weight_layout"] == "xh2_absorbed_v1"
    assert captured["nsamples"] == (256 if method == "autoround" else 128)
    if method == "autoround":
        assert captured["data_type"] == "int_sym_xh"
