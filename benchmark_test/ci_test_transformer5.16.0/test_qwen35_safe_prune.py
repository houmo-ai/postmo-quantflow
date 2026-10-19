"""SAFE artifact packing, phase isolation and real tiny MoE outputs."""

import pytest
import torch
from test_qwen35_decode_prune import Body, RoutedLayer

from xhmodel_merak.xh_llm.models.qwen3_5_moe.dynamic_prune import (
    configure_decode_graph,
    normalize_dynamic_prune,
    prepare_scalars,
)
from xhmodel_merak.xh_llm.models.qwen3_5_moe.safe_prune import pack_safe_data
from xhquant.common.moe_safe import safe_prune_route


def test_fp16_pack_disables_unsafe_loss_before_rounding():
    loss = torch.tensor([[0.0, 1.0001], [1.0, 0.0]])
    data = pack_safe_data(torch.ones(2), torch.ones(2), torch.ones(2, 2), loss)
    assert data.dtype == torch.float16 and torch.isfinite(data).all()
    assert data[0, 2] == 0 and data[0, 4] == 65504
    assert data[1, 1] == 1 and data[1, 3] == 1


def test_safe_packing_is_owned_by_modelzoo():
    assert pack_safe_data.__module__ == "xhmodel_merak.xh_llm.models.qwen3_5_moe.safe_prune"


def test_packing_fuses_priors_once_and_preserves_pair_tables():
    amp, aimer = torch.tensor([1.0, 4.0, 9.0]), torch.tensor([4.0, 9.0, 16.0])
    coeff, loss = torch.arange(1, 10).float().reshape(3, 3), torch.zeros(3, 3)
    packed = pack_safe_data(amp, aimer, coeff, loss)
    assert packed.shape == (3, 7)
    torch.testing.assert_close(packed[:, 0], torch.tensor([2.0, 6.0, 12.0]).half(), rtol=0, atol=0)
    coeff.diagonal().fill_(1)
    loss.diagonal().fill_(65504)
    torch.testing.assert_close(packed[:, 1:4], coeff.half(), rtol=0, atol=0)
    torch.testing.assert_close(packed[:, 4:7], loss.half(), rtol=0, atol=0)


def test_packing_normalizes_forbidden_pairs_without_mutating_inputs():
    amp, aimer = torch.tensor([1.0, 4.0]), torch.tensor([4.0, 9.0])
    coeff = torch.tensor([[3.0, 7.0], [2.0, 4.0]])
    loss = torch.tensor([[0.2, 1e30], [0.3, 0.4]])
    originals = [value.clone() for value in (amp, aimer, coeff, loss)]
    packed = pack_safe_data(amp, aimer, coeff, loss)
    expected = torch.tensor([[2.0, 1.0, 0.0, 65504, 65504], [6.0, 2.0, 1.0, 0.3, 65504]]).half()
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    assert packed.is_contiguous() and packed.dtype == torch.float16 and packed.device.type == "cpu"
    for original, actual in zip(originals, (amp, aimer, coeff, loss), strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("value", [-1.0, 0.0, float("inf"), float("nan")])
def test_fusion_rejects_invalid_raw_priors(value):
    with pytest.raises(ValueError, match="priors"):
        pack_safe_data(torch.tensor([value]), torch.tensor([value]), torch.ones(1, 1), torch.zeros(1, 1))


def test_fusion_avoids_intermediate_product_overflow_and_underflow():
    prior = torch.tensor([1e3, 1e-3])
    packed = pack_safe_data(prior, prior, torch.ones(2, 2), torch.zeros(2, 2))
    torch.testing.assert_close(packed[:, 0], prior.half(), rtol=2e-3, atol=0)


@pytest.mark.parametrize("value", [1e30, 1e-30])
def test_fusion_rejects_priors_outside_fp16_range(value):
    with pytest.raises(ValueError, match="float16"):
        pack_safe_data(torch.tensor([value]), torch.ones(1), torch.ones(1, 1), torch.zeros(1, 1))


def test_packing_rejects_active_coefficients_overflowing_fp16():
    with pytest.raises(ValueError, match="finite"):
        pack_safe_data(torch.ones(2), torch.ones(2), torch.full((2, 2), 1e10), torch.zeros(2, 2))


@pytest.mark.parametrize("option", [{"max_projection_loss": 0.7}, {"even_policy": "none"}, {"renormalize": False}])
def test_packer_no_longer_accepts_control_parameters(option):
    with pytest.raises(TypeError):
        pack_safe_data(torch.ones(4), torch.ones(4), torch.ones(4, 4), torch.zeros(4, 4), **option)


def test_packing_does_not_depend_on_default_activation_dtype():
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        data = pack_safe_data(torch.ones(4), torch.ones(4), torch.ones(4, 4), torch.zeros(4, 4))
    finally:
        torch.set_default_dtype(original)
    assert data.dtype == torch.float16
    assert data.shape == (4, 9)
    assert data[0, 5].item() == 65504


def artifact_config(tmp_path):
    priors = tmp_path / "priors.pt"
    artifact = tmp_path / "exfold.pt"
    torch.save({"amp": {i: torch.ones(4) for i in range(2)}, "aimer": {i: torch.ones(4) for i in range(2)}}, priors)
    torch.save({"coeff": torch.ones(2, 4, 4), "loss": torch.zeros(2, 4, 4), "expert_norms": torch.ones(2, 4)}, artifact)
    return {
        "mode": "decode",
        "algorithm": "safe",
        "threshold": 0.85,
        "ace_priors_path": str(priors),
        "exfold_artifact_path": str(artifact),
        "even_fold_policy": "nearest_mass",
    }


def test_safe_artifacts_pack_into_existing_scalar_and_preserve_prefill(tmp_path):
    from xhquant.api import to_frontend_graph

    config = normalize_dynamic_prune(artifact_config(tmp_path))
    body = Body()
    scalars, path = prepare_scalars(body, config, tmp_path, tmp_path / "export")
    assert set(scalars) == {"0", "1"} and scalars["0"].shape == (4, 9)
    payload = torch.load(path, weights_only=True)
    assert payload["method"] == "safe_exfold_v4"
    assert len(payload["ace_priors_sha256"]) == len(payload["exfold_sha256"]) == 64
    block = body.layers[0].mlp.moeblock
    hidden = torch.tensor([[[0.4, -0.2, 0.7, 0.1]]])
    routes = torch.tensor([[[0.5, 0.26, 0.18, 0.06]]])
    dense = block(hidden, routes)
    prefill = to_frontend_graph(RoutedLayer(block), "TorchFX", (hidden, routes))
    decode = to_frontend_graph(RoutedLayer(block), "TorchFX", (hidden, routes))
    configure_decode_graph({"prefill": prefill, "decode": decode}, {"0": scalars["0"]}, 0.85)
    p, d = prefill.get_submodule("moeblock"), decode.get_submodule("moeblock")
    assert p.s_scalar is None and p.prune_threshold is None
    torch.testing.assert_close(p(hidden, routes), dense)
    weights, ids, capacity = safe_prune_route(routes.reshape(1, 4), scalars["0"], 4, 0.85, True)
    assert capacity == 2
    # Dense execution with explicit folded gate weights is the output oracle.
    folded_routes = torch.zeros(1, 4, dtype=weights.dtype).scatter_add(1, ids, weights).reshape_as(routes)
    torch.testing.assert_close(d(hidden, routes), p(hidden, folded_routes.float()), rtol=2e-3, atol=1e-6)
    original = d.s_scalar.clone()
    d.half()
    assert d.s_scalar.dtype == torch.float16
    torch.testing.assert_close(d.s_scalar, original, rtol=0, atol=0)


def test_safe_requires_both_priors_and_projection_artifact(tmp_path):
    config = artifact_config(tmp_path)
    del config["exfold_artifact_path"]
    with pytest.raises(ValueError, match="SAFE"):
        prepare_scalars(Body(), normalize_dynamic_prune(config), tmp_path, tmp_path)


def test_safe_rejects_missing_layers_and_wrong_expert_count(tmp_path):
    config = artifact_config(tmp_path)
    for payload in (
        {"amp": {0: torch.ones(4)}, "aimer": {0: torch.ones(4)}},
        {"amp": {i: torch.ones(3) for i in range(2)}, "aimer": {i: torch.ones(3) for i in range(2)}},
    ):
        torch.save(payload, config["ace_priors_path"])
        with pytest.raises(ValueError):
            prepare_scalars(Body(), normalize_dynamic_prune(config), tmp_path, tmp_path)


def test_safe_cache_cannot_be_reused_with_other_settings_or_checkpoint(tmp_path):
    config = normalize_dynamic_prune(artifact_config(tmp_path))
    _, path = prepare_scalars(Body(), config, tmp_path, tmp_path / "export")
    cached = normalize_dynamic_prune({"algorithm": "safe", "s_scalar_path": path})
    restored, _ = prepare_scalars(Body(), cached, tmp_path, tmp_path / "export")
    assert restored["0"].shape == (4, 9)
    with pytest.raises(ValueError, match="settings"):
        prepare_scalars(Body(), {**cached, "even_fold_policy": "floor"}, tmp_path, tmp_path / "export")
    (tmp_path / "config.json").write_text('{"different_checkpoint": true}')
    with pytest.raises(ValueError, match="checkpoint"):
        prepare_scalars(Body(), cached, tmp_path, tmp_path / "export")


def test_safe_does_not_silently_ignore_explicit_cache(tmp_path):
    config = normalize_dynamic_prune({**artifact_config(tmp_path), "s_scalar_path": str(tmp_path / "cached.pt")})
    with pytest.raises(ValueError, match="both|either|exclusive"):
        prepare_scalars(Body(), config, tmp_path, tmp_path / "export")


@pytest.mark.parametrize(
    "old_version,old_name",
    [("v1", "moe_safe_s_scalar.pt"), ("v2", "moe_safe_s_scalar_v2.pt"), ("v3", "moe_safe_s_scalar_v3.pt")],
)
def test_safe_fused_cache_versions_and_preserves_old_default(tmp_path, old_version, old_name):
    config = normalize_dynamic_prune(artifact_config(tmp_path))
    priors = {
        "amp": {i: torch.tensor([1.0, 4.0, 9.0, 16.0]) for i in range(2)},
        "aimer": {i: torch.tensor([4.0, 9.0, 16.0, 25.0]) for i in range(2)},
    }
    torch.save(priors, config["ace_priors_path"])
    old_path = tmp_path / old_name
    torch.save({"method": f"safe_exfold_{old_version}"}, old_path)
    old_bytes = old_path.read_bytes()
    scalars, path = prepare_scalars(Body(), config, tmp_path, tmp_path)
    assert path == str(tmp_path / "moe_safe_s_scalar_v4.pt")
    assert old_path.read_bytes() == old_bytes
    torch.testing.assert_close(scalars["0"][:, 0], torch.tensor([2.0, 6.0, 12.0, 20.0]).half(), rtol=0, atol=0)
    cached = normalize_dynamic_prune({"algorithm": "safe", "s_scalar_path": str(old_path)})
    with pytest.raises(ValueError, match="rebuild"):
        prepare_scalars(Body(), cached, tmp_path, tmp_path)


def test_safe_packed_tables_are_fp16_under_half_default(tmp_path):
    config = normalize_dynamic_prune(artifact_config(tmp_path))
    body = Body()
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        scalars, _ = prepare_scalars(body, config, tmp_path, tmp_path / "export")
    finally:
        torch.set_default_dtype(original)
    assert scalars["0"].dtype == torch.float16
    assert scalars["0"].shape == (4, 9)
    assert scalars["0"][0, 5].item() == 65504


@pytest.mark.parametrize(
    "option",
    [{"even_fold_policy": "none"}, {"even_fold_policy": "floor"}, {"max_projection_loss": 0.7}],
)
def test_safe_rejects_nonfixed_controls_at_config_and_load(tmp_path, option):
    config = {**artifact_config(tmp_path), **option}
    with pytest.raises(ValueError, match="fixed"):
        normalize_dynamic_prune(config)
    with pytest.raises(ValueError, match="fixed"):
        prepare_scalars(Body(), config, tmp_path, tmp_path / "export")


@pytest.mark.parametrize(
    "options",
    [
        {"algorithm": "typo"},
        {"algorithm": "safe", "even_fold_policy": "odd"},
        {"algorithm": "safe", "max_projection_loss": -1},
    ],
)
def test_invalid_safe_configuration(options):
    with pytest.raises(ValueError):
        normalize_dynamic_prune(options)


def test_safe_configuration_rejects_unsupported_chip():
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig

    with pytest.raises(ValueError, match="SAFE.*XH2a"):
        XHQwen3_5MoeModelConfig(model_name="tiny", chip_arch="XH3", dynamic_prune={"algorithm": "safe"})
    config = XHQwen3_5MoeModelConfig(model_name="tiny", chip_arch="XH2a", dynamic_prune={"algorithm": "safe"})
    assert config.dynamic_prune["even_fold_policy"] == "nearest_mass"


def test_safe_cli_parses_explicit_artifacts_and_fixed_policy(monkeypatch):
    import sys

    from examples_merak.llm.qwen3_5.qwen3_5_dynamic_prune_workflow import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflow",
            "--model-dir",
            "checkpoint",
            "--config-path",
            "recipe.yaml",
            "--prune-algorithm",
            "safe",
            "--ace-priors-path",
            "priors.pt",
            "--exfold-artifact-path",
            "exfold.pt",
            "--coverage",
            "0.875",
            "--even-fold-policy",
            "nearest_mass",
            "--max-projection-loss",
            "1.0",
        ],
    )
    args = parse_args()
    assert args.prune_algorithm == "safe" and args.threshold == 0.875
    assert args.ace_priors_path == "priors.pt" and args.exfold_artifact_path == "exfold.pt"
    assert args.even_fold_policy == "nearest_mass" and args.max_projection_loss == 1.0


@pytest.mark.parametrize("option", [["--even-fold-policy", "floor"], ["--max-projection-loss", "0.7"]])
def test_safe_cli_rejects_nonfixed_controls(monkeypatch, option):
    import sys

    from examples_merak.llm.qwen3_5.qwen3_5_dynamic_prune_workflow import parse_args

    monkeypatch.setattr(sys, "argv", ["workflow", "--model-dir", "checkpoint", "--config-path", "recipe.yaml", *option])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_6_35b_a3b_dynamic_prune.yaml",
        "qwen3_6_35b_a3b_full_head_w4_decode_prune_256k_fa.yaml",
    ],
)
def test_prune_workflows_default_to_safe_and_accept_artifact_overrides(name):
    from pathlib import Path

    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    path = Path(__file__).resolve().parents[2] / ("configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b") / name
    config = WorkflowConfig.from_file(str(path))
    model = config.export["model"]
    prune = normalize_dynamic_prune(model["dynamic_prune"])
    assert prune["algorithm"] == "safe" and prune["mode"] == "decode"
    assert prune["threshold"] == 0.85 and prune["auto_s_scalar"]
    assert "safe" in model["model_name"] and not model["only_first_block"]
    updated = config.with_overrides(
        {
            "export.model.dynamic_prune.algorithm": "safe",
            "export.model.dynamic_prune.ace_priors_path": "priors.pt",
            "export.model.dynamic_prune.exfold_artifact_path": "exfold.pt",
            "export.model.dynamic_prune.max_projection_loss": 1.0,
            "export.model.dynamic_prune.even_fold_policy": "nearest_mass",
        }
    )
    assert updated.export["model"]["dynamic_prune"]["ace_priors_path"] == "priors.pt"
    assert model["dynamic_prune"]["ace_priors_path"] is None


def test_default_safe_yaml_can_switch_to_acep(name="qwen3_6_35b_a3b_dynamic_prune.yaml"):
    from pathlib import Path

    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    path = Path(__file__).resolve().parents[2] / ("configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b") / name
    switched = WorkflowConfig.from_file(str(path)).with_overrides({"export.model.dynamic_prune.algorithm": "acep"})
    acep = normalize_dynamic_prune(switched.export["model"]["dynamic_prune"])
    assert acep["algorithm"] == "acep" and acep["threshold"] == 0.85
    assert not any(
        key in acep
        for key in ("ace_priors_path", "exfold_artifact_path", "even_fold_policy", "max_projection_loss", "calibration")
    )
    with pytest.raises(ValueError, match="SAFE options require algorithm=safe"):
        normalize_dynamic_prune({"algorithm": "acep", "ace_priors_path": "priors.pt"})


def test_prune_cli_defaults_select_safe_workflow(monkeypatch):
    import sys

    from examples_merak.llm.qwen3_5.qwen3_5_dynamic_prune_workflow import parse_args
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    monkeypatch.setattr(sys, "argv", ["workflow", "--model-dir", "checkpoint"])
    args = parse_args()
    config = WorkflowConfig.from_file(args.config_path)
    assert config.export["model"]["dynamic_prune"]["algorithm"] == "safe"
    assert args.prune_algorithm is None  # Preserve explicit YAML choices.
    assert args.export_output_dir == "work_dirs/qwen3_5_safe_export"


def test_safe_preflight_requires_real_files_and_exclusive_inputs(tmp_path):
    from examples_merak.llm.qwen3_5.qwen3_5_dynamic_prune_workflow import _validate_safe_inputs

    prune = {"algorithm": "safe", "auto_s_scalar": False}
    with pytest.raises(ValueError, match="requires"):
        _validate_safe_inputs(prune)
    priors, exfold = tmp_path / "priors.pt", tmp_path / "exfold.pt"
    prune.update(ace_priors_path=str(priors), exfold_artifact_path=str(exfold))
    with pytest.raises(FileNotFoundError, match="SAFE calibration"):
        _validate_safe_inputs(prune)
    priors.touch()
    with pytest.raises(FileNotFoundError, match="exfold.pt"):
        _validate_safe_inputs(prune)
    exfold.touch()
    _validate_safe_inputs(prune)
    with pytest.raises(ValueError, match="not both"):
        _validate_safe_inputs({**prune, "s_scalar_path": str(priors)})
    _validate_safe_inputs({"algorithm": "safe", "s_scalar_path": str(priors)})
    _validate_safe_inputs({"algorithm": "safe", "auto_s_scalar": True})
    _validate_safe_inputs({"algorithm": "acep"})
    _validate_safe_inputs({"algorithm": "safe", "mode": "off"})


def test_safe_missing_artifacts_preserve_existing_export(monkeypatch, tmp_path):
    import sys

    from examples_merak.llm.qwen3_5.qwen3_5_dynamic_prune_workflow import main

    output = tmp_path / "previous_export"
    output.mkdir()
    previous = output / "keep.txt"
    previous.write_text("previous export")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflow",
            "--model-dir",
            "checkpoint",
            "--export-output-dir",
            str(output),
            "--overwrite",
            "--no-auto-s-scalar",
        ],
    )
    with pytest.raises(ValueError, match="SAFE requires"):
        main()
    assert previous.read_text() == "previous export"
