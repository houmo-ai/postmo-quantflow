"""Small real phase/LSTM exports; full Kokoro audio validation lives in the demo."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from torch import nn

from xhmodel_merak.xh_other_model.models.kokoro.bucketed_runtime import (
    GRAPH_MODE,
    SCHEMA_VERSION,
    KokoroBucketedRuntime,
)
from xhmodel_merak.xh_other_model.models.kokoro.buckets import (
    BUCKET_ROUTES,
    normalize_bucket_routes,
)
from xhmodel_merak.xh_other_model.models.kokoro.export import _export_partition
from xhmodel_merak.xh_other_model.models.kokoro.golden import GoldenCpuRunner, GoldenHmonnxRunner
from xhmodel_merak.xh_other_model.models.kokoro.graph import PrefixReversedBidirectionalLSTMStatic
from xhmodel_merak.xh_other_model.models.kokoro.host import make_reverse_idx
from xhmodel_merak.xh_other_model.models.kokoro.independent_split import (
    PhaseCoreStatic,
    run_duration_alignment_host,
)
from xhmodel_merak.xh_other_model.models.kokoro.quant_export import (
    _convert_hmonnx,
    _hmonnx_artifact_metadata,
)
from xhmodel_merak.xh_other_model.models.kokoro.release import component_path, release_prefix
from xhmodel_merak.xh_other_model.models.kokoro.runtime import OrtRunner
from xhmodel_merak.xh_other_model.models.kokoro.sine_source import StaticSineSource
from xhmodel_merak.xh_other_model.models.kokoro.workflow import _validate_export_config
from xhmodel_merak.xh_other_model.workflows.config import WorkflowConfig


def recipe():
    return {
        "target_device": "XH2a",
        "components": {
            role: {"quant_type": "w16a16_sefp"} for role in ("text_duration", "frame_acoustic", "generator_istft")
        },
    }


@pytest.mark.parametrize("fail_import", [False, True])
def test_pinned_source_import_preserves_input_fingerprint(tmp_path, monkeypatch, fail_import):
    import sys
    from types import SimpleNamespace

    from xhmodel_merak.model_build.fingerprints import resource_sha256
    from xhmodel_merak.xh_other_model.models.kokoro.graph import load_official_model

    source = tmp_path / "kokoro"
    source.mkdir()
    (source / "__init__.py").write_text('__version__ = "0.9.4"\n')
    (source / "model.py").write_text(
        'raise ImportError("missing test dependency")\n' if fail_import else
        'from torch import nn\nclass KModel(nn.Module):\n'
        '    def __init__(self, **kwargs):\n        super().__init__()\n'
    )
    for name in list(sys.modules):
        if name == "kokoro" or name.startswith("kokoro."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    before = resource_sha256(tmp_path)
    assets = SimpleNamespace(source_root=tmp_path, config=tmp_path / "config.json",
                             checkpoint=tmp_path / "model.pth")
    try:
        if fail_import:
            with pytest.raises(ImportError, match="pinned upstream"):
                load_official_model(assets)
        else:
            assert not load_official_model(assets).training
        assert sys.dont_write_bytecode is False
        assert resource_sha256(tmp_path) == before
        assert not list(tmp_path.rglob("*.pyc"))
    finally:
        for name in list(sys.modules):
            if name == "kokoro" or name.startswith("kokoro."):
                sys.modules.pop(name)


def test_kokoro_checked_in_recipe_matches_release_contract():
    root = Path(__file__).resolve().parents[2]
    config = WorkflowConfig.from_file(
        str(root / "configs_merak/workflows/xh2a/other_models/kokoro/kokoro_xh2a_bucketed_w16.yaml")
    )
    values = config.build_export_dict()
    _validate_export_config(values)
    assert config.quant is None
    assert normalize_bucket_routes(values["token_buckets"], values["audio_seconds_buckets"]) == BUCKET_ROUTES
    assert values["seed"] == 1234
    assert values["convert_hmonnx"] is True


@pytest.mark.parametrize("option", ("lstm_variants", "decompose_lstm", "phase_on_npu", "validate_onnx"))
def test_kokoro_rejects_retired_options(option):
    _validate_export_config(recipe())
    with pytest.raises(ValueError, match="obsolete"):
        _validate_export_config({**recipe(), option: False})


def test_kokoro_release_names_and_paired_routes(tmp_path):
    routes = normalize_bucket_routes(None, None)
    assert routes == BUCKET_ROUTES
    assert [(r.token_max_length, r.frame_max_length) for r in routes] == [(32, 160), (64, 320), (128, 640), (256, 1280)]
    prefix = release_prefix(routes, "20260908")
    assert prefix == "hmquant_xh2_kokoro-82m-v1.1-zh_w16a16_t32-64-128-256_f160-320-640-1280_20260908"
    path = component_path(tmp_path, prefix, "text_duration", "t0032", reference=False)
    assert path.parent == tmp_path / "text_duration_t0032"
    assert path.name == f"{prefix}_text_duration_t0032_with_act.onnx"
    assert component_path(tmp_path, prefix, "text_duration", "t0032", reference=True).parent.name == "reference_onnx"
    with pytest.raises(ValueError):
        release_prefix(routes, "20260230")
    with pytest.raises(ValueError):
        normalize_bucket_routes([32], [8])


def test_kokoro_runtime_rejects_old_and_incomplete_metadata(tmp_path):
    with pytest.raises(ValueError, match="schema"):
        KokoroBucketedRuntime(tmp_path, {"graph_mode": "bucketed_precision_split"}, backend="ort", device="cpu")
    with pytest.raises(ValueError, match="incomplete"):
        KokoroBucketedRuntime(
            tmp_path, {"graph_mode": GRAPH_MODE, "schema_version": SCHEMA_VERSION}, backend="ort", device="cpu"
        )


def test_kokoro_cpu_expansion_is_indexed_copy_with_zero_padding():
    logits = torch.tensor([[[0.0, 0.0], [10.0, 10.0], [0.0, 0.0], [10.0, 10.0]]])
    features = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    text = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    result, duration = run_duration_alignment_host(
        {"duration_logits": logits, "duration_features": features, "text_encoded": text},
        speed=torch.ones(1),
        valid_len=torch.tensor([3]),
        frame_max_length=8,
    )
    assert duration.tolist() == [[1, 2, 1, 0]]
    assert result["valid_frames"].item() == 4
    torch.testing.assert_close(result["encoded"][:, :, :4], features[:, [0, 1, 1, 2]].transpose(1, 2), rtol=0, atol=0)
    torch.testing.assert_close(result["asr"][:, :, :4], text[:, :, [0, 1, 1, 2]], rtol=0, atol=0)
    assert torch.count_nonzero(result["encoded"][:, :, 4:]) == 0
    assert torch.count_nonzero(result["asr"][:, :, 4:]) == 0


def test_kokoro_cpu_phase_export_and_portable_golden(tmp_path):
    # Only the real sine source is needed; SourceMerge is not used by phase.
    core = nn.Module()
    core.sine_source = StaticSineSource(nn.Identity(), frame_max_length=4, seed=1234)
    phase = PhaseCoreStatic(core).eval()
    f0 = torch.tensor([[122.2, 122.3, 165.7, 180.1, 151.9, 111.2, 0.0, 0.0]])
    up = core.sine_source.upsample_f0(f0)
    increments = core.sine_source.prepare_phase_increments(up)
    expected = core.sine_source.phase_increments_to_sine(increments)
    torch.testing.assert_close(phase(f0), expected, rtol=0, atol=0)
    path = tmp_path / "phase" / "hmquant_xh2_test_phase_with_act.onnx"
    artifact = _export_partition(
        role="phase_core",
        module=phase,
        feed={"f0": f0},
        expected={"sine": expected},
        output_path=path,
        opset=17,
        expected_lstm_nodes=0,
        npu_graph=False,
    )
    assert artifact.input_names == ("f0",)
    assert artifact.op_counts["CumSum"] == artifact.op_counts["Sin"] == 1
    actual = GoldenCpuRunner(path).run({"f0": f0.numpy()})["sine"]
    np.testing.assert_allclose(actual, expected.numpy(), rtol=0, atol=4e-5)
    link = path.parent / "step_0" / path.name
    assert link.is_symlink() and link.readlink() == Path("..") / path.name
    assert link.resolve() == path
    assert len(list(link.parent.glob("*.npy"))) == 2
    with pytest.raises(FileExistsError):
        GoldenCpuRunner(path).run({"f0": f0.numpy()})


def test_kokoro_lstm_quant_export_always_decomposes_and_golden_links(tmp_path, monkeypatch):
    torch.manual_seed(21)
    source = nn.LSTM(8, 8, batch_first=True, bidirectional=True).eval()
    module = PrefixReversedBidirectionalLSTMStatic(source, 4).eval()
    values = torch.randn(1, 4, 8)
    indices = make_reverse_idx(4, torch.tensor([3])).long()
    feed = {"values": values, "reverse_indices": indices}
    with torch.no_grad():
        expected = module(values, indices)
        unpadded = source(values[:, :3])[0]
    torch.testing.assert_close(expected[:, :3], unpadded, rtol=1e-5, atol=1e-7)
    artifact = _export_partition(
        role="lstm_test",
        module=module,
        feed=feed,
        expected={"output": expected},
        output_path=tmp_path / "float.onnx",
        opset=17,
        expected_lstm_nodes=2,
        npu_graph=True,
    )
    np.testing.assert_allclose(
        OrtRunner(artifact.path).run({k: v.numpy() for k, v in feed.items()})["output"],
        expected.numpy(),
        rtol=1e-5,
        atol=1e-7,
    )
    monkeypatch.setenv("XHQUANT_LSTM_EXPORT_MODE", "native")
    monkeypatch.setenv("XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE", "0")
    target = tmp_path / "lstm" / "hmquant_xh2_test_lstm.onnx"
    _convert_hmonnx(artifact, feed, target, target="XH2a", quant_type="w16a16_sefp", debug=False)
    # The tiny fixture is below the SDK's 100 MB external-data threshold.
    # Serialize the same graph externally to exercise real release symlinks
    # without inflating the LSTM to production size; no graph/weight changes.
    proto = onnx.load(target)
    assert proto.graph.initializer
    onnx.save_model(
        proto,
        target,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{target.stem}_external_data",
        size_threshold=0,
    )
    published = target.with_name("hmquant_xh2_test_lstm_with_act.onnx")
    target.rename(published)
    meta = _hmonnx_artifact_metadata(published, tmp_path)
    assert meta["hmonnx_op_counts"].get("LSTM", 0) == 0
    assert meta["hmonnx_external_data"]
    candidate = GoldenHmonnxRunner(published, "cpu").run({k: v.numpy() for k, v in feed.items()})["output"]
    np.testing.assert_allclose(candidate, expected.numpy(), rtol=0, atol=0.025)
    links = list((published.parent / "step_0").iterdir())
    assert any(p.is_symlink() and p.name == published.name for p in links)
    assert all(not p.readlink().is_absolute() and p.exists() for p in links if p.is_symlink())
    assert any(p.is_dir() for p in links)
    assert __import__("os").environ["XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE"] == "0"
    assert __import__("os").environ["XHQUANT_LSTM_EXPORT_MODE"] == "native"
