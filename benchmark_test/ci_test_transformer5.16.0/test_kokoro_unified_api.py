"""Kokoro delivery and dispatch contracts; no full-model accuracy claim.

Graph files are small ONNX fixtures. Numerical CPU phase and real xhquant LSTM
coverage lives in test_kokoro_release.py; only expensive sessions/G2P are stubbed
here while packaging, validation, token lookup and runtime routing stay real.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _kokoro_fixture import PREFIX, ROLES, write_release

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import file_sha256
from xhmodel_merak.model_build import build_model, describe_model
from xhmodel_merak.model_build.manifest import file_manifest
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.model_inference import inference_model
from xhmodel_merak.model_integrations.kokoro import build, inference
from xhmodel_merak.xh_other_model.models.kokoro import bucketed_runtime, text


MODEL_ID = "kokoro-82m-v1.1-zh"


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "artifact"
    meta = write_release(root)
    plan = SimpleNamespace(model_id=MODEL_ID, components=ROLES)
    return root, meta, plan


def _manifest(root):
    return {"artifact": {"files": file_manifest(root)}}


def _validate_both(root, plan):
    assert build.validate_artifact(root, plan)["loadable"]
    inference.adapter.validate_artifact(
        recipe=get_recipe(MODEL_ID), root=root, manifest=_manifest(root), meta_file=root / "export_meta_info.json"
    )


def test_release_is_consumed_without_reconstructing_frontend_or_source(release):
    root, meta, plan = release
    before = file_manifest(root)
    build.package_runtime(root, plan)  # plan deliberately has no model_dir
    build.sanitize_artifact(root, plan)
    _validate_both(root, plan)
    assert file_manifest(root) == before
    assert not (root / "runtime/frontend.json").exists()
    assert build.validate_artifact(root, plan)["release_prefix"] == PREFIX
    assert meta["phase_boundary"]["input"] == "f0"


@pytest.mark.parametrize(
    "field,value",
    [
        ("graph_mode", "bucketed_precision_split"),
        ("schema_version", 1),
        ("export_complete", False),
        ("hmonnx_converted", False),
        ("sample_rate", 16000),
        ("quant_type", "w8a8"),
        ("target_device", "XH3"),
        ("seed", None),
        ("seed", True),
        ("seed", "1234"),
        ("seed", 1 << 65),
        ("phase_boundary", {"execution": "host_onnxruntime_fp32", "input": "phase_increments", "output": "sine"}),
        ("component_order", list(reversed(ROLES))),
        ("components", {}),
    ],
)
def test_build_and_load_reject_the_same_broken_contract(release, field, value):
    root, meta, plan = release
    meta[field] = value
    (root / "export_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ModelError) as error:
        build.validate_artifact(root, plan)
    assert error.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    with pytest.raises(ValueError):
        inference.adapter.validate_artifact(
            recipe=get_recipe(MODEL_ID), root=root, manifest=_manifest(root), meta_file=root / "export_meta_info.json"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_route",
        "duplicate_route",
        "wrong_route",
        "missing_bucket",
        "native_variant",
        "native_lstm",
        "old_phase_input",
        "wrong_phase_dtype",
    ],
)
def test_rejects_route_and_precision_boundary_drift(release, mutation):
    root, meta, plan = release
    routes = meta["bucket_presets"]["routes"]
    bucket = meta["components"]["text_duration"]["buckets"]["t0032"]
    phase = meta["components"]["phase_core"]["buckets"]["f0160"]
    if mutation == "missing_route":
        routes.pop()
    elif mutation == "duplicate_route":
        routes[1] = copy.deepcopy(routes[0])
    elif mutation == "wrong_route":
        routes[0]["frame_max_length"] = 320
    elif mutation == "missing_bucket":
        meta["components"]["frame_acoustic"]["buckets"].pop("f1280")
    elif mutation == "native_variant":
        bucket["hmonnx_variants"] = {"native": {"hmonnx_file": bucket.pop("hmonnx_file")}}
    elif mutation == "native_lstm":
        bucket["hmonnx_op_counts"]["LSTM"] = 10
    elif mutation == "old_phase_input":
        phase["inputs"][0]["name"] = "phase_increments"
    else:
        phase["inputs"][0]["dtype"] = "FLOAT16"
    (root / "export_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ModelError):
        build.validate_artifact(root, plan)
    with pytest.raises(ValueError):
        inference.adapter.validate_artifact(
            recipe=get_recipe(MODEL_ID), root=root, manifest=_manifest(root), meta_file=root / "export_meta_info.json"
        )


@pytest.mark.parametrize("asset", ["config", "voice", "graph", "external_data"])
@pytest.mark.parametrize("corruption", ["missing", "escape", "symlink", "uninventoried"])
def test_all_runtime_files_are_confined_and_inventoried(release, asset, corruption):
    root, meta, plan = release
    if asset == "config":
        owner, key = meta["runtime_assets"], "config_file"
    elif asset == "voice":
        owner, key = meta["runtime_assets"]["voice_pack"], "file"
    else:
        owner = meta["components"]["text_duration"]["buckets"]["t0032"]
        key = "hmonnx_file"
        if asset == "external_data":
            owner, key = owner["hmonnx_external_data"][0], "file"
    path = root / owner[key]
    manifest = _manifest(root)
    if corruption == "missing":
        path.unlink()
    elif corruption == "escape":
        owner[key] = "../outside"
    elif corruption == "symlink":
        saved = path.with_name(path.name + ".saved")
        path.rename(saved)
        path.symlink_to(saved.name)
    else:
        manifest["artifact"]["files"] = [
            entry for entry in manifest["artifact"]["files"] if entry["path"] != owner[key]
        ]
    (root / "export_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")
    if corruption != "uninventoried":
        with pytest.raises(ModelError):
            build.validate_artifact(root, plan)
    with pytest.raises(ModelError):
        inference.adapter.validate_artifact(
            recipe=get_recipe(MODEL_ID), root=root, manifest=manifest, meta_file=root / "export_meta_info.json"
        )


@pytest.mark.parametrize(
    "voice",
    [
        np.empty((0, 1, 256), np.float32),
        np.full((1, 256), np.nan, np.float32),
        np.asarray(["bad"]),
        np.asarray([object()]),
        np.zeros((1, 256), np.float64),
        np.zeros((2, 256), np.float32),
        np.asarray(1, dtype=np.float32),
    ],
)
def test_malformed_voice_cannot_be_packaged_or_loaded(release, voice):
    root, meta, plan = release
    path = root / meta["runtime_assets"]["voice_pack"]["file"]
    np.save(path, voice, allow_pickle=True)
    with pytest.raises(ModelError):
        build.validate_artifact(root, plan)
    with pytest.raises(ValueError):
        inference.adapter.validate_artifact(
            recipe=get_recipe(MODEL_ID), root=root, manifest=_manifest(root), meta_file=root / "export_meta_info.json"
        )


def _run_kwargs(root):
    return dict(
        recipe=get_recipe(MODEL_ID),
        root=root,
        manifest=_manifest(root),
        meta_file=root / "export_meta_info.json",
        inputs={"text": "你好"},
        device="cpu",
        components=ROLES,
        selected_golden=(),
        golden_artifacts={},
    )


@pytest.mark.parametrize(
    "phonemes,message", [("a" * 255, "split the text"), ("a?", "vocabulary"), ("...", "pronounceable")]
)
def test_public_frontend_uses_model_validation_without_splitting(release, monkeypatch, phonemes, message):
    root, _, _ = release
    monkeypatch.setattr(text, "_chinese_g2p", lambda: lambda _: (phonemes, None))
    monkeypatch.setattr(
        bucketed_runtime,
        "KokoroBucketedRuntime",
        lambda *args, **kwargs: pytest.fail("invalid text must fail before session construction"),
    )
    with pytest.raises(ValueError, match=message):
        inference.adapter.run(**_run_kwargs(root))


@pytest.mark.parametrize("fails", [False, True])
def test_public_dispatch_uses_model_tokens_voice_and_current_runtime_signature(release, monkeypatch, fails):
    root, _, _ = release
    monkeypatch.setattr(text, "_chinese_g2p", lambda: lambda _: ("a", None))
    cleared = []

    class SessionBoundary:
        def __init__(self, work_dir, meta, *, backend, device):
            assert work_dir == root and backend == "hmonnx" and device == "cpu"

        def synthesize(self, tokens, style, *, speed):
            np.testing.assert_array_equal(tokens, [0, 17, 0])
            np.testing.assert_array_equal(style, np.ones((1, 256), np.float32))
            assert speed == 1.0
            if fails:
                raise RuntimeError("session failed")
            return np.ones(2400, np.float32), {"route": "t0032_f0160"}

        def clear_runner_cache(self):
            cleared.append(True)

    monkeypatch.setattr(bucketed_runtime, "KokoroBucketedRuntime", SessionBoundary)
    if fails:
        with pytest.raises(RuntimeError, match="session failed"):
            inference.adapter.run(**_run_kwargs(root))
    else:
        result = inference.adapter.run(**_run_kwargs(root))
        assert result.sample_rate == 24000 and result.num_samples == 2400
        assert result.duration_s == pytest.approx(0.1)
        assert result.segments[0]["route"] == "t0032_f0160"
        assert result.stats["segment_count"] == 1
    assert cleared == [True]


def test_catalog_exposes_only_supported_kokoro_parameters():
    assert set(describe_model(MODEL_ID)) == {"model_dir", "output_dir", "device"}
    full = describe_model(MODEL_ID, full=True)
    assert "misaki==0.9.4" in full["worker_requirements"]
    assert full["recipe_version"] == "3"
    assert full["golden_components"] == []


def test_runtime_retries_paired_route_and_passes_f0_to_cpu_phase(release):
    root, meta, _ = release
    calls = []
    f0 = np.linspace(120, 180, 640, dtype=np.float32).reshape(1, -1)
    sine = np.full((1, 600 * 320, 9), 0.025, dtype=np.float32)

    class GraphBoundary:
        def __init__(self, role, key, path):
            self.role, self.key = role, key
            field = "onnx_file" if role == "phase_core" else "hmonnx_file"
            assert path == root / meta["components"][role]["buckets"][key][field]

        def run(self, feed):
            calls.append((self.role, self.key))
            if self.role == "text_duration":
                capacity = feed["input_ids"].shape[1]
                # 31 valid tokens, eight frames each: F=248 exceeds T32/F160.
                return {
                    "duration_logits": np.zeros((1, capacity, 16), np.float32),
                    "duration_features": np.ones((1, capacity, 3), np.float32),
                    "text_encoded": np.ones((1, 2, capacity), np.float32),
                }
            if self.role == "frame_acoustic":
                assert feed["valid_frames"].item() == 248
                assert feed["encoded"].shape == (1, 3, 320)
                assert np.count_nonzero(feed["encoded"][:, :, 248:]) == 0
                return {"decoder_feature": np.ones((1, 2, 640), np.float32), "f0": f0}
            if self.role == "phase_core":
                assert set(feed) == {"f0"} and feed["f0"].dtype == np.float32
                np.testing.assert_array_equal(feed["f0"], f0)
                return {"sine": sine}
            np.testing.assert_array_equal(feed["f0"], f0)
            np.testing.assert_array_equal(feed["sine"], sine)
            return {"waveform": np.ones((1, 600 * 320), np.float32)}

    runtime = bucketed_runtime.KokoroBucketedRuntime(
        root, meta, backend="hmonnx", device="cpu", runner_factory=GraphBoundary
    )
    try:
        wave, stats = runtime.synthesize(np.ones(31, np.int32), np.ones((1, 256), np.float32))
        assert stats["route"] == "t0064_f0320"
        assert stats["phase_boundary"] == "f0_to_sine_host_fp32"
        assert stats["npu_graph_count"] == 3 and wave.shape == (248 * 600,)
        assert calls == [
            ("text_duration", "t0032"),
            ("text_duration", "t0064"),
            ("frame_acoustic", "f0320"),
            ("phase_core", "f0320"),
            ("generator_istft", "f0320"),
        ]
    finally:
        runtime.clear_runner_cache()


@pytest.fixture
def packaging_workflow(monkeypatch, tmp_path):
    """Stub only the expensive exporter; retain the public build transaction."""
    from xhmodel_merak.model_build import workflow

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    controls = {"failure": None, "graph_hashes": {}}

    class ExportBoundary:
        def __init__(self, model_dir, config_path):
            self.model_dir, self.config_path = model_dir, config_path

        def quant(self, **kwargs):
            return SimpleNamespace(raw_model_dir=self.model_dir, quanted_model_dir=None, skipped=True)

        def export(self, *, output_dir, **kwargs):
            root = Path(output_dir) / PREFIX
            meta = write_release(root)
            config = root / "workflow.yaml"
            config.write_bytes(Path(self.config_path).read_bytes())
            controls["graph_hashes"] = {
                path.relative_to(root).as_posix(): file_sha256(path)
                for path in root.rglob("*")
                if path.is_file() and (path.suffix == ".onnx" or path.name.endswith("external_data"))
            }
            if controls["failure"] == "conversion":
                raise RuntimeError("conversion failed after writing files")
            if controls["failure"] == "validation":
                meta["export_complete"] = False
                (root / "export_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")
            return SimpleNamespace(work_dir=str(root), config_file=str(config), meta=meta)

    monkeypatch.setattr(workflow, "create_workflow", ExportBoundary)
    for module in ("model_build.executor", "model_inference.api"):
        monkeypatch.setattr(f"xhmodel_merak.{module}.validate_worker_requirements", lambda *args, **kwargs: None)
        monkeypatch.setattr(f"xhmodel_merak.{module}.validate_worker_capabilities", lambda *args, **kwargs: None)
    return source, controls


def test_public_build_preserves_release_files_and_loads_after_move(packaging_workflow, monkeypatch, tmp_path):
    source, controls = packaging_workflow
    result = build_model(model_id=MODEL_ID, model_dir=str(source), output_dir=str(tmp_path / "published"), device="cpu")
    root = Path(result.output_dir)
    assert result.loadable
    assert not (root / PREFIX).exists()
    for relative, digest in controls["graph_hashes"].items():
        assert file_sha256(root / relative) == digest
    assert len(list(root.glob("*_t*"))) + len(list(root.glob("*_f*"))) == 16
    assert not list(root.rglob("step_*"))  # build does not generate golden
    moved = tmp_path / "moved"
    root.rename(moved)
    source.rename(tmp_path / "removed-source")
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return "validated and dispatched"

    monkeypatch.setattr(inference.adapter, "run", run)
    assert inference_model(artifact_dir=str(moved), inputs={"text": "你好"}, device="cpu") == "validated and dispatched"
    assert calls[0]["root"] == moved


@pytest.mark.parametrize("failure", ["conversion", "validation"])
def test_failed_kokoro_build_never_publishes_partial_release(packaging_workflow, tmp_path, failure):
    source, controls = packaging_workflow
    controls["failure"] = failure
    output = tmp_path / "published"
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ModelError):
        build_model(model_id=MODEL_ID, model_dir=str(source), output_dir=str(output), device="cpu")
    assert not output.exists()
    assert not [path for path in tmp_path.glob(".*xhbuild*") if path.is_dir()]
    assert not list(tmp_path.glob(".*recipe*"))
    # The stable lock inode is intentionally retained to prevent unlink races;
    # it must be unlocked so a corrected build can retry the same destination.
    from xhmodel_merak.model_build.workspace import acquire_output_lock, release_output_lock

    lock = acquire_output_lock(output)
    release_output_lock(lock)
    assert sentinel.read_text(encoding="utf-8") == "keep"
