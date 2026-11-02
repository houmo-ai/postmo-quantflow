# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from _unified_preparation import prepare_context


def media_fingerprint(pixels, pcm):
    """Hash model pixels and FP32 PCM before lossy media encoding."""
    import hashlib

    return dict(
        pixels=hashlib.sha256(pixels.tobytes()).hexdigest(),
        pcm=hashlib.sha256(pcm.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest(),
    )


def prepare(case, root, monkeypatch):
    catalog, recipe, source, calibration, image, kwargs, inputs, replacements, family = prepare_context(case, root)
    from _sol_h3_small_fixture import prepare_conditioned_source

    from xhmodel_merak.xh_other_model.models.sol_h3 import runtime as runtime_module
    from xhmodel_merak.xh_other_model.models.sol_h3.build import BundleExporter

    source, inputs, _, inventory, profile = prepare_conditioned_source(root, monkeypatch, "all")
    kwargs.update(model_dir=str(source), profile=profile)
    kwargs["deployment_precision"] = {"video": "w8a8_sefp"}
    kwargs["quant_method"] = None
    inputs.update(inventory=inventory, profile=profile)
    media = inputs["generated_media"] = {}
    write_video = runtime_module.write_video

    def capture_generated_media(path, pixels, pcm):
        # Codec threading can change a lossy MP4 even when model outputs match.
        media[str(path)] = media_fingerprint(pixels, pcm)
        return write_video(path, pixels, pcm)

    monkeypatch.setattr(runtime_module, "write_video", capture_generated_media)

    def forbid_export_replay(exporter, name, values):
        raise AssertionError(f"Sol-H3 export must not run an HMONNX component: {name}")

    monkeypatch.setattr(BundleExporter, "infer", forbid_export_replay)
    return kwargs, inputs


def validate_build(case, result, record):
    from pathlib import Path

    from _sol_h3_small_fixture import assert_all_export_layout

    artifact = Path(result.output_dir)
    assert_all_export_layout(
        artifact.parent / "source",
        artifact,
        {"output_path": str(artifact.parent / "result.mp4")},
        "all-dense-32p-64p-5s",
    )
    assert result.deployment_precision == {"video": "w8a8_sefp"}
    import json

    import numpy as np

    meta = json.loads((artifact / "runtime_meta_info.json").read_text())
    assert meta["schema_version"] == 9
    assert meta["ltx_audio_policy"] == {
        "input_name": "audio_mode", "dtype": "int32", "shape": [1],
        "modes": {"joint": 0, "frozen": 1}, "default": "frozen",
    }
    assert meta["attention_contract"] == {"mask_input_index": 4, "min_flash_kv_length": 512}
    assert meta["text_feature_divisor"] == meta["text_refiner_input_divisor"] == 16
    assert meta["text_rms_policy"] == "plain_wide_reduction" and meta["ltx_rope_mode"] == "inputs"
    assert all(item["path"].endswith(".onnx") for item in meta["graphs"].values())
    with np.load(artifact / meta["resources"]["ltx_rope"], allow_pickle=False) as rope:
        assert set(rope.files) == {f"rope_{i}" for i in range(8)}
        assert all(rope[key].dtype == np.float16 and np.isfinite(rope[key]).all() for key in rope.files)


def check_inference(moved, inputs, record):
    import json

    import av
    import numpy as np

    from xhmodel_merak.model_inference import inference_model
    from xhmodel_merak.xh_other_model.models.sol_h3.bundle import mode_metadata
    from xhmodel_merak.xh_other_model.models.sol_h3.runtime import SolH3Runtime

    record["generated_media"] = inputs["generated_media"]
    assert not (moved.parent / "source").exists()
    request = {key: inputs[key] for key in ("prompt", "seed", "output_path")}
    output = inference_model(artifact_dir=str(moved), device="cuda:0", inputs=request)
    assert output.model_id == "sol-h3"
    assert (output.height, output.width, output.frames, output.fps) == (64, 64, 121, 24)
    assert output.stats["draft_updates"] == 4 and output.stats["refiner_updates"] == 3
    assert output.stats["ltx_audio_mode"] == "frozen" and output.stats["ltx_audio_frozen"] is True
    with av.open(output.output_path) as video:
        assert len(video.streams.audio) == 1
        first = np.stack([frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)])
        assert len(first) == 121
    first_media = inputs["generated_media"][output.output_path]
    meta = json.loads((moved / "runtime_meta_info.json").read_text())
    assert meta["schema_version"] == 9 and meta["mode"] == "all" and meta["profile"] == inputs["profile"]
    assert meta["attention_contract"] == {"mask_input_index": 4, "min_flash_kv_length": 512}
    assert meta["ltx_audio_policy"]["modes"] == {"joint": 0, "frozen": 1}
    assert set(meta["branches"]) == {"base", "ref"}
    assert "h3_reference_encode_ref_t17" in meta["reference_vae_tiles"].values()
    assert set(meta["reference_vae_masks"]) == set(meta["reference_vae_tiles"].values())
    image_gears = {name for shape, name in meta["reference_vae_tiles"].items() if shape.startswith("1x")}
    assert image_gears and all(name.startswith("h3_reference_encode_ref_t1_h") for name in image_gears)
    assert "reference_vae_mask" not in meta
    for shape, name in meta["reference_vae_tiles"].items():
        canvas = meta["reference_vae_masks"][name]["shape"]
        tile = list(map(int, shape.split("x")))
        assert canvas[0] == tile[0] and all(a <= b for a, b in zip(tile, canvas, strict=True))
    assert meta["reference_vae_layout"] == "split_v1" and "video_decoder_mask" in meta
    base = meta["reference_vae_base"]
    assert base["shape"] == [1, *meta["draft_size"]]
    assert meta["graphs"][base["graph"]]["input_names"] == ["pixels"]
    assert base["graph"] not in meta["reference_vae_masks"]
    runtime = SolH3Runtime(moved, device="cuda:0")
    try:
        for mode in ("i2va", "l2va", "fl2va", "ref2va", "t2va"):
            result = runtime.generate(
                **{
                    **request,
                    **inputs["inventory"].get(mode, {}),
                    "output_path": str(moved.parent / f"{mode}.mp4"),
                }
            )
            assert result["stats"]["mode"] == mode
            with av.open(result["output_path"]) as video:
                assert len(video.streams.audio) == 1
                frames = np.stack([frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)])
                assert len(frames) == 121
                if mode == "t2va":
                    assert inputs["generated_media"][str(result["output_path"])] == first_media
        assert runtime.bundle_meta["mode"] == "all"
        joint = inference_model(
            artifact_dir=str(moved), device="cuda:0",
            inputs={**request, **inputs["inventory"]["ref2va"], "ltx_audio_mode": "joint",
                    "output_path": str(moved.parent / "ref2va_joint.mp4")},
        )
        assert joint.stats["ltx_audio_mode"] == "joint" and joint.stats["ltx_audio_frozen"] is False
        with av.open(joint.output_path) as video:
            assert len(video.streams.audio) == 1
            assert sum(1 for _ in video.decode(video=0)) == 121
        reference = mode_metadata(meta, "ref2va")
        assert reference["h3_adapter"]["format"] == "peft"
        assert "h3_reference_audio_encode" in reference["graphs"]
        assert reference["capacity_plan"]["reference_audio_tokens"] == 320
        assert (
            reference["graphs"]["h3_denoiser"]["path"] != mode_metadata(meta, "t2va")["graphs"]["h3_denoiser"]["path"]
        )
    finally:
        runtime.close()
    record.update(passed=True, stage="complete", inference=output.to_dict())
    record["sol_h3"] = {
        "modes": ["t2va", "i2va", "l2va", "fl2va", "ref2va"],
        "source_removed": True,
        "return_t2va_bitexact": True,
        "return_t2va_comparison": "pre_codec_pixels_and_pcm",
        "ltx_audio_modes": ["joint", "frozen"],
    }
