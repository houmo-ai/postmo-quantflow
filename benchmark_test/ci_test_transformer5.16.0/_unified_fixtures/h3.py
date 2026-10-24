# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native H3 public fixture; no Sol-H3 or LTX source libraries are required."""

import json
from dataclasses import replace


def prepare(case, root, monkeypatch):
    from _unified_preparation import prepare_context
    from test_h3_native_build import native_source

    from xhmodel_merak.xh_other_model.models.sol_h3 import native_runtime

    from .sol_h3 import media_fingerprint

    catalog, recipe, _, _, _, kwargs, _, _, _ = prepare_context(case, root)
    source, config, modes = native_source(root)
    # Leave adapter resolution to public LoRARequest, including its fingerprint.
    if case.variant == "lora":
        config["model"]["lora"] = {"path": []}
    else:
        config["model"].pop("lora")
    monkeypatch.setitem(
        catalog._recipes,
        case.model_id,
        replace(
            recipe,
            default_profile=config["profile"],
            profiles=(config["profile"],),
            variants=tuple(variant for variant in recipe.variants if variant.name in ("full", "lora")),
            config_replacements=(
                *recipe.config_replacements,
                *((f"export.{key}", value) for key, value in config.items()),
            ),
        ),
    )
    kwargs.update(quant_method=None, deployment_precision={"video": "w8a8_sefp"})
    if case.variant == "lora":
        kwargs["lora"] = dict(paths=[str(source / "adapter")])
    media = {}
    write_video = native_runtime.write_video

    def capture_generated_media(path, pixels, pcm):
        media[str(path)] = media_fingerprint(pixels, pcm)
        return write_video(path, pixels, pcm)

    monkeypatch.setattr(native_runtime, "write_video", capture_generated_media)
    return kwargs, dict(prompt="hello world", seed=7, modes=modes, generated_media=media)


def check_inference(moved, inputs, record):
    import av
    import numpy as np

    from xhmodel_merak.model_inference import inference_model

    record["generated_media"] = inputs["generated_media"]
    meta = json.loads((moved / "runtime_meta_info.json").read_text())
    expected = {"standard", "turbo8"} if record["case"].endswith("/lora") else {"standard"}
    assert set(meta["native_variants"]) == expected
    assert set(meta["mode_profiles"]) == set(inputs["modes"])
    assert meta["native_variants"]["standard"]["steps"] == 2
    assert not any("ltx" in key or "upscale" in key for key in meta["graphs"])
    index = json.loads((moved / "export_meta_info.json").read_text())
    assert set(index.get("lora", {})) == expected - {"standard"}
    if "turbo8" in expected:
        assert meta["native_variants"]["turbo8"]["steps"] == 8
        for branch in ("base", "ref"):
            view = meta["native_variants"]["turbo8"]["branches"][branch]
            assert sum(value["target_count"] for value in view["lora"]["bindings"]) == view["lora"]["target_count"]
            assert set(view["graphs"].values()) <= set(index["lora"]["turbo8"])
    # The worker removes source (including adapter directories) before this hook.
    assert not (moved.parent / "source").exists()
    # Standard covers all mode-specific encoders once; LoRA covers the ref
    # branch and reuses its package for a standard/adapter/standard roundtrip.
    modes, adapter = (["ref2va"], "turbo8") if "turbo8" in expected else (list(inputs["modes"]), None)
    for mode in modes:
        selected = {} if adapter is None else dict(lora_adapter=adapter)
        output = inference_model(
            artifact_dir=str(moved),
            device="cuda:0",
            inputs=dict(
                prompt=inputs["prompt"],
                seed=inputs["seed"],
                output_path=str(moved.parent / f"{mode}.mp4"),
                **inputs["modes"][mode],
                **selected,
            ),
        )
        assert output.model_id == "minimax-h3"
        assert (output.height, output.width, output.frames, output.fps) == (32, 32, 124, 24)
        assert output.stats["draft_updates"] == (2 if adapter is None else 8)
        assert output.stats["refiner_updates"] == 0
        assert output.stats["mode"] == mode
        json.dumps(output.to_dict(), allow_nan=False)
        with av.open(output.output_path) as media:
            assert len(media.streams.audio) == 1
            frames = np.stack([frame.to_ndarray(format="rgb24") for frame in media.decode(video=0)])
            assert frames.shape == (124, 32, 32, 3)
    roundtrip = []
    if "turbo8" in expected:
        from xhmodel_merak.xh_other_model.models.sol_h3.native_runtime import NativeH3Runtime

        runtime = NativeH3Runtime(moved)
        calls = []
        invoke = runtime.invoke

        def track(name, *values):
            calls.append(name)
            return invoke(name, *values)

        runtime.invoke = track
        try:
            for i, selected in enumerate((None, "turbo8", None)):
                generated = runtime.generate(
                    inputs["prompt"], seed=inputs["seed"],
                    output_path=str(moved.parent / f"adapter_roundtrip_{i}.mp4"), lora_adapter=selected,
                )
                assert generated["stats"]["mode"] == "t2va"
                assert generated["stats"]["draft_updates"] == (2 if selected is None else 8)
                with av.open(str(generated["output_path"])) as media:
                    assert len(media.streams.audio) == 1
                    frames = np.stack([frame.to_ndarray(format="rgb24") for frame in media.decode(video=0)])
                    assert frames.shape == (124, 32, 32, 3)
                roundtrip.append(inputs["generated_media"][str(generated["output_path"])])
            assert roundtrip[0] == roundtrip[2]
            assert not any(name.startswith(("h3_visual", "h3_reference_")) for name in calls)
        finally:
            runtime.close()
    record["native_h3"] = dict(
        variants=sorted(expected), modes=modes, adapter=adapter, frames=124, source_removed=True,
        adapter_roundtrip_verified=bool(roundtrip),
        adapter_roundtrip_comparison="pre_codec_pixels_and_pcm" if roundtrip else None,
    )
    record.update(passed=True, stage="complete", inference=output.to_dict())
