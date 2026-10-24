# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Static native H3 mode and adapter-view contracts without model export."""

import json
from copy import deepcopy

import numpy as np
import pytest


def encoder_case(mode, reference_kind="image"):
    from test_h3_native_api import encoder_metadata

    meta = encoder_metadata()
    meta["mode_profiles"] = {mode: meta["mode_profiles"][mode]}
    plan = meta["mode_profiles"][mode]["capacity_plan"]
    meta["capacity_plan"] = deepcopy(plan)
    if mode == "t2va":
        meta["capacity_plan"].update(qwen_visual_max_patches=0, qwen_visual_patch_gears=[])
        plan.update(qwen_visual_max_patches=0, qwen_visual_patch_gears=[])
        meta["visual_graphs"] = []
        meta["graphs"].pop("h3_visual_256")
        meta["processor_files"] = []
        for key in ("qwen_config", "qwen_vision_config", "reference_vae_config"):
            meta["resources"].pop(key)
    if mode not in ("i2va", "l2va", "fl2va"):
        meta.pop("reference_vae_base")
        meta["graphs"].pop("base")
    if mode != "ref2va":
        meta["reference_vae_tiles"] = {}
        meta["reference_vae_masks"] = {}
        meta["graphs"].pop("ref")
        meta["resources"].pop("reference_audio_config")
        meta.pop("reference_audio_frames")
    elif reference_kind == "video":
        meta["reference_vae_tiles"] = {"17x64x64": "ref"}
        meta["reference_vae_masks"]["ref"]["shape"][0] = 17
        plan["reference_audio_tokens"] = 80
        meta["capacity_plan"]["reference_audio_tokens"] = 80
        meta["graphs"]["h3_reference_audio_encode"] = {}
    config = {mode: {"references": {reference_kind: {"max_count": 1}} if mode == "ref2va" else {}}}
    return meta, config


@pytest.mark.parametrize(
    "mode,kind",
    [
        ("t2va", "image"),
        ("i2va", "image"),
        ("ref2va", "image"),
        ("ref2va", "video"),
    ],
)
def test_encoder_contract_requires_only_selected_mode(mode, kind):
    from xhmodel_merak.xh_other_model.models.sol_h3.native_artifacts import _validate_encoders

    meta, config = encoder_case(mode, kind)
    _validate_encoders(meta, config)
    if mode == "t2va":
        meta["graphs"]["h3_reference_encode_unused"] = {}
    elif mode != "ref2va":
        meta["reference_vae_tiles"] = {"1x64x64": "unused"}
    else:
        meta["reference_vae_masks"]["ref"]["shape"][0] = 17 if kind == "image" else 1
    with pytest.raises(ValueError, match="Native H3"):
        _validate_encoders(meta, config)


def test_reference_image_keeps_audio_host_resources_without_encoder():
    from xhmodel_merak.xh_other_model.models.sol_h3.native_artifacts import _validate_encoders

    meta, config = encoder_case("ref2va")
    assert "h3_reference_audio_encode" not in meta["graphs"]
    _validate_encoders(meta, config)
    meta["resources"].pop("reference_audio_config")
    with pytest.raises(ValueError, match="resources"):
        _validate_encoders(meta, config)


def test_native_schema_one_is_rejected_before_graph_loading(tmp_path):
    from xhmodel_merak.xh_other_model.models.sol_h3.native_artifacts import read_metadata

    (tmp_path / "runtime_meta_info.json").write_text(
        json.dumps(dict(kind="native_h3", schema_version=1, complete=True))
    )
    with pytest.raises(ValueError, match="schema-2"):
        read_metadata(tmp_path)


def test_unexported_mode_is_rejected_before_adapter_selection():
    from test_h3_native_api import metadata

    from xhmodel_merak.xh_other_model.models.sol_h3.native_artifacts import mode_metadata

    meta = metadata()
    meta["mode_profiles"] = {"i2va": meta["mode_profiles"]["i2va"]}
    with pytest.raises(ValueError, match="not exported.*i2va"):
        mode_metadata(meta, "t2va")


def test_reference_video_soundtrack_does_not_enable_standalone_audio():
    import torch
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3AudioReference, MiniMaxH3VideoReference

    from xhmodel_merak.xh_other_model.models.sol_h3.request import _validate_durations

    waveform = torch.zeros(2, 64000)
    video = MiniMaxH3VideoReference(
        frames=np.zeros((48, 32, 32, 3), np.uint8), fps=24, audio=waveform, sample_rate=32000
    )
    limits = dict(video=dict(max_count=1, max_duration_seconds=2), audio=dict(max_count=0, max_duration_seconds=2))
    _validate_durations([video], limits)
    with pytest.raises(ValueError, match="audio count exceeds configured maximum 0"):
        _validate_durations([video, MiniMaxH3AudioReference(waveform, 32000)], limits)


@pytest.mark.parametrize("package_mode,selected", [("i2va", "i2va"), ("ref2va", "ref2va"), ("all", "ref2va")])
def test_adapter_switch_without_mode_preserves_current_view(package_mode, selected):
    import torch
    from test_h3_native_api import metadata

    from xhmodel_merak.xh_other_model.models.sol_h3.native_artifacts import mode_metadata
    from xhmodel_merak.xh_other_model.models.sol_h3.native_runtime import NativeH3Runtime

    # This unit exercises state transitions independently of artifact I/O;
    # the public full/LoRA fixtures cover construction from published metadata.
    runtime = NativeH3Runtime.__new__(NativeH3Runtime)
    runtime.device = torch.device("cpu")
    runtime.bundle_meta = metadata()
    runtime.sessions = {}
    if package_mode != "all":
        runtime.bundle_meta["mode_profiles"] = {selected: runtime.bundle_meta["mode_profiles"][selected]}
    runtime.meta = mode_metadata(runtime.bundle_meta, "t2va" if package_mode == "all" else selected)
    if package_mode == "all":
        runtime.select_adapter(mode=selected)
    base = deepcopy(runtime.meta)
    runtime.select_adapter("lora8")
    assert runtime.meta["mode"] == selected and runtime.meta["active_adapter"] == "lora8"
    runtime.select_adapter()
    assert runtime.meta == base
    previous = runtime.meta
    runtime.sessions["keep"] = object()
    with pytest.raises(ValueError, match="Unknown"):
        runtime.select_adapter("missing")
    assert runtime.meta is previous and "keep" in runtime.sessions
    runtime.close()
