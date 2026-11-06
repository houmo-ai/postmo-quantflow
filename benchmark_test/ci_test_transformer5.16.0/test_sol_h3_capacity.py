# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Capacity bounds checked against native processors and canvas rounding."""

from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLVideoProcessor

from xhmodel_merak.xh_other_model.models.sol_h3.capacity import (
    configure_processor,
    image_geometries,
    mode_configs,
    plan_capacities,
    plan_unified_capacities,
    video_geometries,
)


@pytest.fixture
def components():
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    processor = SimpleNamespace(
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, merge_size=2, temporal_patch_size=2, size={"shortest_edge": 1024, "longest_edge": 8192}
        ),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=16, merge_size=2, temporal_patch_size=2, size={"shortest_edge": 1024, "longest_edge": 32768}
        ),
    )
    return dict(tokenizer=tokenizer, processor=processor, max_position_embeddings=262144)


def test_prompt_budget_is_independent_of_chunk_and_target_tokens(components):
    config = dict(
        mode="t2va",
        prompt=dict(max_tokens=513),
        draft_size=[384, 672],
        draft_frames=124,
        qwen3_vl=dict(prefill_chunk_length=256),
    )
    plan = plan_capacities(config, **components)
    assert plan["text_capacity"] == 768 and plan["kv_capacity"] == 768
    assert plan["target_video_tokens"] == 9324 and plan["target_audio_tokens"] == 414
    assert plan["text_tokens"] == 513
    assert plan["video_capacity"] == 9472 and plan["audio_capacity"] == 512
    assert plan["dit_capacity"] == 768 + 9472 + 512
    assert plan["reference_video_tokens"] == plan["reference_audio_tokens"] == 0
    with pytest.raises(ValueError, match="exceeds native"):
        plan_capacities(config, **{**components, "max_position_embeddings": 512})


@pytest.mark.parametrize("mode,count", [("i2va", 1), ("l2va", 1), ("fl2va", 2)])
def test_keyframe_budget_matches_actual_native_image_grid_and_labels(components, mode, count):
    config = dict(mode=mode, prompt=dict(max_tokens=20), draft_size=[32, 64], draft_frames=124)
    plan = plan_capacities(config, **components)
    processor, tokenizer = components["processor"], components["tokenizer"]
    pixels = np.zeros((32, 64, 3), np.uint8)
    grid = processor.image_processor(images=[pixels], return_tensors="pt")["image_grid_thw"][0]
    tokens = int(grid.prod()) // 4
    overhead = sum(
        len(tokenizer(f"<Picture {i}>: ", add_special_tokens=False)["input_ids"]) + 2 for i in range(1, count + 1)
    )
    assert plan["text_tokens"] == 20 + count * tokens + overhead
    assert plan["text_capacity"] == ((plan["text_tokens"] + 255) // 256) * 256
    assert plan["reference_video_tokens"] == count * 2


def test_native_video_odd_sampling_and_audio_labels_fit_derived_bounds(components):
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep as Step

    config = dict(
        mode="ref2va",
        prompt=dict(max_tokens=20),
        draft_size=[32, 64],
        draft_frames=124,
        references=dict(
            image=dict(max_count=2, vae_short_edge=32),
            video=dict(max_count=1, max_duration_seconds=2.1),
            audio=dict(max_count=1, max_duration_seconds=2.75),
        ),
    )
    plan = plan_capacities(config, **components)
    processor = components["processor"]
    sampled, times = Step._sample_video_condition_frames(np.zeros((51, 32, 64, 3), np.uint8), 24, 2, 2)
    assert len(sampled) == 5 and len(times) == 3
    grid = processor.video_processor(videos=[np.stack(sampled)], do_sample_frames=False, return_tensors="pt")[
        "video_grid_thw"
    ][0]
    assert int(grid[0]) == 3
    # The native geometry helper reports floor(T/2); real patchify repeats the odd tail.
    floor_patches = processor.video_processor.get_num_of_video_patches(5, 32, 64)
    assert int(grid.prod()) == floor_patches // 2 * 3
    assert plan["video_qwen_tokens"] >= int(grid.prod()) // 4
    assert plan["reference_audio_tokens"] == 440  # audio reference + optional video soundtrack
    assert plan["dit_capacity"] == sum(plan[key] for key in ("text_capacity", "video_capacity", "audio_capacity"))


@pytest.mark.parametrize("draft_frames", [124, 362])
def test_compact_reference_budget_is_independent_of_target_duration(components, draft_frames):
    from xhmodel_merak.xh_other_model.models.sol_h3.reference_geometry import image_size, video_size

    config = dict(
        mode="ref2va", prompt=dict(max_tokens=1024), draft_size=[384, 672], draft_frames=draft_frames,
        references=dict(
            independent_duration=True,
            image=dict(
                max_count=3, target_area=384 * 672,
                vae_gears=[[384, 672], [448, 672], [672, 448], [512, 512]],
            ),
            video=dict(max_count=1, max_duration_seconds=5, vae_size=[384, 672]),
            audio=dict(max_count=1, max_duration_seconds=5),
        ),
    )
    plan = plan_capacities(config, **components)
    assert plan["reference_video_tokens"] == 32 * 12 * 21 + 3 * 14 * 21
    assert plan["reference_audio_tokens"] == 2 * 2 * 5 * 40
    assert plan["qwen_visual_max_patches"] <= 1280
    gears = config["references"]["image"]["vae_gears"]
    assert image_size(1080, 1920, gears, 384 * 672) == (384, 672)
    assert image_size(1920, 1080, gears, 384 * 672) == (672, 448)
    assert video_size(1080, 1920, config["references"]["video"]["vae_size"]) == (384, 672)


@pytest.mark.parametrize("suffix", ["", "_15s"])
def test_portrait_and_landscape_workflows_plan_equal_capacities(components, suffix):
    from pathlib import Path

    import yaml

    workflows = Path(__file__).resolve().parents[2] / "configs_merak/workflows/xh2a/other_models/sol_h3"
    plans = []
    for orientation in ("", "_portrait"):
        path = workflows / f"sol_h3_all_384p_768p{orientation}{suffix}_w4.yaml"
        config = yaml.safe_load(path.read_text())["export"]
        plans.append(plan_unified_capacities(config, **components))
    assert plans[0] == plans[1]


def test_native_rounding_geometries_and_processor_override_are_shared(components):
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size

    geometries = set(video_geometries([384, 672]))
    for ratio in np.linspace(0.25, 4, 10001):
        assert resolve_canvas_size(float(ratio), 1, 32, 384, 384 * 672) in geometries
    assert (32, 32) not in geometries
    assert (256, 256) not in geometries
    assert (384, 384) in geometries
    assert len(geometries) < 80
    assert (512, 2048) in image_geometries(512)
    changed = configure_processor(components["processor"], dict(image_max_pixels=4096, video_max_pixels=8192))
    assert changed.image_processor.size["longest_edge"] == 4096
    assert changed.video_processor.size["longest_edge"] == 8192
    assert components["processor"].image_processor.size["longest_edge"] == 8192


@pytest.mark.parametrize("size", [(64, 96), (384, 384), (768, 1344)])
def test_video_canvas_gears_cover_native_resizing(size):
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size

    gears = set(video_geometries(size))
    for ratio in np.linspace(0.25, 4, 10001):
        assert resolve_canvas_size(float(ratio), 1, 32, min(size), size[0] * size[1]) in gears
    assert (32, 32) not in gears


@pytest.mark.parametrize(
    "changes",
    [
        dict(mode="unknown"),
        dict(prompt=dict(max_tokens=True)),
        dict(references=dict(image=dict(max_count=1))),
        dict(mode="ref2va", references=dict(audio=dict(max_count=1))),
    ],
)
def test_invalid_or_ambiguous_capacity_policy_is_rejected(components, changes):
    config = dict(mode="t2va", prompt=dict(max_tokens=20), draft_size=[32, 64], draft_frames=124)
    config.update(changes)
    with pytest.raises(ValueError):
        plan_capacities(config, **components)


def test_chunk_size_changes_cache_rounding_but_not_dit_or_visual_capacity(components):
    config = dict(mode="fl2va", prompt=dict(max_tokens=513), draft_size=[32, 64], draft_frames=124)
    plans = [
        plan_capacities({**config, "qwen3_vl": {"prefill_chunk_length": chunk}}, **components) for chunk in (128, 256)
    ]
    for key in ("text_capacity", "dit_capacity", "image_qwen_tokens", "reference_video_tokens"):
        assert plans[0][key] == plans[1][key]
    for plan, chunk in zip(plans, (128, 256), strict=True):
        assert plan["kv_capacity"] % chunk == 0
        assert plan["text_capacity"] <= plan["kv_capacity"] < plan["text_capacity"] + chunk


def test_default_processor_budget_is_preserved_and_capacities_align(components):
    processor = components["processor"]
    configured = configure_processor(processor, dict(prefill_chunk_length=256))
    assert configured.image_processor.size == processor.image_processor.size
    assert configured.video_processor.size == processor.video_processor.size
    # A non-divisor chunk must satisfy both the hardware alignment and chunk ABI.
    config = dict(
        mode="t2va",
        prompt=dict(max_tokens=1025),
        draft_size=[32, 64],
        draft_frames=124,
        qwen3_vl=dict(prefill_chunk_length=192),
    )
    plan = plan_capacities(config, **components)
    assert plan["text_tokens"] == 1025 and plan["text_capacity"] == 1280
    assert plan["kv_capacity"] == 1536
    for key in (
        "text_capacity",
        "kv_capacity",
        "video_capacity",
        "audio_capacity",
        "dit_capacity",
        "qwen_visual_max_patches",
    ):
        assert plan[key] % 256 == 0
    assert plan["target_video_tokens"] == 74 and plan["target_audio_tokens"] == 414
    assert plan["dit_capacity"] == 1280 + 256 + 512
    with pytest.raises(ValueError, match="exceeds native"):
        plan_capacities(config, **{**components, "max_position_embeddings": 1400})


def test_unified_modes_share_graph_slots_without_changing_request_limits(components):
    config = dict(
        mode="all",
        prompt=dict(max_tokens=1024),
        draft_size=[384, 672],
        draft_frames=124,
        references=dict(image=dict(max_count=1, vae_short_edge=2048)),
        qwen3_vl=dict(prefill_chunk_length=256),
    )
    configs = mode_configs(config)
    assert tuple(configs) == ("t2va", "i2va", "l2va", "fl2va", "ref2va")
    assert "references" not in configs["t2va"] and "references" not in configs["fl2va"]
    assert configs["ref2va"]["references"] == config["references"]
    configs["t2va"]["prompt"]["max_tokens"] = 3
    assert config["prompt"]["max_tokens"] == configs["i2va"]["prompt"]["max_tokens"] == 1024
    plan = plan_unified_capacities(config, **components)
    assert plan["branches"]["base"]["video_capacity"] == 9984
    assert plan["branches"]["ref"]["video_capacity"] == 25856
    assert plan["modes"]["t2va"]["reference_video_tokens"] == 0
    assert plan["modes"]["fl2va"]["reference_video_tokens"] == 504
    assert plan["modes"]["ref2va"]["reference_video_tokens"] == 16384
    for mode, slots in plan["modes"].items():
        assert slots["prompt_max_tokens"] == 1024
        assert slots["kv_capacity"] == plan["encoder"]["kv_capacity"]
        assert slots["qwen_visual_max_patches"] == plan["encoder"]["qwen_visual_max_patches"]
        branch = plan["branches"]["ref" if mode == "ref2va" else "base"]
        for key in ("text_capacity", "video_capacity", "audio_capacity", "dit_capacity"):
            assert slots[key] == branch[key] and slots[key] % 256 == 0
    with pytest.raises(ValueError, match="mode=all"):
        mode_configs({**config, "mode": "t2va"})


def test_small_visual_gears_merge_without_padding_native_large_inputs_to_maximum(components):
    processor = SimpleNamespace(
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, merge_size=2, temporal_patch_size=2,
            size={"shortest_edge": 65536, "longest_edge": 16777216},
        ),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=16, merge_size=2, temporal_patch_size=2,
            size={"shortest_edge": 4096, "longest_edge": 25165824},
        ),
    )
    config = dict(
        mode="all", prompt=dict(max_tokens=1024), draft_size=[384, 672], draft_frames=124,
        references=dict(image=dict(max_count=1, vae_short_edge=2048),
                        video=dict(max_count=1, max_duration_seconds=5)),
    )
    plan = plan_unified_capacities(config, **{**components, "processor": processor})
    assert plan["encoder"]["qwen_visual_patch_gears"] == [1280, 65536]
    assert plan["encoder"]["qwen_visual_max_patches"] == 65536
    for mode in plan["modes"].values():
        assert mode["qwen_visual_patch_gears"] == [1280, 65536]
    single = plan_capacities(mode_configs(config)["i2va"], **{**components, "processor": processor})
    assert single["qwen_visual_patch_gears"] == [1024]
    assert single["image_qwen_tokens"] == 252
