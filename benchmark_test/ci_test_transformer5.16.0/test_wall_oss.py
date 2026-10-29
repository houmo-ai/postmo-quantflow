"""Wall's non-square geometry, routed causal cache and real export contracts."""

import numpy as np
import pytest
import torch
from _wall_oss_small_fixture import make_small_source
from PIL import Image

from xhmodel_merak.xh_other_model.models.wall_oss.graph import ActionDenoiser, ExpertStack
from xhmodel_merak.xh_other_model.models.wall_oss.processor import resize_image, vision_geometry


@pytest.fixture
def processor_resources(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from _unified_fixtures.wall_oss import prepare

    from xhmodel_merak.xh_other_model.models.wall_oss.build import package_resources

    kwargs, _ = prepare(SimpleNamespace(model_id="wall-oss-0.5"), tmp_path, monkeypatch)
    package_resources(kwargs["model_dir"], tmp_path / "resources")
    return tmp_path / "resources"


@pytest.mark.parametrize("instruction", ["token_4 " * 1500, "token_4 你好🙂 " * 1000], ids=["english", "unicode"])
def test_instruction_truncation_preserves_images_state_template_and_action_space(processor_resources, instruction):
    from xhmodel_merak.xh_other_model.models.wall_oss.processor import WallProcessor

    processor = WallProcessor(processor_resources, prefix_capacity=1024, truncate_instruction=True)
    images = [Image.new("RGB", (448, 448)), Image.new("RGB", (448, 448))]
    data = processor.prepare(instruction, images, [1.0] * 26, "x2_normal")
    assert data["ids"].shape[1] + 32 <= 1024
    assert (data["ids"] == processor.image_token_id).sum().item() == 512
    assert data["grid"].tolist() == [[1, 32, 32], [1, 32, 32]]
    assert "Proprioception: " + " ".join(["511"] * 26) in data["prompt"]
    assert data["prompt"].endswith("<|im_end|>\n<|im_start|>assistant\n")
    assert instruction.startswith(data["effective_instruction"])
    assert data["effective_instruction"].strip() and "�" not in data["effective_instruction"]
    stats = data["truncation"]
    assert stats["truncated"] and stats["side"] == "right"
    assert stats["original_prefix_tokens"] > 992
    assert stats["retained_prefix_tokens"] == data["ids"].shape[1]
    assert stats["retained_instruction_tokens"] < stats["original_instruction_tokens"]
    assert data["rope"][0].shape[2] == data["ids"].shape[1] + 32


def test_short_instruction_is_unchanged_and_legacy_mode_rejects_overflow(processor_resources):
    from xhmodel_merak.xh_other_model.models.wall_oss.processor import WallProcessor

    images = [Image.new("RGB", (448, 448)), Image.new("RGB", (448, 448))]
    legacy = WallProcessor(processor_resources, prefix_capacity=1024)
    current = WallProcessor(processor_resources, prefix_capacity=1024, truncate_instruction=True)
    before = legacy.prepare("token_4 token_5", images, [0.0] * 26, "x2_normal")
    after = current.prepare("token_4 token_5", images, [0.0] * 26, "x2_normal")
    assert not after["truncation"]["truncated"]
    assert after["effective_instruction"] == "token_4 token_5"
    assert before["prompt"] == after["prompt"]
    torch.testing.assert_close(before["ids"], after["ids"], rtol=0, atol=0)
    torch.testing.assert_close(before["rope"][0], after["rope"][0], rtol=0, atol=0)
    with pytest.raises(ValueError, match="exceed exported cache capacity"):
        legacy.prepare("token_4 " * 1500, images, [0.0] * 26, "x2_normal")


def test_fixed_prompt_is_never_truncated_to_make_room(processor_resources):
    from xhmodel_merak.xh_other_model.models.wall_oss.processor import WallProcessor

    processor = WallProcessor(processor_resources, prefix_capacity=544, truncate_instruction=True)
    images = [Image.new("RGB", (448, 448)), Image.new("RGB", (448, 448))]
    with pytest.raises(ValueError, match="fixed prompt"):
        processor.prepare("token_4", images, [1.0] * 26, "x2_normal")


@pytest.mark.parametrize(
    "fields",
    [{}, {"prefill_chunk_length": 256}, {"prefill_length": 256}, {"prefill_chunk_length": 256, "prefill_length": 256}],
)
def test_prefill_chunk_configuration_normalizes_legacy_name_without_mutation(fields):
    from xhmodel_merak.xh_other_model.models.wall_oss.build import validate_config

    config = dict(
        model=dict(quant_scheme=dict(target_device="XH2a", quant_type="w8a8_sefp")), target_device="XH2a", **fields
    )
    before = dict(config)
    actual = validate_config(config)
    assert actual["prefill_chunk_length"] == 256
    assert "prefill_length" not in actual
    assert actual["context_length"] == 1024
    assert config == before


@pytest.mark.parametrize(
    "fields",
    [
        {"prefill_chunk_length": 256, "prefill_length": 128},
        {"prefill_chunk_length": 0},
        {"prefill_length": True},
        {"prefill_chunk_length": 256, "context_length": 200},
    ],
)
def test_prefill_chunk_configuration_rejects_conflicts_and_invalid_capacities(fields):
    from xhmodel_merak.xh_other_model.models.wall_oss.build import validate_config

    with pytest.raises(ValueError):
        validate_config(
            dict(
                model=dict(quant_scheme=dict(target_device="XH2a", quant_type="w8a8_sefp")),
                target_device="XH2a",
                **fields,
            )
        )


def test_prefill_metadata_requires_a_length_and_preserves_legacy_cache_capacity():
    from xhmodel_merak.xh_other_model.models.wall_oss.config import normalize_prefill_config

    legacy = dict(prefill_length=256, context_length=2048)
    assert normalize_prefill_config(legacy) == dict(prefill_chunk_length=256, context_length=2048)
    with pytest.raises(ValueError, match="prefill_chunk_length"):
        normalize_prefill_config(dict(context_length=744))


@pytest.mark.parametrize("suffix", ["", "_w8a16", "_w8a8"])
def test_export_yaml_uses_canonical_chunk_name_and_1024_cache(suffix):
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    path = root / "configs_merak/workflows/xh2a/other_models/wall_oss" / f"wall_oss_05{suffix}.yaml"
    config = yaml.safe_load(path.read_text())["export"]
    assert config["context_length"] == 1024
    assert config["prefill_chunk_length"] == 256
    assert config["visual_capacity"] == 1024
    assert "prefill_length" not in config


@pytest.mark.parametrize(
    "size,expected",
    [((640, 480), (448, 336)), ((1920, 1080), (448, 252)), ((1080, 1920), (252, 448)), ((224, 224), (448, 448))],
)
def test_resize_preserves_official_long_edge_and_patch_alignment(size, expected):
    image = Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    assert resize_image(image, 448).size == expected


@pytest.mark.parametrize("hw", [(32, 32), (24, 32), (32, 18), (8, 6)])
def test_window_geometry_preserves_merge_order_and_excludes_padding(hw):
    geometry = vision_geometry(*hw, capacity=1024, head_dim=80, window_size=112, patch_size=14)
    order, reverse, cos, sin, mask, full_range, window_range = geometry
    n = hw[0] * hw[1]
    torch.testing.assert_close(order[:n].sort().values, torch.arange(n, dtype=torch.int32))
    assert int(mask.sum()) == n
    assert cos.shape == sin.shape == (1, 1, 1024, 80)
    assert full_range[0, :n, 1].eq(n).all()
    assert full_range[0, n:].eq(0).all()
    assert (window_range[0, :n, 1] - window_range[0, :n, 0] <= 64).all()
    merged = torch.arange(n // 4)[order[:n].reshape(-1, 4)[:, 0] // 4]
    torch.testing.assert_close(merged[reverse[: n // 4]], torch.arange(n // 4))


def test_separate_prefix_and_action_matches_joint_causal_and_reuses_prefix():
    torch.manual_seed(12)
    source = make_small_source()
    prefix = ExpertStack(source, expert=0, query_capacity=8, cache_capacity=16, sdk=False)
    action = ActionDenoiser(source, query_capacity=4, cache_capacity=16, sdk=False)
    x = torch.randn(1, 5, 128)
    noisy = torch.randn(1, 4, 6)
    dof = torch.ones_like(noisy)
    cos = torch.ones(1, 1, 9, 64)
    sin = torch.zeros_like(cos)
    caches = tuple(torch.zeros(1, 1, 16, 64) for _ in range(4))
    args = (torch.tensor([0], dtype=torch.int32), torch.tensor([5], dtype=torch.int32))
    with torch.no_grad():
        _, present = prefix(torch.nn.functional.pad(x, (0, 0, 0, 3)), cos[:, :, :8], sin[:, :, :8], *args, caches)
        frozen = tuple(t[:, :, :5].clone() for t in present)
        result, _ = action(
            noisy,
            dof,
            torch.zeros(1, 64),
            cos[:, :, :4],
            sin[:, :, :4],
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int32),
            present,
        )
        joint = source.joint(x, noisy, dof, torch.zeros(1, 64), cos, sin)
        torch.testing.assert_close(result, joint, atol=2e-6, rtol=2e-5)
        again, _ = action(
            noisy + 0.3,
            dof,
            torch.zeros(1, 64),
            cos[:, :, :4],
            sin[:, :, :4],
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int32),
            present,
        )
        assert not torch.equal(result, again)
        for old, new in zip(frozen, present, strict=True):
            torch.testing.assert_close(old, new[:, :, :5], atol=0, rtol=0)


@pytest.mark.parametrize("hw", [(8, 8), (12, 10), (10, 12)])
def test_packed_visual_matches_native_qwen25_window_and_merge(hw):
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

    from xhmodel_merak.xh_other_model.models.wall_oss.vision import PackedVision

    cfg = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=160,
        intermediate_size=192,
        num_heads=2,
        out_hidden_size=128,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=112,
        fullatt_block_indexes=[1],
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(4)
    source = Qwen2_5_VisionTransformerPretrainedModel(cfg).eval()
    graph = PackedVision(source, 128, sdk=False).eval()
    n = hw[0] * hw[1]
    patches = torch.randn(n, 1176)
    geometry = vision_geometry(*hw, capacity=128, head_dim=80)
    with torch.no_grad():
        expected = source(patches, torch.tensor([[1, *hw]])).pooler_output
        padded = torch.nn.functional.pad(patches, (0, 0, 0, 128 - n), value=77.0)[None]
        actual = graph(padded, *geometry)
    torch.testing.assert_close(actual[0, : n // 4], expected, atol=2e-6, rtol=2e-5)
    assert actual[:, n // 4 :].count_nonzero() == 0


def test_recipe_source_identity_and_rejected_public_inputs(tmp_path):
    import json

    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.model_build import describe_model
    from xhmodel_merak.model_catalog import get_recipe
    from xhmodel_merak.model_inference.inputs import validate_inputs
    from xhmodel_merak.model_integrations.wall_oss.source import resolve_config

    recipe = get_recipe("wall-oss-0.5")
    description = describe_model("wall-oss-0.5", full=True)
    assert description["task"] == "action-prediction"
    assert description["deployment_precision_options"]["policy"]["choices"] == [
        "w16a16_sefp",
        "w8a16_sefp",
        "w8a8_sefp",
    ]
    valid = dict(instruction="pick up the cup", images=["front.png", "wrist.png"], state=[0.0] * 26)
    values = validate_inputs(recipe, valid)
    assert values["steps"] == 10 and values["norm_key"] == "x2_normal"
    for patch in [
        dict(images=["front.png"]),
        dict(state=[0.0] * 25),
        dict(state=[float("nan")] * 26),
        dict(steps=0),
        dict(dof_mask=[2] * 26),
        dict(unknown=True),
    ]:
        with pytest.raises(ModelError):
            validate_inputs(recipe, {**valid, **patch})
    (tmp_path / "config.json").write_text(json.dumps(dict(model_type="qwen2_5_vl")))
    with pytest.raises(ModelError, match="Wall-OSS"):
        resolve_config(recipe=recipe, model_dir=tmp_path)


def test_normalization_uses_requested_embodiment_and_handles_zero_delta():
    from xhmodel_merak.xh_other_model.models.wall_oss.processor import WallProcessor

    processor = WallProcessor.__new__(WallProcessor)
    processor.normalizers = {"propri": {"robot_a": {"min": [0.0] * 26, "delta": [0.0, 2.0] + [1.0] * 24}}}
    result = processor.normalize([1.0] * 26, "robot_a", "propri")
    np.testing.assert_array_equal(result, np.array([1.0, 0.0] + [1.0] * 24))
    with pytest.raises(ValueError, match="norm_key"):
        processor.normalize([1.0] * 26, "missing", "propri")
