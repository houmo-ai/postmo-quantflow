"""Artifact/schema boundary checks; real learned graphs live in the graph/E2E cases."""

import json
from copy import deepcopy
from pathlib import Path

import onnx
import pytest
import torch
import yaml
from onnx import TensorProto, helper

from xhmodel_merak.model_integrations.sol_h3.artifact import validate_precision
from xhmodel_merak.xh_other_model.models.sol_h3.artifacts import (
    _validate_ltx_reference_policy,
    read_metadata,
)
from xhmodel_merak.xh_other_model.models.sol_h3.build import validate_build_config


@pytest.mark.parametrize(
    "latent_frames,output_frames,tiles,graphs",
    [
        (16, 121, [(0, 16, 0, 121)], ["ltx_video_decode"]),
        (46, 361, [(0, 16, 0, 121), (12, 29, 96, 225), (25, 42, 200, 329), (38, 46, 304, 361)],
         ["ltx_video_decode_t16", "ltx_video_decode_t17", "ltx_video_decode_t8"]),
    ],
)
def test_decoder_temporal_gears_route_tiles_and_stitch_every_frame(
    tmp_path, monkeypatch, latent_frames, output_frames, tiles, graphs
):
    from types import SimpleNamespace

    import numpy as np
    from _sol_h3_tiny_components import video_decoder_source
    from torch.nn import functional as F

    from xhmodel_merak.xh_other_model.models.sol_h3 import source_io
    from xhmodel_merak.xh_other_model.models.sol_h3.build import BundleExporter
    from xhmodel_merak.xh_other_model.models.sol_h3.runtime import SolH3Runtime

    native = [SimpleNamespace(
        in_coords=(slice(0, 1), slice(0, 128), slice(start, end), slice(0, 1), slice(0, 2)),
        out_coords=(slice(0, 1), slice(0, 3), slice(first, last), slice(0, 2), slice(0, 2)),
        masks_1d=[torch.ones(1) for _ in range(5)],
    ) for start, end, first, last in tiles]
    monkeypatch.setattr(source_io, "video_decoder", lambda *_: (video_decoder_source(), native, True))
    monkeypatch.setattr(source_io, "prepare_video_decoder", lambda source: source)
    runner = object.__new__(BundleExporter)
    runner.assets, runner.cfg = None, {}
    runner.refined_latent = torch.linspace(-1, 1, latent_frames).view(1, 1, -1, 1, 1).expand(1, 128, -1, 1, 2)
    runner.meta = {"schema_version": 8, "mode": "all", "video_decoder_graphs": [], "graphs": {}}
    runner.progress = lambda *_: None
    runner.resource = lambda _name, masks, **_: np.savez(tmp_path / "masks.npz", **masks)
    runner.graph = lambda name, *_args, **_kwargs: runner.meta["graphs"].update({name: {}})
    runner.video_vae()
    assert runner.meta["video_decoder_graphs"] == graphs
    assert [tile["output"][2][:2] for tile in runner.meta["decoder_tiles"]] == [[a, b] for _, _, a, b in tiles]

    # The learned decoder is replaced by a linear ramp; tile routing, overlap
    # normalization and the final frame still run through the real Host.
    runtime = object.__new__(SolH3Runtime)
    runtime.meta = {**runner.meta, "refiner_frames": output_frames, "refiner_size": [2, 2]}
    runtime.device, runtime.sessions = torch.device("cpu"), {}
    runtime.numpy_resource = lambda _: np.load(tmp_path / "masks.npz")
    runtime.invoke = lambda _name, sample, *_: (
        F.interpolate(sample[:, 0, :, 0, 0].unsqueeze(1), size=(sample.shape[2] - 1) * 8 + 1,
                      mode="linear", align_corners=True).view(1, 1, -1, 1, 1).expand(-1, 3, -1, 2, 2),
    )
    result = runtime.video_decode(runner.refined_latent)
    expected = ((torch.linspace(-1, 1, output_frames) + 1) * 127.5).round().byte().numpy()
    assert result.shape == (output_frames, 2, 2, 3)
    np.testing.assert_allclose(result[:, 0, 0, 0], expected, atol=1, rtol=0)


@pytest.mark.parametrize("epsilon", [None, 2**-24])
def test_encoder_epsilon_is_an_explicit_optional_build_policy(epsilon):
    repo = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (repo / "configs_merak/workflows/xh2a/other_models/sol_h3/sol_h3_all_384p_768p_w4.yaml").read_text()
    )["export"]
    config["ltx_reference_conditioning"] = {"enabled": True, "encoder_norm_eps": epsilon}
    before = deepcopy(config)
    result = validate_build_config(config)
    assert result["ltx_reference_conditioning"] == before["ltx_reference_conditioning"]
    assert config == before


@pytest.mark.parametrize("epsilon", [2**-25, True])
def test_encoder_epsilon_rejects_unsupported_values(epsilon):
    with pytest.raises(ValueError, match="encoder_norm_eps"):
        validate_build_config({"ltx_reference_conditioning": {"encoder_norm_eps": epsilon}})


def test_only_ltx_reference_image_encoder_has_the_w16_component_exception():
    meta = {
        "quant_type": "w8a8_sefp",
        "feature_graphs": ["qwen"],
        "graphs": {
            "qwen": {"quant_type": "w4a8h0_ssfp"},
            "ltx_image_encode": {"quant_type": "w16a16_sefp"},
            "ltx_denoiser": {"quant_type": "w8a8_sefp"},
        },
    }
    validate_precision(meta, deployment={"video": "w8a8_sefp"}, checkpoint_weight="w4")
    for name, bad in (("ltx_image_encode", "w8a8_sefp"), ("ltx_denoiser", "w16a16_sefp")):
        changed = deepcopy(meta)
        changed["graphs"][name]["quant_type"] = bad
        with pytest.raises(ValueError, match=name):
            validate_precision(changed, deployment={"video": "w8a8_sefp"}, checkpoint_weight="w4")


def test_ltx_policy_distinguishes_target_frames_from_the_appended_guide():
    policy = {
        "enabled": True,
        "graph_mode": "static_maximum_guide",
        "first_frame_tokens": 1008,
        "target_tokens": 16128,
        "video_tokens": 17136,
        "end_pixel_frame": 120,
        "encoder_residual_scale": 32.0,
        "encoder_norm_eps": None,
        "image_crf": 33,
    }
    assert _validate_ltx_reference_policy({"refiner_size": [768, 1344], "ltx_reference_conditioning": policy}) is policy
    for key, value in (
        ("target_tokens", 17136),
        ("video_tokens", 16128),
        ("end_pixel_frame", 15),
        ("encoder_norm_eps", 2**-25),
    ):
        invalid = {**policy, key: value}
        with pytest.raises(ValueError):
            _validate_ltx_reference_policy({"refiner_size": [768, 1344], "ltx_reference_conditioning": invalid})


def test_role_graph_rope_abi_uses_the_native_six_frequency_blocks(tmp_path):
    # Synthetic ONNX is intentional here: this test checks serialized port
    # validation only, independently of the unrelated expensive stages.
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
    from test_sol_h3_unified_api import bundle_fixture

    from xhmodel_merak.xh_other_model.models.sol_h3.role_layout import ROLE_NAMES, role_spans

    meta = bundle_fixture(tmp_path)
    caps = [9472, 512, 512, 0, 0]
    count = sum(caps)
    plan = dict(
        video_capacity=9472,
        audio_capacity=512,
        text_capacity=512,
        dit_capacity=count,
        prompt_max_tokens=512,
        target_video_tokens=9324,
        target_audio_tokens=414,
        reference_video_tokens=0,
        reference_audio_tokens=0,
        text_tokens=512,
        presentation_overhead=0,
    )
    meta.update(
        capacity_plan=plan,
        prompt_max_tokens=512,
        h3_packed_inputs=True,
        h3_rope_config={"rope_freq_dim": 16, "rope_theta": 10000.0},
    )
    meta["h3_role_layout"] = dict(
        format="fixed_adaln_role_slots_v1",
        role_names=list(ROLE_NAMES),
        role_capacities=caps,
        role_spans=[list(v) for v in role_spans(caps)],
        attention_block_size=256,
        attention_heads=56,
        attention_mode="broadcast_key_mask",
        sequence_capacity=count,
    )
    cos, _ = MiniMaxH3RotaryPosEmbed(**meta["h3_rope_config"])(torch.zeros(1, 3))
    width = cos.shape[-1]
    assert width == 96
    shapes = {
        "video_tokens": [1, 9472, 96],
        "audio_tokens": [1, 512, 64],
        "text_features": [1, 512, 128],
        "step": [1],
        "cos": [1, 1, count, width],
        "sin": [1, 1, count, width],
        "role_pack": [count],
        "head_unpack": [9984],
        "key_mask": [1, 1, 1, count],
    }
    names = list(shapes)
    inputs = [
        helper.make_tensor_value_info(
            n, TensorProto.INT32 if n in ("step", "role_pack", "head_unpack") else TensorProto.FLOAT16, s
        )
        for n, s in shapes.items()
    ]
    outputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT16, shapes[k])
        for n, k in (("video_velocity", "video_tokens"), ("audio_velocity", "audio_tokens"))
    ]
    nodes = [
        helper.make_node("Identity", [k], [n])
        for n, k in (("video_velocity", "video_tokens"), ("audio_velocity", "audio_tokens"))
    ]
    graph = meta["graphs"]["h3_denoiser"]
    graph.update(input_names=names, output_names=[v.name for v in outputs])
    path = tmp_path / graph["path"]
    model = helper.make_model(helper.make_graph(nodes, "ports_only", inputs, outputs))
    onnx.save(model, path)
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    assert read_metadata(tmp_path)["h3_role_layout"]["sequence_capacity"] == count
    for value in model.graph.input:
        if value.name in ("cos", "sin"):
            value.type.tensor_type.shape.dim[-1].dim_value = width // 2
    onnx.save(model, path)
    with pytest.raises(ValueError, match="fixed-role graph ABI"):
        read_metadata(tmp_path)
