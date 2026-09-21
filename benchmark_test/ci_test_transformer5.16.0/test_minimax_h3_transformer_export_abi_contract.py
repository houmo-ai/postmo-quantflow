"""CPU contracts for MiniMax-H3 transformer export / HMONNX ABI."""

import pytest
import torch
from torch import nn

from xhmodel_merak.xh_other_model.models.minimax_h3.components_hmonnx import _HMONNXTransformerModule
from xhmodel_merak.xh_other_model.models.minimax_h3.minimax_h3_converter import (
    TENSOR_NAMES,
    MiniMaxH3Converter,
)
from xhmodel_merak.xh_other_model.models.minimax_h3.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3Transformer3DModel,
)
from xhmodel_merak.xh_other_model.models.minimax_h3.transformer_wrapper import (
    MINIMAX_H3_FIXED_TEMB_TIMESTEPS,
    MiniMaxH3TransformerWrapper,
    build_minimax_h3_transformer_conditioning,
    build_pack_gather_indices,
    build_packed_attention_mask,
    disable_transformer_block_cpu_offload,
    pad_minimax_h3_temb,
    prepare_transformer_block_cpu_offload,
)


TRANSFORMER_INPUT_NAMES = TENSOR_NAMES["transformer"][0]


class _RecordingRuntime:
    def __init__(self, expected_shapes):
        self.expected_shapes = expected_shapes
        self.inputs = None

    def supports(self, *inputs: torch.Tensor) -> bool:
        return tuple(tuple(tensor.shape) for tensor in inputs) == self.expected_shapes

    def __call__(self, *inputs: torch.Tensor):
        self.inputs = inputs
        return (inputs[0], inputs[1])


def _make_model() -> MiniMaxH3Transformer3DModel:
    torch.manual_seed(5)
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=2,
        audio_in_channels=4,
        patch_size=(1, 1, 1),
        text_dim=12,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=1,
        enable_ffn_scaling=False,
    ).eval()


def test_tensor_names_use_host_precomputed_transformer_abi():
    assert TRANSFORMER_INPUT_NAMES == (
        "hidden_states",
        "audio_hidden_states",
        "encoder_hidden_states",
        "temb",
        "timestep_indices",
        "adaln_indices",
        "rotary_cos",
        "rotary_sin",
        "video_indices",
        "audio_indices",
        "pack_gather_indices",
        "encoder_attention_mask",
        "packed_attention_mask",
    )
    assert "timestep" not in TRANSFORMER_INPUT_NAMES
    assert "token_tags" not in TRANSFORMER_INPUT_NAMES
    assert "position_ids" not in TRANSFORMER_INPUT_NAMES
    assert "text_indices" not in TRANSFORMER_INPUT_NAMES


def test_build_pack_gather_indices_handles_noncontiguous_layout():
    text_indices = torch.tensor([0, 1], dtype=torch.long)
    video_indices = torch.tensor([2, 3, 5], dtype=torch.long)
    audio_indices = torch.tensor([4], dtype=torch.long)

    pack_gather_indices = build_pack_gather_indices(text_indices, video_indices, audio_indices)

    assert pack_gather_indices.dtype == torch.int32
    torch.testing.assert_close(
        pack_gather_indices,
        torch.tensor([0, 1, 2, 3, 5, 4], dtype=torch.int32),
    )

    concat = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    packed = concat[:, pack_gather_indices.long(), :]
    torch.testing.assert_close(packed.view(-1), torch.tensor([0, 1, 2, 3, 5, 4], dtype=torch.float32))

    with pytest.raises(ValueError, match="permutation"):
        build_pack_gather_indices(
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([2, 2], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
        )


def test_build_packed_attention_mask_writes_text_rows_only():
    encoder_attention_mask = torch.zeros(1, 1, 1, 3, dtype=torch.float16)
    encoder_attention_mask[..., 2:] = torch.finfo(torch.float16).min
    text_indices = torch.tensor([0, 1, 2], dtype=torch.long)

    packed = build_packed_attention_mask(encoder_attention_mask, text_indices, sequence_length=6)

    assert tuple(packed.shape) == (1, 1, 1, 6)
    torch.testing.assert_close(packed[..., :3], encoder_attention_mask)
    torch.testing.assert_close(packed[..., 3:], torch.zeros(1, 1, 1, 3, dtype=torch.float16))

    with pytest.raises(ValueError, match="text_indices"):
        build_packed_attention_mask(encoder_attention_mask, torch.tensor([0, 1], dtype=torch.long), 6)


def test_build_minimax_h3_transformer_conditioning_matches_fp_formula():
    model = _make_model()
    sequence_length = 6
    timestep = torch.tensor([0.25, 0.75], dtype=torch.float32)
    timestep_indices = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    token_tags = torch.tensor([1, 1, 0, 0, 2, 2], dtype=torch.long)
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32)
    position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32)

    temb, adaln_indices, rotary_cos, rotary_sin = build_minimax_h3_transformer_conditioning(
        model,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        fixed_temb_timesteps=None,
    )

    expected_temb = model.time_embedder(model.time_proj(timestep)).half()
    expected_adaln = (timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags).to(torch.int32)
    expected_cos, expected_sin = model.rope(position_ids)

    torch.testing.assert_close(temb, expected_temb)
    torch.testing.assert_close(adaln_indices, expected_adaln)
    torch.testing.assert_close(rotary_cos, expected_cos.half())
    torch.testing.assert_close(rotary_sin, expected_sin.half())


def test_pad_minimax_h3_temb_to_fixed_export_capacity():
    temb = torch.randn(2, 8, dtype=torch.float16)
    padded = pad_minimax_h3_temb(temb, MINIMAX_H3_FIXED_TEMB_TIMESTEPS)

    assert tuple(padded.shape) == (MINIMAX_H3_FIXED_TEMB_TIMESTEPS, 8)
    torch.testing.assert_close(padded[:2], temb)
    torch.testing.assert_close(padded[2:], torch.zeros(1, 8, dtype=torch.float16))

    model = _make_model()
    sequence_length = 6
    timestep = torch.tensor([0.25, 0.75], dtype=torch.float32)
    timestep_indices = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    token_tags = torch.tensor([1, 1, 0, 0, 2, 2], dtype=torch.long)
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32)

    temb, adaln_indices, _, _ = build_minimax_h3_transformer_conditioning(
        model,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
    )
    assert tuple(temb.shape) == (MINIMAX_H3_FIXED_TEMB_TIMESTEPS, 8)
    assert int(timestep_indices.max()) < timestep.numel()
    assert int(adaln_indices.max()) < timestep.numel() * MINIMAX_H3_MODALITY_NUM

    with pytest.raises(ValueError, match="exceeds the fixed export capacity"):
        pad_minimax_h3_temb(torch.randn(4, 8), MINIMAX_H3_FIXED_TEMB_TIMESTEPS)


def test_export_wrapper_consumes_precomputed_abi_and_requires_mask():
    # Wrapper itself does not cast dtypes; host precompute aligns temb to model.dtype.
    model = _make_model().half()
    model.time_proj = model.time_proj.float()
    model.time_embedder = model.time_embedder.float()
    model.rope = model.rope.float()
    wrapper = MiniMaxH3TransformerWrapper(model)

    video = torch.randn(1, 2, 2, dtype=torch.float16)
    audio = torch.randn(1, 1, 4, dtype=torch.float16)
    text = torch.randn(1, 3, 12, dtype=torch.float16)
    sequence_length = 6
    timestep = torch.tensor([0.5], dtype=torch.float32)
    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    token_tags = torch.tensor([1, 1, 1, 0, 0, 2], dtype=torch.long)
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32)
    position_ids[:3, 0] = torch.arange(3, dtype=torch.float32)
    video_indices = torch.tensor([3, 4], dtype=torch.long)
    audio_indices = torch.tensor([5], dtype=torch.long)
    text_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    attention_mask = torch.zeros(1, 1, 1, 3, dtype=torch.float16)
    attention_mask[..., 2:] = torch.finfo(attention_mask.dtype).min
    packed_attention_mask = build_packed_attention_mask(
        attention_mask,
        text_indices,
        sequence_length=sequence_length,
    )
    pack_gather_indices = build_pack_gather_indices(
        text_indices,
        video_indices,
        audio_indices,
        sequence_length=sequence_length,
    )

    temb, adaln_indices, rotary_cos, rotary_sin = build_minimax_h3_transformer_conditioning(
        model,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
    )
    # Wrapper does not cast; caller/host must feed body-aligned conditioning tensors.
    temb = temb.to(dtype=torch.float16)
    rotary_cos = rotary_cos.to(dtype=torch.float16)
    rotary_sin = rotary_sin.to(dtype=torch.float16)

    with torch.no_grad():
        video_out, audio_out = wrapper(
            video,
            audio,
            text,
            temb,
            timestep_indices,
            adaln_indices,
            rotary_cos,
            rotary_sin,
            video_indices,
            audio_indices,
            pack_gather_indices,
            attention_mask,
            packed_attention_mask,
        )

    assert video_out.shape == video.shape
    assert audio_out.shape == audio.shape

    with pytest.raises(TypeError):
        wrapper(
            video,
            audio,
            text,
            temb,
            timestep_indices,
            adaln_indices,
            rotary_cos,
            rotary_sin,
            video_indices,
            audio_indices,
            pack_gather_indices,
            attention_mask,
        )

    assert hasattr(wrapper.transformer.transformer_blocks[0].attn, "masked_add")
    assert hasattr(wrapper.transformer.token_refiner.refiner_blocks[0].attn, "masked_add")
    assert not hasattr(wrapper.transformer, "mask_scatter")
    assert not hasattr(wrapper.transformer, "pack_scatter")
    assert hasattr(wrapper.transformer, "pack_concat")
    assert hasattr(wrapper.transformer, "pack_gather")
    assert hasattr(wrapper.transformer, "video_gather")
    assert hasattr(wrapper.transformer, "audio_gather")
    assert hasattr(wrapper.transformer.transformer_blocks[0], "adaln_gather")
    assert hasattr(wrapper.transformer.transformer_blocks[0], "norm1_h")
    assert hasattr(wrapper.transformer.transformer_blocks[0], "norm2_h")
    assert hasattr(wrapper.transformer.token_refiner, "final_norm_h")
    assert hasattr(wrapper.transformer.norm_out, "norm_h")


def test_token_refiner_attention_skips_attn_fixed_scale():
    """TokenRefiner residual is plain add; only main-block attn keeps /16 + rhs_scale=16."""
    model = _make_model().half()
    assert model.token_refiner.refiner_blocks[0].attn.apply_attn_fixed_scale is False
    assert model.transformer_blocks[0].attn.apply_attn_fixed_scale is True

    wrapper = MiniMaxH3TransformerWrapper(model)
    ref_attn = wrapper.transformer.token_refiner.refiner_blocks[0].attn
    main_attn = wrapper.transformer.transformer_blocks[0].attn
    assert type(ref_attn).__name__ == "_MiniMaxH3TokenRefinerAttention"
    assert type(main_attn).__name__ == "_MiniMaxH3Attention"
    assert type(ref_attn)._apply_attn_fixed_scale is False
    assert type(main_attn)._apply_attn_fixed_scale is True

    with torch.no_grad():
        for src, dst in zip(ref_attn.parameters(), main_attn.parameters(), strict=True):
            dst.copy_(src)
        hidden_states = torch.randn(1, 2, 16, dtype=torch.float16)
        attention_mask = torch.zeros(1, 1, 2, 2, dtype=torch.float16)
        ref_out = ref_attn(hidden_states, attention_mask=attention_mask)
        main_out = main_attn(hidden_states, attention_mask=attention_mask)
    torch.testing.assert_close(main_out, ref_out / 16, rtol=1e-3, atol=1e-3)


def test_hmonnx_transformer_host_precomputes_and_rejects_missing_mask():
    model = _make_model()
    sequence_length = 6
    video = torch.randn(1, 2, 2)
    audio = torch.randn(1, 1, 4)
    text = torch.randn(1, 3, 12)
    timestep = torch.tensor([0.5], dtype=torch.float32)
    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    token_tags = torch.tensor([1, 1, 1, 0, 0, 2], dtype=torch.long)
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32)
    position_ids[:3, 0] = torch.arange(3, dtype=torch.float32)
    video_indices = torch.tensor([3, 4], dtype=torch.long)
    audio_indices = torch.tensor([5], dtype=torch.long)
    text_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    attention_mask = torch.zeros(1, 1, 1, 3)
    packed_attention_mask = build_packed_attention_mask(
        attention_mask,
        text_indices,
        sequence_length=sequence_length,
    )
    pack_gather_indices = build_pack_gather_indices(
        text_indices,
        video_indices,
        audio_indices,
        sequence_length=sequence_length,
    )

    temb, adaln_indices, rotary_cos, rotary_sin = build_minimax_h3_transformer_conditioning(
        model,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
    )
    assert tuple(temb.shape) == (MINIMAX_H3_FIXED_TEMB_TIMESTEPS, model.config.time_embed_dim)
    expected_shapes = (
        tuple(video.shape),
        tuple(audio.shape),
        tuple(text.shape),
        tuple(temb.shape),
        tuple(timestep_indices.shape),
        tuple(adaln_indices.shape),
        tuple(rotary_cos.shape),
        tuple(rotary_sin.shape),
        tuple(video_indices.shape),
        tuple(audio_indices.shape),
        tuple(pack_gather_indices.shape),
        tuple(attention_mask.shape),
        tuple(packed_attention_mask.shape),
    )
    runtime = _RecordingRuntime(expected_shapes)
    adapter = _HMONNXTransformerModule(runtime, model, TRANSFORMER_INPUT_NAMES)
    expected_config = model.config
    adapter._release_fallback()

    assert adapter.fallback is None
    assert adapter.config is expected_config
    assert adapter.config.patch_size == expected_config.patch_size
    assert isinstance(adapter.time_proj, nn.Module)
    assert isinstance(adapter.time_embedder, nn.Module)
    assert isinstance(adapter.rope, nn.Module)

    output = adapter(
        hidden_states=video,
        audio_hidden_states=audio,
        encoder_hidden_states=text,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        encoder_attention_mask=attention_mask,
    )

    assert runtime.inputs is not None
    assert len(runtime.inputs) == len(TRANSFORMER_INPUT_NAMES)
    # Adapter aligns temb/rotary to model/body dtype; graph indices stay int32.
    # Shorter unique-timestep vectors are zero-padded to the fixed export capacity.
    expected_temb = temb.to(dtype=model.dtype)
    expected_rotary_cos = rotary_cos.to(dtype=model.dtype)
    expected_rotary_sin = rotary_sin.to(dtype=model.dtype)
    torch.testing.assert_close(runtime.inputs[3], expected_temb)
    assert tuple(runtime.inputs[3].shape) == (MINIMAX_H3_FIXED_TEMB_TIMESTEPS, model.config.time_embed_dim)
    torch.testing.assert_close(
        runtime.inputs[3][timestep.numel() :],
        torch.zeros_like(runtime.inputs[3][timestep.numel() :]),
    )
    torch.testing.assert_close(runtime.inputs[5], adaln_indices.to(torch.int32))
    torch.testing.assert_close(runtime.inputs[6], expected_rotary_cos)
    torch.testing.assert_close(runtime.inputs[7], expected_rotary_sin)
    assert runtime.inputs[4].dtype == torch.int32
    assert runtime.inputs[8].dtype == torch.int32
    assert runtime.inputs[9].dtype == torch.int32
    assert runtime.inputs[10].dtype == torch.int32
    torch.testing.assert_close(runtime.inputs[10], pack_gather_indices.to(torch.int32))
    assert output[0].shape == video.shape
    assert output[1].shape == audio.shape

    with pytest.raises(ValueError, match="encoder_attention_mask"):
        adapter(
            hidden_states=video,
            audio_hidden_states=audio,
            encoder_hidden_states=text,
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            encoder_attention_mask=None,
        )

    with pytest.raises(ValueError, match="attention_kwargs"):
        adapter(
            hidden_states=video,
            audio_hidden_states=audio,
            encoder_hidden_states=text,
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            encoder_attention_mask=attention_mask,
            attention_kwargs={"scale": 1.0},
        )


def test_hmonnx_transformer_raises_on_shape_mismatch_without_fallback():
    model = _make_model()
    runtime = _RecordingRuntime(((1, 2, 2),))
    adapter = _HMONNXTransformerModule(
        runtime,
        model,
        ("hidden_states",),
    )
    adapter._release_fallback()

    with pytest.raises(RuntimeError, match="do not match the exported graph"):
        adapter(
            hidden_states=torch.randn(1, 3, 2),
            audio_hidden_states=torch.randn(1, 1, 4),
            encoder_hidden_states=torch.randn(1, 3, 12),
            timestep=torch.tensor([0.5]),
            timestep_indices=torch.zeros(6, dtype=torch.long),
            token_tags=torch.zeros(6, dtype=torch.long),
            position_ids=torch.zeros(6, 3),
            video_indices=torch.tensor([3, 4], dtype=torch.long),
            audio_indices=torch.tensor([5], dtype=torch.long),
            text_indices=torch.tensor([0, 1, 2], dtype=torch.long),
            encoder_attention_mask=torch.zeros(1, 1, 1, 3),
        )


def test_converter_builds_in_range_adaln_and_packing_indices():
    """Export dummies must stay inside AdaLN table rows and packed-sequence bounds."""
    converter = MiniMaxH3Converter.__new__(MiniMaxH3Converter)
    converter.device = torch.device("cpu")
    converter.seed = 0
    num_timesteps = MINIMAX_H3_FIXED_TEMB_TIMESTEPS
    text_len, video_len, audio_len = 5, 2, 1
    sequence_length = text_len + video_len + audio_len
    converter.cfg = {
        "minimax_h3": {
            "input_shapes": {
                "transformer": [
                    {"name": "hidden_states", "shape": [1, video_len, 2], "dtype": "float16"},
                    {"name": "audio_hidden_states", "shape": [1, audio_len, 4], "dtype": "float16"},
                    {"name": "encoder_hidden_states", "shape": [1, text_len, 12], "dtype": "float16"},
                    {"name": "temb", "shape": [num_timesteps, 8], "dtype": "float16"},
                    {"name": "timestep_indices", "shape": [sequence_length], "dtype": "int64"},
                    {"name": "adaln_indices", "shape": [sequence_length], "dtype": "int64"},
                    {"name": "rotary_cos", "shape": [sequence_length, 6], "dtype": "float16"},
                    {"name": "rotary_sin", "shape": [sequence_length, 6], "dtype": "float16"},
                    {"name": "video_indices", "shape": [video_len], "dtype": "int64"},
                    {"name": "audio_indices", "shape": [audio_len], "dtype": "int64"},
                    {"name": "pack_gather_indices", "shape": [sequence_length], "dtype": "int64"},
                    {
                        "name": "encoder_attention_mask",
                        "shape": [1, 1, 1, text_len],
                        "dtype": "float16",
                        "valid_length": 3,
                    },
                    {
                        "name": "packed_attention_mask",
                        "shape": [1, 1, 1, sequence_length],
                        "dtype": "float16",
                    },
                ]
            }
        }
    }

    inputs = converter._build_inputs("transformer", pipe=None)
    names = TRANSFORMER_INPUT_NAMES
    tensors = dict(zip(names, inputs, strict=True))

    timestep_indices = tensors["timestep_indices"]
    adaln_indices = tensors["adaln_indices"]
    video_indices = tensors["video_indices"]
    audio_indices = tensors["audio_indices"]
    pack_gather_indices = tensors["pack_gather_indices"]
    text_indices = torch.arange(text_len, dtype=torch.int64)

    assert int(timestep_indices.min()) >= 0
    assert int(timestep_indices.max()) < num_timesteps
    assert int(adaln_indices.min()) >= 0
    assert int(adaln_indices.max()) < num_timesteps * MINIMAX_H3_MODALITY_NUM

    torch.testing.assert_close(
        video_indices,
        torch.arange(text_len, text_len + video_len, dtype=torch.int64),
    )
    torch.testing.assert_close(
        audio_indices,
        torch.arange(text_len + video_len, sequence_length, dtype=torch.int64),
    )
    # Converter materializes indices in the configured export dtype (int64 here).
    torch.testing.assert_close(
        pack_gather_indices,
        torch.arange(sequence_length, dtype=torch.int64),
    )

    expected_tags = torch.empty(sequence_length, dtype=torch.int64)
    expected_tags[text_indices] = 1
    expected_tags[video_indices] = 0
    expected_tags[audio_indices] = 2
    torch.testing.assert_close(adaln_indices, timestep_indices * MINIMAX_H3_MODALITY_NUM + expected_tags)


def test_prepare_transformer_block_cpu_offload_parks_blocks_and_restores_flag():
    """Warm-up helper parks blocks on CPU and must be disabled before Dynamo capture."""
    torch.manual_seed(5)
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=2,
        audio_in_channels=4,
        patch_size=(1, 1, 1),
        text_dim=12,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=1,
        enable_ffn_scaling=False,
    ).eval()
    # Export packing uses hardcoded float16 zeros for Dynamo-friendly tracing.
    model = model.half()
    model.time_proj = model.time_proj.float()
    model.time_embedder = model.time_embedder.float()
    model.rope = model.rope.float()
    wrapper = MiniMaxH3TransformerWrapper(model)
    transformer = wrapper.transformer
    assert len(transformer.transformer_blocks) == 2

    prepare_transformer_block_cpu_offload(transformer, "cpu")
    assert transformer.enable_block_cpu_offload is True
    assert all(next(block.parameters()).device.type == "cpu" for block in transformer.transformer_blocks)

    video = torch.randn(1, 2, 2, dtype=torch.float16)
    audio = torch.randn(1, 1, 4, dtype=torch.float16)
    text = torch.randn(1, 3, 12, dtype=torch.float16)
    sequence_length = 6
    timestep = torch.tensor([0.5], dtype=torch.float32)
    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    token_tags = torch.tensor([1, 1, 1, 0, 0, 2], dtype=torch.long)
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32)
    position_ids[:3, 0] = torch.arange(3, dtype=torch.float32)
    video_indices = torch.tensor([3, 4], dtype=torch.long)
    audio_indices = torch.tensor([5], dtype=torch.long)
    text_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    attention_mask = torch.zeros(1, 1, 1, 3, dtype=torch.float16)
    packed_attention_mask = build_packed_attention_mask(
        attention_mask,
        text_indices,
        sequence_length=sequence_length,
    )
    pack_gather_indices = build_pack_gather_indices(
        text_indices,
        video_indices,
        audio_indices,
        sequence_length=sequence_length,
    )

    temb, adaln_indices, rotary_cos, rotary_sin = build_minimax_h3_transformer_conditioning(
        transformer,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
    )
    temb = temb.to(dtype=torch.float16)
    rotary_cos = rotary_cos.to(dtype=torch.float16)
    rotary_sin = rotary_sin.to(dtype=torch.float16)

    with torch.no_grad():
        video_out, audio_out = wrapper(
            video,
            audio,
            text,
            temb,
            timestep_indices,
            adaln_indices,
            rotary_cos,
            rotary_sin,
            video_indices,
            audio_indices,
            pack_gather_indices,
            attention_mask,
            packed_attention_mask,
        )

    assert video_out.shape == video.shape
    assert audio_out.shape == audio.shape
    assert all(next(block.parameters()).device.type == "cpu" for block in transformer.transformer_blocks)

    disable_transformer_block_cpu_offload(transformer)
    assert transformer.enable_block_cpu_offload is False


def test_wrapper_demo_adapter_to_keeps_blocks_on_cpu_under_block_offload():
    """ComponentsManager CustomOffloadHook calls adapter.to(cuda); that must not hoist every block."""
    from examples_merak.aigc.minimaxh3.wrapper_demo import _ExportABITransformerAdapter

    if not torch.cuda.is_available():
        pytest.skip("CUDA required to exercise CustomOffloadHook-style adapter.to(cuda)")

    torch.manual_seed(6)
    model = (
        MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            hidden_size=16,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=2,
            audio_in_channels=4,
            patch_size=(1, 1, 1),
            text_dim=12,
            freq_dim=8,
            time_embed_hidden_dim=16,
            time_embed_dim=8,
            rope_freq_dim=1,
            enable_ffn_scaling=False,
        )
        .eval()
        .half()
    )

    class _Config:
        patch_size = (1, 1, 1)

    wrapped = MiniMaxH3TransformerWrapper(model)
    adapter = _ExportABITransformerAdapter(
        wrapped,
        config=_Config(),
        dtype=torch.float16,
        pad_temb_to_3=True,
    )
    prepare_transformer_block_cpu_offload(adapter.wrapped.transformer, "cpu")
    assert adapter._block_cpu_offload_enabled is True

    # Simulate ComponentsManager onload of the managed transformer component.
    adapter.to("cuda:0")
    body = adapter.wrapped.transformer
    assert next(body.proj_in.parameters()).device.type == "cuda"
    assert all(next(block.parameters()).device.type == "cpu" for block in body.transformer_blocks)
    assert adapter.device.type == "cuda"
    # Footprint should reflect shared modules + one parked block, not the full tree.
    full = sum(param.nelement() * param.element_size() for param in adapter.parameters())
    assert adapter.get_memory_footprint() < full
