from .audio_vae_decoder_wrapper import (
    MiniMaxH3AudioVAEDecoderWrapper,
    wrap_audio_vae_decoder,
)
from .audio_vae_encoder_wrapper import (
    MiniMaxH3AudioVAEEncoderWrapper,
    wrap_audio_vae_encoder,
)
from .lora import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_FILENAME,
    DEFAULT_LORA_SCALE,
    default_lora_path,
    load_minimax_h3_lora,
)
from .model import XHMiniMaxH3Model
from .text_conditioning import (
    DEFAULT_TEXT_SEQUENCE_LENGTH,
    build_encoder_attention_mask,
    install_fixed_length_text_conditioning,
    normalize_fixed_length_text_conditioning,
)
from .text_encoder_wrapper import (
    DEFAULT_INPUT_SEQUENCE_LENGTH,
    DEFAULT_TEXT_ENCODER_LAYER,
    MiniMaxH3TextEncoderWrapper,
    build_minimax_h3_attention_bias,
    build_minimax_h3_position_ids,
    build_minimax_h3_text_graph_inputs,
    build_minimax_h3_text_inputs,
    wrap_text_encoder,
)
from .transformer_wrapper import (
    MiniMaxH3TransformerWrapper,
    build_minimax_h3_transformer_conditioning,
    build_pack_gather_indices,
    build_packed_attention_mask,
    wrap_transformer,
)
from .vae_decoder_wrapper import MiniMaxH3VAEDecoderWrapper, wrap_vae_decoder
from .vae_encoder_wrapper import MiniMaxH3VAEEncoderTileWrapper, MiniMaxH3VAEEncoderWrapper, wrap_vae_encoder
from .vision_encoder_wrapper import (
    DEFAULT_VISION_MAX_PATCHES,
    MiniMaxH3VisionEncoderWrapper,
    build_vision_attention_mask,
    build_vision_encoder_graph_inputs,
    build_vision_geometry_inputs,
    pad_vision_pixel_values,
    wrap_vision_encoder,
)


__all__ = [
    "DEFAULT_INPUT_SEQUENCE_LENGTH",
    "DEFAULT_TEXT_ENCODER_LAYER",
    "DEFAULT_TEXT_SEQUENCE_LENGTH",
    "DEFAULT_LORA_ALPHA",
    "DEFAULT_LORA_FILENAME",
    "DEFAULT_LORA_SCALE",
    "DEFAULT_VISION_MAX_PATCHES",
    "MiniMaxH3AudioVAEDecoderWrapper",
    "MiniMaxH3AudioVAEEncoderWrapper",
    "MiniMaxH3TextEncoderWrapper",
    "MiniMaxH3TransformerWrapper",
    "MiniMaxH3VAEDecoderWrapper",
    "MiniMaxH3VAEEncoderWrapper",
    "MiniMaxH3VAEEncoderTileWrapper",
    "MiniMaxH3VisionEncoderWrapper",
    "XHMiniMaxH3Model",
    "build_encoder_attention_mask",
    "build_minimax_h3_attention_bias",
    "build_minimax_h3_position_ids",
    "build_minimax_h3_text_graph_inputs",
    "build_minimax_h3_text_inputs",
    "build_minimax_h3_transformer_conditioning",
    "build_pack_gather_indices",
    "build_packed_attention_mask",
    "build_vision_attention_mask",
    "build_vision_encoder_graph_inputs",
    "build_vision_geometry_inputs",
    "default_lora_path",
    "install_fixed_length_text_conditioning",
    "load_minimax_h3_lora",
    "normalize_fixed_length_text_conditioning",
    "pad_vision_pixel_values",
    "wrap_audio_vae_decoder",
    "wrap_audio_vae_encoder",
    "wrap_text_encoder",
    "wrap_transformer",
    "wrap_vae_decoder",
    "wrap_vae_encoder",
    "wrap_vision_encoder",
]
