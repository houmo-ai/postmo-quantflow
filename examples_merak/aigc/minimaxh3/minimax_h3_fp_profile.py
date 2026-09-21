#!/usr/bin/env python3
"""Static module-level FLOPs profile for MiniMax-H3 FL2VA.

The script uses checkpoint dimensions and does not load model weights. FLOPs
count one multiply-add as two operations and include linear, convolution and
attention QK/AV matmuls. Elementwise activations, norms and data movement are
omitted.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


TFLOP = 1e12
PFLOP = 1e15


def linear_flops(tokens: int, in_dim: int, out_dim: int) -> int:
    return 2 * tokens * in_dim * out_dim


def attention_flops(tokens: int, inner_dim: int) -> int:
    return 4 * tokens * tokens * inner_dim


def swiglu_flops(tokens: int, hidden_size: int, ffn_dim: int) -> int:
    return 6 * tokens * hidden_size * ffn_dim


def conv3d_flops(t: int, h: int, w: int, in_channels: int, out_channels: int, kernel: int) -> int:
    return 2 * t * h * w * in_channels * out_channels * kernel**3


def conv1d_flops(length: int, in_channels: int, out_channels: int, kernel: int, groups: int = 1) -> int:
    return 2 * length * out_channels * kernel * (in_channels // groups)


def geglu_flops(tokens: int, hidden_size: int, ffn_dim: int) -> int:
    return 2 * linear_flops(tokens, hidden_size, ffn_dim) + linear_flops(tokens, ffn_dim, hidden_size)


def fmt_flops(value: int) -> str:
    if value >= PFLOP:
        return f"{value / PFLOP:.3f} PFLOPs"
    return f"{value / TFLOP:.3f} TFLOPs"


@dataclass(frozen=True)
class Shape:
    width: int = 1344
    height: int = 768
    frames: int = 124
    fps: int = 24
    text_tokens: int = 1048
    steps: int = 4

    @property
    def latent_frames(self) -> int:
        if self.frames % 17 != 5:
            raise ValueError("MiniMax-H3 frames must have the form 17 * n + 5")
        return (self.frames - 5) // 17 * 5 + 2

    @property
    def latent_height(self) -> int:
        return self.height // 16

    @property
    def latent_width(self) -> int:
        return self.width // 16

    @property
    def rows_per_video_frame(self) -> int:
        return self.latent_height // 2 * (self.latent_width // 2)

    @property
    def target_video_rows(self) -> int:
        return self.latent_frames * self.rows_per_video_frame

    @property
    def condition_video_rows(self) -> int:
        return self.rows_per_video_frame

    @property
    def audio_rows(self) -> int:
        return 2 * round(self.frames / self.fps * 40)

    @property
    def audio_latent_frames(self) -> int:
        return round(self.frames / self.fps * 40)

    @property
    def audio_samples(self) -> int:
        return self.audio_latent_frames * 800

    @property
    def packed_rows(self) -> int:
        return self.text_tokens + self.condition_video_rows + self.audio_rows + self.target_video_rows


def transformer_flops(shape: Shape) -> dict[str, int]:
    hidden_size = 5376
    attention_inner = 56 * 128
    ffn_dim = 14336
    video_rows = shape.condition_video_rows + shape.target_video_rows
    sequence = shape.packed_rows

    input_projection = (
        linear_flops(video_rows, 24 * 4, hidden_size)
        + linear_flops(shape.audio_rows, 32, hidden_size)
        + linear_flops(shape.text_tokens, 5120, hidden_size)
    )
    refiner_layer = (
        3 * linear_flops(shape.text_tokens, hidden_size, attention_inner)
        + linear_flops(shape.text_tokens, attention_inner, hidden_size)
        + attention_flops(shape.text_tokens, attention_inner)
        + swiglu_flops(shape.text_tokens, hidden_size, ffn_dim)
    )
    adaln = linear_flops(3, 2688, 6 * hidden_size)
    block = (
        3 * linear_flops(sequence, hidden_size, attention_inner)
        + linear_flops(sequence, attention_inner, hidden_size)
        + attention_flops(sequence, attention_inner)
        + swiglu_flops(sequence, hidden_size, ffn_dim)
        + adaln
    )
    output_heads = linear_flops(sequence, hidden_size, 24 * 4) + linear_flops(sequence, hidden_size, 32)
    total = input_projection + 2 * refiner_layer + 50 * block + output_heads
    return {
        "input projections": input_projection,
        "text token refiner (2 layers)": 2 * refiner_layer,
        "one packed block": block,
        "packed block stack (50 layers)": 50 * block,
        "video and audio output heads": output_heads,
        "single transformer forward": total,
    }


def qwen3_vl_flops(shape: Shape, raw_vision_tokens: int = 4128) -> dict[str, int]:
    text_hidden = 5120
    query_inner = 64 * 128
    kv_inner = 8 * 128
    text_ffn = 25600
    text_layer = (
        linear_flops(shape.text_tokens, text_hidden, query_inner)
        + 2 * linear_flops(shape.text_tokens, text_hidden, kv_inner)
        + linear_flops(shape.text_tokens, query_inner, text_hidden)
        + attention_flops(shape.text_tokens, query_inner)
        + swiglu_flops(shape.text_tokens, text_hidden, text_ffn)
    )

    vision_hidden = 1152
    vision_ffn = 4304
    vision_layer = (
        4 * linear_flops(raw_vision_tokens, vision_hidden, vision_hidden)
        + attention_flops(raw_vision_tokens, vision_hidden)
        + 2 * linear_flops(raw_vision_tokens, vision_hidden, vision_ffn)
    )
    return {
        "Qwen3-VL vision tower (27 layers)": 27 * vision_layer,
        "Qwen3-VL text tower through layer 50": 50 * text_layer,
        "Qwen3-VL conditioner total": 27 * vision_layer + 50 * text_layer,
    }


def split_tile_count(length: int, tile_size: int = 256, min_overlap: int = 64) -> int:
    if tile_size >= length:
        return 1
    count = math.ceil(length / tile_size)
    while tile_size * count - min_overlap * (count - 1) < length:
        count += 1
    return count


def vae_encoder_tile_flops(tile_size: int = 256) -> int:
    t = 1
    h = w = tile_size
    total = conv3d_flops(t, h, w, 3, 128, 3)
    channels = (128, 256, 256, 512, 512, 1024)
    spatial_factors = (2, 2, 2, 2, 1, 1)
    in_channels = 128
    for out_channels, factor in zip(channels, spatial_factors, strict=True):
        for layer in range(2):
            block_in = in_channels if layer == 0 else out_channels
            total += conv3d_flops(t, h, w, block_in, out_channels, 3)
            total += conv3d_flops(t, h, w, out_channels, out_channels, 3)
            if block_in != out_channels:
                total += conv3d_flops(t, h, w, block_in, out_channels, 1)
        in_channels = out_channels
        if factor > 1:
            h //= factor
            w //= factor
            total += conv3d_flops(t, h, w, out_channels, out_channels, 3)
    total += conv3d_flops(t, h, w, 1024, 48, 3)
    total += conv3d_flops(t, h, w, 48, 48, 1)
    return total


def vae_decoder_call_flops(latent_frames: int, latent_height: int, latent_width: int) -> int:
    tokens = latent_frames * latent_height * latent_width + 5
    hidden_size = 2048
    ffn_dim = 8192
    layer = (
        4 * linear_flops(tokens, hidden_size, hidden_size)
        + attention_flops(tokens, hidden_size)
        + swiglu_flops(tokens, hidden_size, ffn_dim)
    )
    return (
        linear_flops(tokens - 5, 24, hidden_size)
        + 36 * layer
        + linear_flops(tokens, hidden_size, 3 * 4 * 16 * 16)
    )


def visual_vae_flops(shape: Shape) -> dict[str, int]:
    spatial_tiles = split_tile_count(shape.height) * split_tile_count(shape.width)
    encode = spatial_tiles * vae_encoder_tile_flops()

    num_tokens = shape.latent_frames + 3
    padded_tokens = num_tokens + (-num_tokens) % 5
    temporal_calls = padded_tokens // 5 - 1
    tiled_call = vae_decoder_call_flops(7, 16, 16)
    tiled_decode = temporal_calls * spatial_tiles * tiled_call
    full_decode = vae_decoder_call_flops(shape.latent_frames, shape.latent_height, shape.latent_width)
    return {
        f"Visual VAE encode one keyframe ({spatial_tiles} spatial tiles)": encode,
        f"Visual VAE decode released tiling ({temporal_calls}x{spatial_tiles} calls)": tiled_decode,
        "Visual VAE decode without tiling (upper bound)": full_decode,
    }


def _audio_activation_flops(length: int, channels: int) -> int:
    upsampled = length * 2
    return conv1d_flops(upsampled, channels, channels, 12, groups=channels) + conv1d_flops(
        length, channels, channels, 12, groups=channels
    )


def _audio_residual_unit_flops(length: int, dim: int) -> int:
    return conv1d_flops(length, dim, dim, 7) + conv1d_flops(length, dim, dim, 1)


def _audio_amp_block_flops(length: int, channels: int, kernel: int, dilations: tuple[int, ...] = (1, 3, 5)) -> int:
    total = 0
    for _ in dilations:
        total += _audio_activation_flops(length, channels)
        total += conv1d_flops(length, channels, channels, kernel)
        total += _audio_activation_flops(length, channels)
        total += conv1d_flops(length, channels, channels, kernel)
    return total


def audio_vae_flops(shape: Shape) -> dict[str, int]:
    samples = shape.audio_samples
    latent_frames = shape.audio_latent_frames
    stereo = 2

    encode = conv1d_flops(samples, 1, 64, 7)
    length = samples
    channels = 64
    for stride in (2, 4, 4, 5, 5):
        residual_dim = channels
        encode += 3 * _audio_residual_unit_flops(length, residual_dim)
        length //= stride
        encode += conv1d_flops(length, residual_dim, residual_dim * 2, 2 * stride)
        channels = residual_dim * 2
    encode += conv1d_flops(length, channels, 2048, 3)
    encode += (
        linear_flops(latent_frames, 2048, 32)
        + linear_flops(latent_frames, 2048, 2048 * 3)
        + attention_flops(latent_frames, 2048)
        + linear_flops(latent_frames, 32, 32)
        + geglu_flops(latent_frames, 32, 64)
        + 2 * conv1d_flops(latent_frames, 32, 32, 1)
    )

    decode = conv1d_flops(latent_frames, 32, 2048, 1)
    decode += conv1d_flops(latent_frames, 2048, 1024, 7)
    length = latent_frames
    in_channels = 1024
    for rate, kernel in zip((5, 5, 2, 2, 2, 2, 2), (9, 9, 4, 4, 4, 4, 4), strict=True):
        out_channels = in_channels // 2
        length *= rate
        decode += conv1d_flops(length, in_channels, out_channels, kernel)
        for res_kernel in (3, 7, 11):
            decode += _audio_amp_block_flops(length, out_channels, res_kernel)
        in_channels = out_channels
    decode += _audio_activation_flops(length, in_channels)
    decode += conv1d_flops(length, in_channels, 1, 7)

    return {
        f"Audio VAE encode stereo waveform ({stereo} x {samples} samples)": stereo * encode,
        f"Audio VAE decode stereo latents ({stereo} x {latent_frames} frames)": stereo * decode,
    }


def print_table(rows: dict[str, int]) -> None:
    print("| Module | FLOPs |")
    print("| --- | ---: |")
    for name, value in rows.items():
        print(f"| {name} | {fmt_flops(value)} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--text-tokens", type=int, default=1048)
    parser.add_argument("--raw-vision-tokens", type=int, default=4128)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    shape = Shape(args.width, args.height, args.frames, text_tokens=args.text_tokens, steps=args.steps)

    transformer = transformer_flops(shape)
    conditioner = qwen3_vl_flops(shape, args.raw_vision_tokens)
    vae = visual_vae_flops(shape)
    audio = audio_vae_flops(shape)
    total = (
        conditioner["Qwen3-VL conditioner total"]
        + vae[next(name for name in vae if name.startswith("Visual VAE encode"))]
        + args.steps * transformer["single transformer forward"]
        + vae[next(name for name in vae if name.startswith("Visual VAE decode released"))]
        + audio[next(name for name in audio if name.startswith("Audio VAE decode"))]
    )

    print(f"## MiniMax-H3 FL2VA {shape.width}x{shape.height}, {shape.frames} frames")
    print()
    print(f"- Latent: [24,{shape.latent_frames},{shape.latent_height},{shape.latent_width}]")
    print(f"- Packed sequence: {shape.packed_rows} rows = text {shape.text_tokens} + keyframe {shape.condition_video_rows} + audio {shape.audio_rows} + target video {shape.target_video_rows}")
    print("- One denoising step uses one transformer forward; the checkpoint has no CFG branch.")
    print(f"- Audio latent: {shape.audio_latent_frames} frames/channel at 40 Hz, stereo packed as {shape.audio_rows} rows.")
    print()
    print_table(conditioner)
    print()
    print_table(transformer)
    print()
    print_table(vae)
    print()
    print_table(audio)
    print()
    print(f"Estimated end-to-end total ({args.steps} steps, released VAE tiling): {fmt_flops(total)}")


if __name__ == "__main__":
    main()