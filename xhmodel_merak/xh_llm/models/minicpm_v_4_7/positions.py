"""Host-side canvas MRoPE, shared by autoregressive and speculative runtimes."""

from __future__ import annotations

import torch

from xhmodel_merak.xh_llm.models.qwen3_5.data_preprocess import Qwen3_5_DataPreprocess


def canvas_positions(config, model_inputs: dict, *, downsample_mode: str):
    ids = model_inputs["input_ids"]
    if getattr(config, "mrope_mode", None) != "canvas":
        return None, None
    from .auto_classes import get_minicpmv47_modeling_module

    MiniCPMV4_7Model = get_minicpmv47_modeling_module().MiniCPMV4_7Model

    # get_rope_index only uses the model config and its position utilities.
    # This host-only shell allocates no vision, decoder, or embedding weights.
    owner = MiniCPMV4_7Model.__new__(MiniCPMV4_7Model)
    torch.nn.Module.__init__(owner)
    owner.config = config
    types = model_inputs.get("mm_token_type_ids")
    if types is None:
        types = torch.zeros_like(ids)
        types[ids == config.image_token_id] = 1
        types[ids == config.video_token_id] = 2
    return owner.get_rope_index(
        ids,
        attention_mask=model_inputs.get("attention_mask"),
        target_sizes=model_inputs.get("target_sizes"),
        target_sizes_videos=model_inputs.get("target_sizes_videos"),
        mm_token_type_ids=types,
        downsample_mode=downsample_mode,
    )


def position_chunk(position_ids, rope_deltas, *, offset: int, length: int):
    if offset >= position_ids.shape[-1]:
        return torch.arange(offset, offset + length, device=position_ids.device).view(1, 1, -1).expand(
            3, position_ids.shape[1], -1
        ) + rope_deltas.unsqueeze(0)
    chunk = position_ids[..., offset : offset + length]
    if chunk.shape[-1] < length:
        padding = torch.zeros((*chunk.shape[:-1], length - chunk.shape[-1]), dtype=chunk.dtype, device=chunk.device)
        chunk = torch.cat((chunk, padding), dim=-1)
    return chunk


class MiniCPMV47DataPreprocess(Qwen3_5_DataPreprocess):
    def forward(self, data):
        outputs = list(super().forward(data))
        positions = data.get("position_ids")
        if positions is not None:
            if positions.ndim != 3 or positions.shape[:2] != (3, 1):
                raise ValueError("MiniCPM position_ids must have shape [3, 1, sequence]")
            positions = positions.to(device=outputs[0].device, dtype=torch.int64)
            if positions.shape[-1] != self.input_sequence_length:
                raise ValueError("MiniCPM position_ids must match the padded graph input length")
            outputs[1:4] = [positions[axis, 0] for axis in range(3)]
        return tuple(outputs)
