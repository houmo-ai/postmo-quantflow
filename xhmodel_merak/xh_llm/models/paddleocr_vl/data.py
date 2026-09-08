# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

import torch

from ...llm_data_processor import BaseLLMInputProcessor


class PaddleOCRLLMInputProcessor(BaseLLMInputProcessor):
    def forward(self, data):
        base = super().forward(data)
        embeddings, past_length, current_length, keys, values = base
        positions = data.get("position_ids")
        if positions is None:
            positions = torch.arange(embeddings.shape[1], device=self._device) + past_length[0]
            positions = positions.expand(3, -1)
        if positions.ndim == 3:
            positions = positions[:, 0]
        if positions.shape[0] != 3 or positions.shape[1] > embeddings.shape[1]:
            raise ValueError("position_ids must have shape [3, sequence_length]")
        positions = torch.nn.functional.pad(positions, (0, embeddings.shape[1] - positions.shape[1]))
        return (embeddings, *positions.to(torch.int64).unbind(0), past_length, current_length, keys, values)
