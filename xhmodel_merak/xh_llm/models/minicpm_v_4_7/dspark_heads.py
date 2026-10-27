# Copyright (c) 2026 DeepSpec contributors
# SPDX-License-Identifier: MIT
"""Vanilla Markov and confidence heads used only by MiniCPM DSpark."""

from __future__ import annotations

import torch
import torch.nn as nn

from xhquant import nn as xhnn


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        del hidden
        return self.markov_w2(self.markov_w1(token_ids))


class DSparkConfidenceHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, markov: torch.Tensor | None = None) -> torch.Tensor:
        features = hidden if markov is None else torch.cat((hidden, markov), dim=-1)
        return self.proj(features).squeeze(-1)


def greedy_dspark_proposals(
    logits, hidden, anchor_token_id, markov, confidence, with_markov, steps=None, argmax=None, raw_scores=False,
):
    if argmax is None:
        argmax = xhnn.ArgMax(axis=-1, keepdim=False)
    if steps is None:
        steps = logits.shape[1]
    proposals = []
    scores = []
    previous = anchor_token_id.reshape(-1)
    for step in range(int(steps)):
        adjusted = logits[:, step] + markov(previous, hidden[:, step])
        token = argmax(adjusted).reshape(-1)
        markov_hidden = markov.markov_w1(token) if with_markov else None
        score = confidence(hidden[:, step], markov_hidden)
        proposals.append(token)
        scores.append(score)
        previous = token
    tokens = torch.stack(proposals, dim=1)
    stacked = torch.stack(scores, dim=1)
    return tokens, stacked if raw_scores else torch.sigmoid(stacked)
