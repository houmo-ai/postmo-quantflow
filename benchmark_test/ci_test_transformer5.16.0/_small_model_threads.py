# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Bound CPU thread overhead for tiny real-model regression modules."""

import pytest


@pytest.fixture(scope="module")
def small_model_cpu_threads():
    import torch

    previous = torch.get_num_threads()
    # Tiny GEMMs and reductions do less work than a large host thread pool's
    # scheduling overhead, especially inside a CPU-limited GPU CI container.
    torch.set_num_threads(min(previous, 4))
    try:
        yield
    finally:
        torch.set_num_threads(previous)
