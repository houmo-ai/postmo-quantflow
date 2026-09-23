"""Keep freshly constructed MOSS test models independent of earlier loaders."""

import pytest
import torch
from transformers import modeling_utils


# Pytest imports test modules before running test bodies. Capture the normal
# constructors here; a preceding GPTQModel load may leave these as no-ops.
_TORCH_INITIALIZERS = {
    name: getattr(torch.nn.init, name)
    for name in ("kaiming_uniform_", "uniform_", "normal_")
}
_HAS_HF_INIT_WEIGHTS = hasattr(modeling_utils, "_init_weights")
_HF_INIT_WEIGHTS = getattr(modeling_utils, "_init_weights", None)


@pytest.fixture
def initialized_moss_test_models(monkeypatch):
    for name, initializer in _TORCH_INITIALIZERS.items():
        monkeypatch.setattr(torch.nn.init, name, initializer)
    if _HAS_HF_INIT_WEIGHTS:
        monkeypatch.setattr(modeling_utils, "_init_weights", _HF_INIT_WEIGHTS)
    else:
        monkeypatch.delattr(modeling_utils, "_init_weights", raising=False)
