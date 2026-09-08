from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class Runner(Protocol):
    input_names: list[str]

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


class OrtRunner:
    def __init__(self, model: str | Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [value.name for value in self.session.get_inputs()]
        self.output_names = [value.name for value in self.session.get_outputs()]

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = self.session.run(None, {name: feed[name] for name in self.input_names})
        return dict(zip(self.output_names, outputs, strict=True))


class HmonnxRunner:
    def __init__(self, model: str | Path, device: str = "cuda:0") -> None:
        import onnx

        from xhquant.api import HMONNXInference

        proto = onnx.load(str(model), load_external_data=False)
        self.input_names = [value.name for value in proto.graph.input]
        self.output_names = [value.name for value in proto.graph.output]
        self.device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.session = HMONNXInference(str(model))
        self.session.to(self.device)

    @torch.no_grad()
    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        inputs = [_to_hmonnx_tensor(feed[name]).to(self.device) for name in self.input_names]
        outputs = self.session(*inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        return {
            name: value.detach().float().cpu().numpy() for name, value in zip(self.output_names, outputs, strict=True)
        }


def _to_hmonnx_tensor(value: np.ndarray) -> torch.Tensor:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.integer):
        return torch.from_numpy(array.astype(np.int32, copy=False))
    return torch.from_numpy(array.astype(np.float16, copy=False))
