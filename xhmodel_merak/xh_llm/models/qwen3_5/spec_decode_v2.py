"""V2 sessions for the existing MTP/DFlash state machine.

All graphs are loaded together: graph precision and shared initializer identity
must survive target/draft session construction. No legacy session fallback.
"""
from __future__ import annotations

from pathlib import Path

import torch

from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2
from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader

from .qwen3_5_spec_decode_onnx_model import Qwen3_5SpecDecodeONNXModel


def _graph_path(config) -> str:
    return str(config["onnx"] if isinstance(config, dict) else config.onnx)


class Qwen3_5SpecDecodeV2Model(Qwen3_5SpecDecodeONNXModel):  # noqa: N801
    """Reuse speculative scheduling/ABI with shared, packed HMONNX V2 graphs."""

    def __init__(self, *, prefill, decode, draft, session_device, **kwargs):
        device = torch.device(session_device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self._session_device = device
        if kwargs.get("resource_tight_mode", False):
            raise ValueError("V2 shared sessions do not support resource_tight_mode session unloading")
        for name in (
            "auto_offload_max_memory", "prefill_auto_offload_max_memory", "decode_auto_offload_max_memory",
        ):
            if kwargs.get(name) is not None:
                raise ValueError(f"{name} is a legacy runtime option; V2 derives placement from session_device")
        if kwargs.get("enable_cuda_graph", False):
            for name, default in (
                ("cuda_graph_warmup_runs", 3), ("cuda_graph_graph_warmup_runs", 6),
                ("cuda_graph_clone_outputs", True),
            ):
                if kwargs.get(name, default) != default:
                    raise ValueError(f"V2 requires the SDK default {name}={default!r}")

        files = {"prefill": _graph_path(prefill), "decode": _graph_path(decode)}
        draft_configs = draft if isinstance(draft, dict) else {"decode": draft}
        for name in ("prefill", "context", "context_decode", "decode"):
            if draft_configs.get(name) is not None:
                files[f"draft_{name}"] = _graph_path(draft_configs[name])
        loader = MultiHMONNXLoader(files)
        self._v2_graphs = loader.graphs
        self._v2_target_layers = None
        self.runtime_summary = {
            "backend": "HMONNXInferenceV2",
            "device": str(device),
            "initializer_pool": dict(loader.initializer_pool.summary()),
        }
        try:
            super().__init__(prefill=prefill, decode=decode, draft=draft, **kwargs)
        finally:
            # Sessions now own the tensors; do not retain another graph pool.
            self._v2_graphs.clear()

    def _create_hmonnx_session(self, onnx_path: str, session_name: str):
        config = HMONNXInferenceConfig(
            enable_cuda_graph=self._should_enable_cuda_graph(session_name),
            enable_auto_offload=self.auto_offload and self._session_device.type == "cuda",
            exec_devices=[self._session_device],
            layers=self._v2_target_layers if session_name == "decode" else None,
        )
        session = HMONNXInferenceV2.from_onnx_graph(onnx_path, self._v2_graphs.pop(session_name), config)
        if session_name == "prefill":
            self._v2_target_layers = session.get_layer_infos()
        self._set_hmonnx_session_golden_dir(session, Path(onnx_path).parent)
        return session

    def _apply_auto_offload(self, session, max_memory=None):
        # Placement is applied during V2 construction, including reuse of the
        # target prefill layer map by verify. Never wrap it in the legacy mapper.
        if not isinstance(session, HMONNXInferenceV2):
            raise TypeError("Speculative V2 runtime requires HMONNXInferenceV2 sessions")

    def _check_device(self, device):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device != self._session_device:
            raise ValueError(f"V2 sessions are fixed on {self._session_device}; cannot move to {device}")
        return device

    def _set_exec_device(self, device):
        self._exec_device = self._check_device(device)

    def _set_device(self, device):
        self._device = self._check_device(device)
        if self.token_embedding is not None:
            self.token_embedding.to(self._device)
        return self

    def _get_session_cuda_graph_status(self, session):
        if not isinstance(session, HMONNXInferenceV2):
            return super()._get_session_cuda_graph_status(session)
        interpreter = session.interpreter
        return {
            "enabled": session._session_config.enable_cuda_graph,
            "captured": bool(getattr(interpreter, "has_captured_graph", False)),
            "reason": getattr(interpreter, "capture_unavailable_reason", None),
            "backend": "HMONNXInferenceV2",
        }


__all__ = ["Qwen3_5SpecDecodeV2Model"]
