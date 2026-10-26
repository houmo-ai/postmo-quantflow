"""A model-owned backend selection, independent of later environment changes."""

from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class ExecutionBackend:
    name: str
    classes: Mapping[str, tuple[str, str]]
    policy: Any
    use_inference_v2: bool

    def __post_init__(self):
        object.__setattr__(self, "classes", MappingProxyType(dict(self.classes)))

    @property
    def supports_cuda_graph(self) -> bool:
        return self.policy.SUPPORTS_CUDA_GRAPH

    @property
    def supports_auto_offload(self) -> bool:
        return self.policy.SUPPORTS_AUTO_OFFLOAD

    def resolve_class(self, name: str):
        try:
            module_name, class_name = self.classes[name]
        except KeyError:
            raise ValueError(f"Backend {self.name!r} does not provide interface {name!r}") from None
        return getattr(import_module(module_name), class_name)

    def prepare_graphs(self, prefill, decode, *, prepare, **options):
        return self.policy.prepare_graphs(prefill, decode, prepare=prepare, **options)

    def create_session(self, path, **options):
        return self.policy.create_session(self, path, **options)


def create_v2_session(
    backend, path, *, onnx_graph, enable_cuda_graph, enable_auto_offload, enable_golden, device_map, layer_infos
):
    config = backend.resolve_class("HMONNXInferenceConfig")()
    config.enable_cuda_graph = bool(enable_cuda_graph and backend.supports_cuda_graph)
    config.enable_auto_offload = bool(enable_auto_offload and backend.supports_auto_offload)
    config.exec_devices = device_map
    config.enable_golden = enable_golden
    config.layers = layer_infos
    implementation = backend.resolve_class("HMONNXInferenceV2")
    if onnx_graph is None:
        session = implementation(path, config)
    else:
        session = implementation.from_onnx_graph(path, onnx_graph, config)
    return session, session.device
