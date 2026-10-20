"""Keep the oversized Qwen4 PLE lookup on CPU using SDK node placement.

The floating table or packed SEFP4 initializers remain on CPU. Only the
selected embeddings move to the next graph node's device. Parser adapters
are scoped to loading; the shared SDK source remains unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from xhquant.xhonnxruntime.auto_offload_graph_model import NodeInfo
from xhquant.xhonnxruntime.hmonnx_inference_v2 import (
    AutoOffloadGraphModuleInterpreter,
)
from xhquant.xhonnxruntime.hmonnx_inference_v2 import (
    HMONNXInferenceV2 as SDKInferenceV2,
)

from ...hmonnx.hmonnx_model import HMONNXModel
from .ple_sefp4 import PLE_SEFP4_OP


HOST_PLE_MIN_BYTES = 64 << 30


def _graph_attr(graph, target):
    value = graph
    for name in str(target).split("."):
        value = getattr(value, name)
    return value


def find_host_ple_gathers(graph, *, min_nbytes=HOST_PLE_MIN_BYTES):
    """Identify single-use PLE tables by explicit packed ABI or float layout."""
    result = {}
    for node in graph.graph.nodes:
        if node.op != "call_module":
            continue
        module = graph.get_submodule(str(node.target))
        names = getattr(module, "input_names", ())
        if getattr(module, "op_type", None) == PLE_SEFP4_OP:
            if len(names) != 3 or len(node.args) != 3:
                raise ValueError("Packed Qwen4 PLE requires two weights and ids")
            weight_nodes = node.args[:2]
            if any(not isinstance(item, torch.fx.Node) or item.op != "get_attr" for item in weight_nodes):
                raise ValueError("Packed Qwen4 PLE weights must be initializers")
            packed, exponent = [_graph_attr(graph, item.target) for item in weight_nodes]
            groups = (module.embedding_dim + 63) // 64
            if (
                packed.dtype != torch.int8 or packed.ndim != 3 or tuple(packed.shape[1:]) != (groups, 32)
                or exponent.dtype != torch.int16 or tuple(exponent.shape) != (packed.shape[0], groups, 1)
            ):
                raise ValueError("Invalid packed Qwen4 PLE storage layout")
            # Placement follows the original table's logical size, not its
            # compressed size, which can be below the legacy 64 GiB threshold.
            if packed.shape[0] * module.embedding_dim * 2 < min_nbytes:
                continue
            if any(len(item.users) != 1 for item in weight_nodes):
                raise ValueError("Oversized Qwen4 PLE weights must feed exactly one Gather")
            if packed.device.type != "cpu" or exponent.device.type != "cpu":
                raise ValueError("Oversized Qwen4 PLE must remain CPU-resident before session placement")
            result[node.name] = sum(item.numel() * item.element_size() for item in (packed, exponent))
            continue
        if (
            getattr(module, "op_type", None) != "Gather"
            or int(getattr(module, "axis", -1)) != 0
            or len(names) != 2
            or "ple" not in names[0]
            or not names[0].endswith(("ngram_embedding.weight", "ngram_embedding_weight"))
            or len(node.args) != 2
        ):
            continue
        weight_node = node.args[0]
        if not isinstance(weight_node, torch.fx.Node) or weight_node.op != "get_attr":
            continue
        weight = _graph_attr(graph, weight_node.target)
        if not torch.is_tensor(weight) or weight.ndim != 2 or weight.dtype != torch.float16:
            continue
        nbytes = weight.numel() * weight.element_size()
        if nbytes < min_nbytes:
            continue
        if len(weight_node.users) != 1:
            raise ValueError("Oversized Qwen4 PLE table must feed exactly one Gather")
        if weight.device.type != "cpu":
            raise ValueError("Oversized Qwen4 PLE must remain CPU-resident before session placement")
        result[node.name] = nbytes
    return result


class HostPLEAutoOffloadInterpreter(AutoOffloadGraphModuleInterpreter):
    """Use the SDK's placement algorithm for GPUs, then pin only PLE to CPU."""

    def __init__(self, graph_module, gpu_ids, enable_golden, *, owner, host_ple_nodes):
        self.host_ple_nodes = dict(host_ple_nodes)
        super().__init__(graph_module, gpu_ids, enable_golden, owner=owner)

    def _get_configured_layer_infos(self):
        # Prefill and decode have different graph nodes and padding. Re-infer
        # both using shared tensors' actual devices; never project a CPU Gather
        # placement using unrelated prefill Tag intervals.
        return None

    def _to_cpu(self, tensor):
        if tensor.device.type == "cpu" or not any(
            size > 1 and stride == 0 for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        ):
            return super()._to_cpu(tensor)
        # Golden must transfer the small broadcast source first. Copying the
        # expanded CUDA view directly can materialize a multi-GB GPU buffer.
        shape = tensor.shape
        compact = tensor.detach()
        if type(compact) is not torch.Tensor:
            compact = compact.as_subclass(torch.Tensor)
        for dim, (size, stride) in enumerate(zip(shape, tensor.stride(), strict=True)):
            if size > 1 and stride == 0:
                compact = compact.narrow(dim, 0, 1)
        return compact.cpu().expand(shape).contiguous()

    def _get_node_memory_footprint(self, node, seen_params):
        if node.name in self.host_ple_nodes:
            return 0
        return super()._get_node_memory_footprint(node, seen_params)

    def infer_auto_device_map(self, max_memory=None):
        super().infer_auto_device_map(max_memory)
        for name, nbytes in self.host_ple_nodes.items():
            self.node_device_map[name] = NodeInfo(name, "cpu", nbytes)
        # Retain complete per-node placement, including functional getitems and
        # the output node. This runs before the SDK moves registered buffers.
        groups = []
        for node in self.graph.nodes:
            if node.op in {"placeholder", "get_attr"}:
                continue
            device = self.node_device_map[node.name].device
            if not groups or groups[-1][0] != device:
                groups.append((device, []))
            groups[-1][1].append(node)
        self._layer_infos = [
            self._make_layer_info(index, nodes, device) for index, (device, nodes) in enumerate(groups)
        ]
        self._assign_placeholder_devices_from_consumers()


def enable_qwen4_exp_inplace_kv(session, required_names):
    """Enable and verify each explicit K/V writer, independently of QSA caches."""
    from xhquant.xhonnxruntime.parsers.llm_cache import LLMCache

    writers = {}
    for module in session.graph_module.modules():
        if not isinstance(module, LLMCache) or len(module.input_names) < 4:
            continue
        name = module.input_names[3]
        if name not in required_names:
            continue
        if name in writers or module.only_handle_old_cache or module.attention_max_length >= 0:
            raise ValueError(f"Invalid Qwen4 explicit K/V cache writer: {name}")
        writers[name] = module
    if set(writers) != set(required_names):
        raise ValueError("Qwen4 explicit cache graph lacks paired K/V LLMCache writers")
    session.enable_inplace_cache_updates()
    if not all(module.reuse_output_storage for module in writers.values()):
        raise ValueError("Qwen4 explicit K/V writers did not enable in-place storage")


class Qwen4ExpHostPLEInferenceV2(SDKInferenceV2):
    HOST_PLE_MIN_BYTES = HOST_PLE_MIN_BYTES

    def _create_interpreter(self, graph_module):
        host_nodes = find_host_ple_gathers(graph_module, min_nbytes=self.HOST_PLE_MIN_BYTES)
        if not host_nodes and not self._session_config.enable_auto_offload:
            return super()._create_interpreter(graph_module)
        devices = self._get_configured_exec_devices()
        if all(not isinstance(device, int) and torch.device(device).type == "cpu" for device in devices):
            return super()._create_interpreter(graph_module)
        if not self._session_config.enable_auto_offload:
            raise ValueError("Oversized Qwen4 PLE requires auto-offload with host Gather placement")
        if self._session_config.enable_cuda_graph:
            raise ValueError("Host Qwen4 PLE lookup requires CUDA Graph to be disabled")
        self.host_ple_nodes = dict(host_nodes)
        return HostPLEAutoOffloadInterpreter(
            graph_module,
            list(self._get_auto_offload_max_memory()),
            self._session_config.enable_golden,
            owner=self,
            host_ple_nodes=host_nodes,
        )


class _Qwen4ExpSessionPolicy:
    """Reuse the selected backend while scoping Qwen4's loading adapters."""

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def create_session(self, backend, path, **options):
        from .xhquant_extensions import qwen4_exp_xhquant_extension_scope

        with qwen4_exp_xhquant_extension_scope():
            return self.base.create_session(backend, path, **options)


def qwen4_exp_execution_backend(backend):
    """Bind Host PLE sessions to this model without changing shared factories."""
    if backend.name != "hmonnx":
        raise NotImplementedError("Qwen4 host PLE placement requires the SDK HMONNX backend")
    if isinstance(backend.policy, _Qwen4ExpSessionPolicy):
        return backend
    classes = dict(backend.classes)
    if backend.use_inference_v2:
        classes["HMONNXInferenceV2"] = (
            Qwen4ExpHostPLEInferenceV2.__module__, Qwen4ExpHostPLEInferenceV2.__name__,
        )
    return replace(backend, classes=classes, policy=_Qwen4ExpSessionPolicy(backend.policy))


class Qwen4ExpServingGraphModel(HMONNXModel):
    def _create_hmonnx_session(self, enable_golden, enable_auto_offload):
        from .xhquant_extensions import qwen4_exp_xhquant_extension_scope

        backend = self._backend
        self._backend = qwen4_exp_execution_backend(backend)
        try:
            with qwen4_exp_xhquant_extension_scope():
                return super()._create_hmonnx_session(enable_golden, enable_auto_offload)
        finally:
            self._backend = backend
