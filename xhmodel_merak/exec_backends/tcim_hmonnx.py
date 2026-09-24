from __future__ import annotations

import os
import re
from collections import namedtuple
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from .path_utils import resolve_tcim_model_path

TensorInfo = namedtuple("TensorInfo", ["name", "shape", "dtype"])

ENV_TCIM_BACKEND_NAME = "XHMODEL_TCIM_BACKEND_NAME"
_CACHE_TENSOR_NAMES = {"CacheTensor", "HybridCacheTensor"}
_SHARED_CONTEXTS: dict[tuple[int, str], Any] = {}
_CONV_CACHE_OUTPUT = re.compile(r"^conv_cache_out_(?:(q|k|v)_)?(\d+)(?:_\d+)?$")
_RECURRENT_CACHE_OUTPUT = re.compile(r"^recurrent_state_out_(\d+)(?:_\d+)?$")
_PRESENT_KV_OUTPUT = re.compile(r"^present_(key|value)_cache_?(\d*)$")

_NUMPY_TO_TORCH_DTYPE = {
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.uint16): torch.uint16,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.uint32): torch.uint32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.bool_): torch.bool,
}


def _load_tcim_runtime():
    try:
        import tcim_lite as tcim
    except ImportError as exc:
        raise ImportError("tcim_lite is required when XHMODEL_INFERENCE_BACKEND=tcim") from exc
    return tcim.runtime


def _numpy_to_torch_dtype(dtype) -> torch.dtype:
    np_dtype = np.dtype(dtype)
    mapped = _NUMPY_TO_TORCH_DTYPE.get(np_dtype)
    if mapped is None:
        raise TypeError(f"Unsupported TCIM dtype {dtype!r}")
    return mapped


def _is_torch_tensor(value: Any) -> bool:
    return torch.is_tensor(value)


def _is_cache_tensor(value: Any) -> bool:
    return isinstance(value, TcimCacheTensor) or value.__class__.__name__ in _CACHE_TENSOR_NAMES or (
        _is_torch_tensor(value) and bool(getattr(value, "is_custom_tensor", False))
    )


def _align_array_to_info(array: np.ndarray, info: TensorInfo) -> np.ndarray:
    expected_shape = tuple(int(dim) for dim in info.shape)
    expected_dtype = _torch_dtype_to_numpy(info.dtype)
    if array.dtype != expected_dtype:
        if expected_dtype in {np.dtype(np.int16), np.dtype(np.uint16)} and array.dtype == np.dtype(np.float16):
            array = np.ascontiguousarray(array).view(expected_dtype)
        elif expected_dtype == np.dtype(np.float16) and array.dtype in {
            np.dtype(np.int16),
            np.dtype(np.uint16),
        }:
            array = np.ascontiguousarray(array).view(np.float16)
        else:
            raise TypeError(f"input {info.name} expects dtype {expected_dtype}, got {array.dtype}")
    array = np.ascontiguousarray(array)
    if array.shape != expected_shape:
        raise ValueError(f"input {info.name} expects shape {expected_shape}, got {array.shape}")
    return array


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if value.__class__.__name__ in _CACHE_TENSOR_NAMES:
        value = getattr(value, "data", value)
    if _is_torch_tensor(value):
        return np.ascontiguousarray(value.detach().cpu().numpy())
    return np.ascontiguousarray(np.asarray(value))


def _parse_device_id(device: torch.device) -> int:
    if device.type == "cpu":
        return 0
    index = device.index
    return 0 if index is None else int(index)


def _first_exec_device(exec_devices: Any) -> torch.device | None:
    if exec_devices is None:
        return None
    if isinstance(exec_devices, (str, int, torch.device)):
        candidate = exec_devices
    else:
        values = list(exec_devices)
        if not values:
            return None
        candidate = values[0]
    if isinstance(candidate, int):
        return torch.device("cuda", candidate)
    text = str(candidate).strip()
    if text.isdigit():
        return torch.device("cuda", int(text))
    return torch.device(candidate)


def _shared_context_key(device_id: int) -> tuple[int, str]:
    return device_id, os.getenv(ENV_TCIM_BACKEND_NAME, "Xh2HalBackend")


def _get_shared_context(device_id: int):
    key = _shared_context_key(device_id)
    context = _SHARED_CONTEXTS.get(key)
    if context is not None:
        return context
    runtime = _load_tcim_runtime()
    backend_name = key[1]
    device_manager = runtime.DevManager([device_id], backend_name=backend_name)
    weight_manager = runtime.WeightManager(device_manager)
    context = {
        "runtime": runtime,
        "device_manager": device_manager,
        "weight_manager": weight_manager,
    }
    _SHARED_CONTEXTS[key] = context
    return context


@dataclass
class TcimHMONNXInferenceConfig:
    enable_cuda_graph: bool = False
    enable_auto_offload: bool = False
    enable_golden: bool = False
    exec_devices: list[str | torch.device | int] | str | torch.device | int | None = None
    enable_fast: bool = False
    layers: list[Any] | None = None


class _EmptyInitializerPool:
    def summary(self) -> dict:
        return {}


class TcimLLMHMONNXLoader:
    def __init__(self, prefill_onnx_file: str, decode_onnx_file: str) -> None:
        self.prefill_onnx_file = str(prefill_onnx_file)
        self.decode_onnx_file = str(decode_onnx_file)
        self.prefill_graph = None
        self.decode_graph = None
        self.initializer_pool = _EmptyInitializerPool()


class TcimMultiHMONNXLoader:
    def __init__(self, graph_files: Mapping[str, str]) -> None:
        self.graph_files = dict(graph_files)
        self.graphs = dict.fromkeys(self.graph_files)
        self.initializer_pool = _EmptyInitializerPool()


class TcimCacheTensor(torch.Tensor):
    """A TCIM device-buffer owner with a CacheTensor-compatible lifecycle.

    A zero seed initializes the pre-allocated TCIM input directly on device; a
    non-zero seed is copied once when the first session input is bound. Subsequent
    sessions receive the same TCIM device tensor via ``set_dev_input`` and cache
    state is never copied back to a Torch tensor after each invocation.
    """

    is_custom_tensor = True

    @staticmethod
    def __new__(cls, data=None, requires_grad: bool = False):
        if data is None:
            raise ValueError("TCIM CacheTensor requires an initial tensor")
        tensor = data if torch.is_tensor(data) else torch.as_tensor(data)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.shape,
            strides=tensor.stride(),
            storage_offset=0,
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            requires_grad=requires_grad,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        del cls, func, types, args, kwargs
        raise RuntimeError("generic Torch operations are unsupported for a TCIM device cache")

    def __init__(self, data=None, requires_grad: bool = False) -> None:
        del requires_grad
        if data is None:
            raise ValueError("TCIM CacheTensor requires an initial tensor")
        tensor = data if torch.is_tensor(data) else torch.as_tensor(data)
        self._seed = tensor.detach()
        self._shape = torch.Size(tensor.shape)
        self._dtype = tensor.dtype
        self._device = tensor.device
        try:
            self._seed_is_zero = tensor.numel() == 0 or not bool(torch.count_nonzero(tensor).item())
        except (RuntimeError, NotImplementedError):
            self._seed_is_zero = False
        self._dev_tensor = None
        self._bindings: list[tuple[Any, str]] = []
        self._snapshot = None
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("TCIM cache is closed")

    @property
    def data(self):
        self._ensure_open()
        if self._dev_tensor is not None:
            raise RuntimeError("bound TCIM cache has no host Torch storage; use its device binding")
        return self._seed

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args: Any, **kwargs: Any):
        self._ensure_open()
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        target_device = self._device if device is None else torch.device(device)
        target_dtype = self._dtype if dtype is None else dtype
        if self._dev_tensor is not None:
            if target_device != self._device or target_dtype != self._dtype:
                raise RuntimeError("a bound TCIM cache cannot change device or dtype")
            return self
        self._seed = self._seed.to(device=target_device, dtype=target_dtype)
        self._device = target_device
        self._dtype = target_dtype
        return self

    def clone(self, *args: Any, **kwargs: Any):
        self._ensure_open()
        del args, kwargs
        if self._dev_tensor is not None:
            raise RuntimeError("clone() of a bound TCIM cache is unsupported; use snapshot()/rollback()")
        return type(self)(self._seed.clone())

    def detach(self):
        return self

    def __getitem__(self, key):
        self._ensure_open()
        if self._dev_tensor is not None and (key == slice(None) or key is Ellipsis):
            return self
        if self._dev_tensor is not None:
            raise RuntimeError("slicing a bound TCIM cache is unsupported")
        return self._seed.__getitem__(key)

    def __setitem__(self, key, value) -> None:
        self._ensure_open()
        if self._dev_tensor is not None:
            if (key == slice(None) or key is Ellipsis) and value is self:
                return
            raise RuntimeError("host assignment to a bound TCIM cache is unsupported")
        self._seed.__setitem__(key, value)

    def bind(self, session: "TcimHMONNXInference", input_name: str) -> None:
        self._ensure_open()
        session._ensure_loaded()
        info = session.get_input(input_name)
        if tuple(info.shape) != tuple(self.shape) or info.dtype != self.dtype:
            raise ValueError(
                f"TCIM cache {input_name!r} ABI mismatch: "
                f"session=({tuple(info.shape)}, {info.dtype}), cache=({tuple(self.shape)}, {self.dtype})"
            )
        module = session._module
        if self._dev_tensor is None:
            if self._seed_is_zero:
                dev_tensor = module.get_dev_input(input_name)
                set_zero = getattr(dev_tensor, "set_zero", None)
                if not callable(set_zero):
                    raise RuntimeError("TCIM runtime tensor has no device-side set_zero")
                set_zero()
                self._dev_tensor = dev_tensor
            else:
                seed = _align_array_to_info(_to_numpy(self._seed), info)
                module.set_input(input_name, seed)
                self._dev_tensor = module.get_dev_input(input_name)
            self._seed = None
        else:
            module.set_dev_input(input_name, self._dev_tensor)
        binding = (module, input_name)
        if binding not in self._bindings:
            self._bindings.append(binding)
        session._bound_caches.add(input_name)
        session._bound_cache_objects[input_name] = self

    def adopt_output(self, session: "TcimHMONNXInference", output_name: str) -> None:
        self._ensure_open()
        info = session.get_output(output_name)
        if tuple(info.shape) != tuple(self.shape) or info.dtype != self.dtype:
            raise ValueError(
                f"TCIM cache output {output_name!r} ABI mismatch: "
                f"session=({tuple(info.shape)}, {info.dtype}), cache=({tuple(self.shape)}, {self.dtype})"
            )
        get_dev_output = getattr(session._module, "get_dev_output", None)
        if not callable(get_dev_output):
            raise RuntimeError("TCIM runtime module does not expose device outputs for cache hand-off")
        self._dev_tensor = get_dev_output(output_name)

    def reset(self) -> None:
        self._ensure_open()
        self._snapshot = None
        if self._dev_tensor is None:
            self._seed.zero_()
            return
        set_zero = getattr(self._dev_tensor, "set_zero", None)
        if not callable(set_zero):
            raise RuntimeError("TCIM runtime tensor does not support device-side reset")
        set_zero()

    def snapshot(self) -> None:
        self._ensure_open()
        if self._dev_tensor is None:
            self._snapshot = self._seed.clone()
            return
        clone = getattr(self._dev_tensor, "clone", None)
        if not callable(clone):
            raise RuntimeError("TCIM runtime tensor does not support device-side snapshot")
        self._snapshot = clone(auto_copy=True)

    def commit(self, accepted_count: int | None = None) -> None:
        self._ensure_open()
        if accepted_count is not None and int(accepted_count) < 0:
            raise ValueError("accepted_count must be non-negative")
        self._snapshot = None

    def rollback(self) -> None:
        self._ensure_open()
        if self._snapshot is None:
            raise RuntimeError("TCIM cache has no snapshot to roll back")
        if self._dev_tensor is None:
            self._seed.copy_(self._snapshot)
        else:
            copy_to = getattr(self._snapshot, "copy_to", None)
            if not callable(copy_to):
                raise RuntimeError("TCIM runtime tensor does not support device-side rollback")
            copy_to(self._dev_tensor)
        self._snapshot = None

    def close(self) -> None:
        self._bindings.clear()
        self._snapshot = None
        self._dev_tensor = None
        self._seed = None
        self._closed = True

    def __repr__(self) -> str:
        state = "closed" if self._closed else ("bound" if self._dev_tensor is not None else "unbound")
        return f"{type(self).__name__}(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, {state})"


class TcimHybridCacheTensor(TcimCacheTensor):
    def __init__(self, data=None, requires_grad: bool = False) -> None:
        super().__init__(data=data, requires_grad=requires_grad)
        self._cache_valid_len = 0

    @property
    def cache_valid_len(self):
        return self._cache_valid_len

    @cache_valid_len.setter
    def cache_valid_len(self, value) -> None:
        self._cache_valid_len = value

    def reset(self) -> None:
        super().reset()
        self._cache_valid_len = 0


def _is_kv_cache_name(name: str) -> bool:
    lowered = name.lower()
    return "kcache" in lowered or "vcache" in lowered


def _is_runtime_cache_name(name: str) -> bool:
    lowered = name.lower()
    return _is_kv_cache_name(lowered) or "cache" in lowered or "recurrent_state" in lowered


def _cache_input_candidates(output_name: str) -> tuple[str, ...]:
    candidates = [output_name]
    match = _CONV_CACHE_OUTPUT.fullmatch(output_name)
    if match:
        branch, layer = match.groups()
        suffix = f"{branch}_{layer}" if branch else layer
        candidates.append(f"past_conv_cache_{suffix}")
    match = _RECURRENT_CACHE_OUTPUT.fullmatch(output_name)
    if match:
        candidates.append(f"past_recurrent_state_{match.group(1)}")
    match = _PRESENT_KV_OUTPUT.fullmatch(output_name)
    if match:
        kind, layer = match.groups()
        candidates.append(f"past_{kind}_cache{f'_{layer}' if layer else ''}")
    return tuple(candidates)


class TcimHMONNXInference(torch.nn.Module):
    def __init__(self, onnx_file: str, session_config: TcimHMONNXInferenceConfig | None = None) -> None:
        super().__init__()
        self._onnx_file = str(onnx_file)
        self._hmm_path = resolve_tcim_model_path(onnx_file)
        self._session_config = session_config or TcimHMONNXInferenceConfig()
        configured_device = _first_exec_device(self._session_config.exec_devices)
        self._device = configured_device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._device_id = _parse_device_id(self._device)
        self._runtime = None
        self._module = None
        self._shared_context = None
        self._bound_caches: set[str] = set()
        self._bound_cache_objects: dict[str, TcimCacheTensor] = {}
        self.inputs: list[TensorInfo] = []
        self.outputs: list[TensorInfo] = []
        self.name = Path(self._onnx_file).stem
        self._step = 0
        self._dtype = torch.float16
        self._precision_mode = None
        self.golden_dir = str(Path(self._onnx_file).parent)
        self.save_golden_dir = self.golden_dir
        self._initialized = False
        self.enable_cuda_graph = False
        self.enable_auto_offload = False
        self.has_captured_graph = False
        self.capture_signature = None
        self.capture_unavailable_reason = "TCIM executes a compiled HMM and does not use HMONNX CUDA Graph"
        self.capture_disabled_reason = self.capture_unavailable_reason
        self.interpreter = SimpleNamespace(
            has_captured_graph=False,
            capture_signature=None,
            capture_disabled_reason=self.capture_disabled_reason,
            _capture_disabled_reason=self.capture_disabled_reason,
            capture_unavailable_reason=self.capture_unavailable_reason,
            clear=lambda *args, **kwargs: None,
            set_capture_cache_key=lambda *args, **kwargs: None,
        )
        if self._session_config.enable_golden:
            raise NotImplementedError("golden capture is not supported by the tcim HMONNX backend")
        # HMONNXInference exposes input/output metadata immediately after
        # construction.  Keep that observable contract for backend-neutral
        # callers which inspect ``session.inputs`` directly.
        self._ensure_loaded()

    def initialize(self) -> None:
        self._ensure_loaded()
        self._initialized = True

    @property
    def hmonnx_file(self) -> str:
        return self._onnx_file

    @property
    def model_name(self) -> str:
        return self.name

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        self._step = int(value)

    @property
    def precision_mode(self):
        return self._precision_mode

    @property
    def exec_device(self) -> torch.device:
        return self._device

    @exec_device.setter
    def exec_device(self, device: str | torch.device) -> None:
        self.to(device)

    @property
    def enable_auto_offload(self) -> bool:
        return False

    @enable_auto_offload.setter
    def enable_auto_offload(self, value: bool) -> None:
        del value

    def to(self, *args: Any, **kwargs: Any):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if dtype is not None:
            self._dtype = dtype
        if device is None:
            return self
        parsed = torch.device(device)
        device_id = _parse_device_id(parsed)
        if self._module is not None and device_id != self._device_id:
            self._module = None
            self._shared_context = None
            self._bound_caches = set()
            self._bound_cache_objects = {}
            self.inputs = []
            self.outputs = []
        self._device = parsed
        self._device_id = device_id
        return self

    def cuda(self, device: int | torch.device | None = None):
        if isinstance(device, int):
            target = torch.device("cuda", index=device)
        elif device is None:
            target = torch.device("cuda")
        else:
            target = torch.device(device)
        return self.to(target)

    def cpu(self, *args: Any, **kwargs: Any):
        return self.to(torch.device("cpu"))

    def get_input_names(self) -> list[str]:
        self._ensure_loaded()
        return [info.name for info in self.inputs]

    def get_output_names(self) -> list[str]:
        self._ensure_loaded()
        return [info.name for info in self.outputs]

    def get_input(self, name) -> TensorInfo:
        self._ensure_loaded()
        for info in self.inputs:
            if info.name == name:
                return info
        raise ValueError(f"input {name} not found")

    def get_output(self, name) -> TensorInfo:
        self._ensure_loaded()
        for info in self.outputs:
            if info.name == name:
                return info
        raise ValueError(f"output {name} not found")

    def set_exec_device(self, device: str | torch.device):
        return self.to(device)

    def to_fast_mode(self):
        return self

    def to_aligned_mode(self):
        return self

    def set_precision_mode(self, mode):
        self._precision_mode = mode
        return self

    @property
    def save_golden(self) -> bool:
        return False

    @save_golden.setter
    def save_golden(self, value: bool) -> None:
        if value:
            raise NotImplementedError("save_golden is not supported by the tcim HMONNX backend")

    def reset_step(self) -> None:
        self._step = 0

    def update_step(self, interval: int = 1) -> None:
        self._step += int(interval)

    def get_golden_dir(self) -> str:
        return str(self.save_golden_dir)

    def get_onnx_external_data_files(self) -> list[str]:
        return []

    def copy_hmonnx(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise NotImplementedError("copy_hmonnx() is not available for an already compiled HMM")

    def symlink_hmonnx(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise NotImplementedError("symlink_hmonnx() is not available for an already compiled HMM")

    def get_layer_infos(self) -> list[Any]:
        return list(self._session_config.layers or [])

    def warmup_without_cuda_graph(self, *args: Any):
        return self.forward(*args)

    def configure_auto_offload(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def clear_cuda_graph_state(self) -> None:
        self.has_captured_graph = False

    def _forward_without_cuda_graph(self, *args: Any):
        return self.forward(*args)

    def enable_inplace_cache_updates(self) -> int:
        return sum(1 for info in self.inputs if _is_runtime_cache_name(info.name))

    def enable_inplace_cache_output_reuse(self, output_names) -> None:
        del output_names

    def enable_borrowed_cuda_graph_outputs(self, output_names) -> int:
        del output_names
        return 0

    def release(self) -> None:
        self._module = None
        self._shared_context = None
        self._bound_caches.clear()
        self._bound_cache_objects.clear()
        self.inputs = []
        self.outputs = []
        self._initialized = False

    close = release

    def forward(self, *args: Any, **kwargs: Any):
        if kwargs:
            raise TypeError("TcimHMONNXInference.forward() does not accept keyword inputs")
        values = list(chain.from_iterable(arg if isinstance(arg, (list, tuple)) else (arg,) for arg in args))
        self._ensure_loaded()
        if len(values) != len(self.inputs):
            raise ValueError(f"Input number mismatch, expected {len(self.inputs)}, got {len(values)}")
        feed = {info.name: value for info, value in zip(self.inputs, values)}
        return self.run(feed)

    def run(self, input_feed: dict[str, Any]):
        self._ensure_loaded()
        module = self._module
        expected_names = {info.name for info in self.inputs}
        missing = expected_names.difference(input_feed)
        extra = set(input_feed).difference(expected_names)
        if missing or extra:
            raise ValueError(f"Input names mismatch, missing={sorted(missing)}, extra={sorted(extra)}")
        for info in self.inputs:
            value = input_feed[info.name]
            if isinstance(value, TcimCacheTensor):
                value.bind(self, info.name)
                continue
            if info.name in self._bound_caches:
                continue
            if _is_cache_tensor(value):
                raise TypeError(
                    f"TCIM cache input {info.name!r} must use "
                    "xhmodel_merak.exec_backends.CacheTensor to avoid Host/Device copies"
                )
            array = _align_array_to_info(_to_numpy(value), info)
            module.set_input(info.name, array)
        module.run()
        module.sync()
        outputs = []
        for info in self.outputs:
            cache = next(
                (
                    self._bound_cache_objects[name]
                    for name in _cache_input_candidates(info.name)
                    if name in self._bound_cache_objects
                ),
                None,
            )
            if cache is not None:
                cache.adopt_output(self, info.name)
                outputs.append(cache)
                continue
            host = module.get_output(info.name)
            array = np.ascontiguousarray(host.numpy())
            outputs.append(torch.as_tensor(array).to(self._device))
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def _ensure_loaded(self) -> None:
        if self._module is not None:
            return
        context = _get_shared_context(self._device_id)
        runtime = context["runtime"]
        option = runtime.Option(context["weight_manager"])
        module = runtime.load(str(self._hmm_path), option=option)
        self._runtime = runtime
        self._module = module
        self._shared_context = context
        self._bound_caches = set()
        self._bound_cache_objects = {}
        self.inputs = self._collect_infos(module, input_side=True)
        self.outputs = self._collect_infos(module, input_side=False)

    def _collect_infos(self, module, *, input_side: bool) -> list[TensorInfo]:
        count = module.get_num_inputs() if input_side else module.get_num_outputs()
        getter_name = module.get_input_name if input_side else module.get_output_name
        getter_info = module.get_input_info if input_side else module.get_output_info
        infos = []
        for index in range(count):
            name = getter_name(index)
            tcim_info = getter_info(name)
            infos.append(
                TensorInfo(
                    name,
                    torch.Size([int(dim) for dim in tcim_info.shape]),
                    _numpy_to_torch_dtype(tcim_info.dtype),
                )
            )
        return infos


class TcimHMONNXGoldenInference(TcimHMONNXInference):
    def __init__(self, onnx_file: str) -> None:
        super().__init__(onnx_file)
        self.golden_dir = str(Path(onnx_file).parent)
        self.legacy_mode = False


class TcimHMONNXGraphGoldenInference(TcimHMONNXGoldenInference):
    def __init__(
        self,
        onnx_file: str,
        *,
        enable_cuda_graph: bool = True,
        warmup_runs: int = 3,
        graph_warmup_runs: int = 6,
        clone_outputs: bool = True,
    ) -> None:
        super().__init__(onnx_file)
        self.requested_cuda_graph = bool(enable_cuda_graph)
        self.warmup_runs = int(warmup_runs)
        self.graph_warmup_runs = int(graph_warmup_runs)
        self.clone_outputs = bool(clone_outputs)
        self.enable_cuda_graph = False


class TcimHMONNXCUDAGraphInference(TcimHMONNXGraphGoldenInference):
    pass


class TcimHMONNXInferenceV2(TcimHMONNXInference):
    def __init__(
        self,
        hmonnx_file: str,
        session_config: TcimHMONNXInferenceConfig | None = None,
    ) -> None:
        super().__init__(hmonnx_file, session_config=session_config)
        self.onnx_graph = None

    @classmethod
    def from_onnx_graph(
        cls,
        hmonnx_file: str,
        onnx_graph: Any,
        session_config: TcimHMONNXInferenceConfig | None = None,
    ) -> "TcimHMONNXInferenceV2":
        session = cls(hmonnx_file, session_config=session_config)
        session.onnx_graph = None
        return session


def _torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    mapping = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.uint16: np.uint16,
        torch.uint32: np.uint32,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.bool: np.bool_,
    }
    mapped = mapping.get(dtype)
    if mapped is None:
        raise TypeError(f"Unsupported torch dtype {dtype}")
    return np.dtype(mapped)
