"""Standard TorchFX export of a Qwen4 static tensor wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from xhmodel_merak.configuration_utils import HFModelConfig

from ...base_model import XHSubModel
from ...kv_cache_mixin import EmptyKVCacheMixin
from ...shared_head import SharedHeadMixin
from ...types import LLMModelState
from .cache_contract import Qwen4ExpCacheContract, qwen4_exp_cache_input_specs, resolve_qwen4_exp_cache_shape
from .checkpoint import _adapt_gptq_linears
from .modeling import build_qwen4_exp_text_export_wrapper, extract_qwen4_exp_text_config, extract_qwen4_exp_text_model
from .static_runtime import _mrope_inputs as _sample_mrope_position_inputs


class _StaticInputProcessor:
    def __init__(self, names, inplace):
        self.names, self.inplace = names, inplace
        self.device = "cpu"

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        return self

    def __call__(self, data):
        from xhquant.core import CacheTensor

        names = iter(self.names)

        def prepare(value):
            if isinstance(value, (tuple, list)):
                return type(value)(prepare(item) for item in value)
            name = next(names)
            value = value.detach().clone().to(self.device)
            return CacheTensor(value) if data["cache"] and name in self.inplace else value

        return prepare(data["inputs"])


class _StaticModel(SharedHeadMixin, XHSubModel):
    """A static Qwen4 component using the same lifecycle as Qwen3.5 MTP."""

    def __init__(self, wrapper, inputs, names, outputs, quant_config, quant_device, *,
                 model_name="qwen4_exp_text", chip_arch="XH2a"):
        from ...utils import unfold_args

        dtype = next(value.dtype for value in unfold_args(inputs) if value.is_floating_point())
        super().__init__(HFModelConfig(
            model_name=model_name, chip_arch=chip_arch, dtype=str(dtype).removeprefix("torch."),
            quant_scheme=quant_config or {"quant_type": "w8a8h1_sefp"}, quant_device=quant_device,
        ))
        self._source_wrapper, self._sample_inputs = wrapper, inputs
        self._input_names, self._output_names = tuple(names), tuple(outputs)
        contract = getattr(wrapper, "cache_contract", None)
        specs = qwen4_exp_cache_input_specs(contract) if contract is not None else {}
        self._inplace = {name for name in names if name.startswith(
            ("past_key_cache_", "past_value_cache_", "qsa_indexer_key_cache_")
        )}
        self._inplace.update(
            spec.name for spec in specs.values()
            if spec.state_kind == "delta_recurrent" and wrapper._linear_attention_uses_gdr_cache_state(spec.layer_idx)
        )

    def get_native_model(self):
        return self._source_wrapper

    def _to_wrap(self, wrapper):
        self._wrap_model = wrapper
        del self._source_wrapper

    def get_dummy_inputs(self):
        return {"inputs": self._sample_inputs, "cache": self._state != LLMModelState.WRAP}

    def _get_data_preprocessor(self):
        return _StaticInputProcessor(self._input_names, self._inplace)

    def get_kvcache_mixin(self):
        return EmptyKVCacheMixin()

    def _to_fronted(self, wrapper):
        from xhquant.api import FrontendType, to_frontend_graph

        inputs = self.get_data_preprocessor()(self.get_dummy_inputs())
        return to_frontend_graph(wrapper, FrontendType.TorchFX, inputs)

    def _to_quanted(self, frontend_model, state, **kwargs):
        return super()._to_quanted(frontend_model, state, infer_shape=False, **kwargs)

    def get_export_cfg(self):
        return {"input_names": list(self._input_names), "output_names": list(self._output_names)}

    def export_hmonnx(self, output_dir):
        return super()._export_hmonnx(output_dir)


def _module_device(module: torch.nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return buffer.device if buffer is not None else fallback


def _build_qwen4_exp_zero_cache_inputs(
    text_model: torch.nn.Module,
    text_config: Any,
    *,
    batch_size: int,
    context_length: int,
) -> tuple[torch.Tensor, ...]:
    contract = Qwen4ExpCacheContract.from_text_config(text_config)
    fallback_device = text_model.embed_tokens.weight.device
    fallback_dtype = text_model.embed_tokens.weight.dtype
    values: list[torch.Tensor] = []
    for spec in contract.tensors:
        shape = resolve_qwen4_exp_cache_shape(spec, batch_size=batch_size, context_length=context_length)
        if spec.layer_idx >= 0:
            device = _module_device(text_model.layers[spec.layer_idx], fallback_device)
        else:
            device = fallback_device
        dtype = fallback_dtype if spec.dtype == "floating" else torch.int32
        if spec.state_kind == "ple_ngram_history":
            eos_token_id = getattr(text_config, "eos_token_id", 0)
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]
            values.append(torch.full(shape, int(eos_token_id), dtype=dtype, device=device))
        else:
            values.append(torch.zeros(shape, dtype=dtype, device=device))
    return tuple(values)


@torch.no_grad()
def build_qwen4_exp_serving_sample_inputs(
    native_model: torch.nn.Module,
    *,
    input_sequence_length: int,
    context_length: int,
    text_config: Any | None = None,
) -> tuple[torch.nn.Module, Any, tuple[torch.Tensor, ...]]:
    """Build one static prefill/decode sample for the serving exporter."""

    text_model = extract_qwen4_exp_text_model(native_model).eval()
    resolved_config = extract_qwen4_exp_text_config(text_config or text_model)
    embedding = text_model.embed_tokens
    device = embedding.weight.device
    batch_size = 1
    token_ids = torch.zeros(
        (batch_size, int(input_sequence_length)),
        dtype=torch.long,
        device=device,
    )
    cache_inputs = _build_qwen4_exp_zero_cache_inputs(
        text_model,
        resolved_config,
        batch_size=batch_size,
        context_length=int(context_length),
    )
    from .qsa_host_controls import build_qsa_host_controls

    indexers = [layer.self_attn.indexer for layer in text_model.layers if hasattr(layer, "self_attn")]
    controls = ()
    if indexers:
        topks = {min(int(indexer.block_topk), int(context_length) // 4) for indexer in indexers}
        if len(topks) != 1:
            raise ValueError("QSA layers must share the host-control candidate width")
        controls = build_qsa_host_controls(
            0,
            input_sequence_length,
            input_sequence_length,
            context_length,
            topks.pop(),
            device=device,
            dtype=embedding.weight.dtype,
        )
    return (
        native_model,
        resolved_config,
        (
            embedding(token_ids),
            torch.zeros((1,), dtype=torch.int32, device=device),
            torch.full((1,), int(input_sequence_length), dtype=torch.int32, device=device),
            token_ids.to(torch.int32),
            *_sample_mrope_position_inputs(
                batch_size=batch_size,
                input_length=int(input_sequence_length),
                past_length=0,
                device=device,
            ),
            (*cache_inputs, *controls),
        ),
    )


@dataclass(frozen=True)
class Qwen4ExpStaticExportResult:
    onnx_path: str
    hmonnx_path: str
    io_manifest_path: str
    converted_gptq_modules: tuple[str, ...] = ()
    fixed_quantized_head: torch.nn.Module | None = field(default=None, repr=False, compare=False)

    def to_dict(self):
        # This operator is an in-process ownership handle, not artifact metadata.
        return {
            name: getattr(self, name)
            for name in (
                "onnx_path",
                "hmonnx_path",
                "io_manifest_path",
                "converted_gptq_modules",
            )
        }


def export_qwen4_exp_hmonnx(
    native_model,
    output_dir,
    *,
    sample_inputs,
    quant_config=None,
    quant_device="cpu",
    text_config=None,
    input_sequence_length=None,
    cache_length=1,
    batch_size=1,
    use_fused_gdr=False,
    linear_attention_mode=None,
    return_hidden_states=False,
    ple_quant_type="sefp4",
    export_wrapper=None,
    export_input_names=None,
    export_output_names=None,
    validate_export_graph_parity=False,
    shared_quantized_head=None,
    capture_quantized_head=False,
    model_name="qwen4_exp_text",
    ple_resources=None,
):
    """Export one already-static wrapper with the standard SDK lifecycle.

    Nested cache tuples remain intact for TorchFX tracing. The SDK frontend
    flattens them; PTQ and serialization consume the same shared flat ABI.
    Components use XHSubModel for PTQ, fixed-head binding and serialization.
    """
    import onnx

    from ...big_hf_model_helper import _copy_module_shared_params
    from ...mrope import MROPE_CURRENT_POSITION_NAMES, MROPE_POOL_POSITION_NAMES
    from ...shared_head import quantized_head
    from ...utils import unfold_args
    from .modeling import Qwen4ExpExportableAttention, Qwen4ExpExportableGatedDeltaNet, adapt_qwen4_exp_rms_norms
    from .ple_export import validate_qwen4_exp_ple_inputs
    from .precision import resolve_ple_quant_type
    from .qsa_host_controls import QSA_HOST_CONTROL_NAMES
    from .streaming_ple import PLEExportResources
    from .xhquant_extensions import qwen4_exp_xhquant_extension_scope

    ple_quant_type = resolve_ple_quant_type(ple_quant_type)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ple_resources = ple_resources if ple_resources is not None else PLEExportResources(root)
    if export_wrapper is None:
        native_model = _copy_module_shared_params(native_model)
    converted = _adapt_gptq_linears(native_model)
    if export_wrapper is None:
        from .streaming_ple import prepare_native_ple

        ple_specs = prepare_native_ple(
            native_model, root, quant_device=quant_device or "cpu", resources=ple_resources,
        ) if ple_quant_type == "sefp4" else ()
        config = extract_qwen4_exp_text_config(text_config or native_model)
        width = int(input_sequence_length or sample_inputs[0].shape[1])
        wrapper = build_qwen4_exp_text_export_wrapper(
            native_model,
            text_config=config,
            batch_size=batch_size,
            input_sequence_length=width,
            cache_length=cache_length,
            return_hidden_states=return_hidden_states,
        ).eval()
        wrapper.ple_specs = tuple(ple_specs)
        if ple_specs:
            from .host_ple_inputs import HostPLEInputs

            host = HostPLEInputs(ple_resources.root / "model.onnx", {"version": 1, "layers": list(ple_specs)})
            cache = dict(zip(wrapper.cache_keys, sample_inputs[-1][:len(wrapper.cache_keys)], strict=True))
            values = host.prepare(sample_inputs[3], cache, int(sample_inputs[2].item()), device=sample_inputs[0].device)
            sample_inputs = (*sample_inputs[:-1], (*sample_inputs[-1], *values))
        adapt_qwen4_exp_rms_norms(wrapper)
        if use_fused_gdr:
            for module in tuple(wrapper.modules()):
                if isinstance(module, Qwen4ExpExportableGatedDeltaNet):
                    module.enable_fused_gdr(linear_attention_mode=linear_attention_mode)
        if len(sample_inputs) != 11 or not isinstance(sample_inputs[-1], (tuple, list)):
            raise ValueError("Qwen4 wrapper samples require ten fixed inputs and one state/control tuple")
        qsa = [m for m in wrapper.modules() if isinstance(m, Qwen4ExpExportableAttention)]
        cache_names = tuple(qwen4_exp_cache_input_specs(wrapper.cache_contract))
        export_input_names = export_input_names or (
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "ple_input_ids",
            *MROPE_CURRENT_POSITION_NAMES,
            *MROPE_POOL_POSITION_NAMES,
            *cache_names,
            *(QSA_HOST_CONTROL_NAMES if qsa else ()),
            *(spec["input_name"] for spec in ple_specs),
        )
        output_specs = set(spec.name for spec in wrapper.output_cache_specs())
        export_output_names = export_output_names or (
            "logits",
            *(("target_hidden_states",) if return_hidden_states else ()),
            *(
                f"present_{name}"
                for name, spec in qwen4_exp_cache_input_specs(wrapper.cache_contract).items()
                if spec.name in output_specs
            ),
        )
    else:
        wrapper = export_wrapper
        ple_specs = getattr(wrapper, "ple_specs", ())
    if export_input_names is None or export_output_names is None:
        raise ValueError("Static wrapper export requires explicit input and output names")
    flat = tuple(unfold_args(sample_inputs))
    if len(flat) != len(export_input_names):
        raise ValueError("Static wrapper input count does not match the declared ABI")
    # Shared tensor ownership helper keeps PTQ module replacement independent
    # from the caller's native model without copying its large parameter data.
    wrapper = _copy_module_shared_params(wrapper)
    component = _StaticModel(
        wrapper, sample_inputs, export_input_names, export_output_names, quant_config, quant_device,
        model_name=model_name,
    )
    if shared_quantized_head is not None:
        component.bind_target_head(shared_quantized_head)
    with qwen4_exp_xhquant_extension_scope():
        path = Path(component.export_hmonnx(str(root)))
        fixed_head = quantized_head(component.quanted_model) if capture_quantized_head else None
        parity = []
        if validate_export_graph_parity:
            from .golden_runtime import _validate_hmonnx_runtime_parity

            flat = tuple(unfold_args(component.get_data_preprocessor()(component.get_dummy_inputs())))
            parity = _validate_hmonnx_runtime_parity(
                path, flat, component.quanted_model, device=quant_device or "cpu", reference_device="cpu"
            )
    onnx.checker.check_model(str(path))
    validate_qwen4_exp_ple_inputs(path, ple_specs, ple_resources.root)
    ple = {"status": "externalized" if ple_specs else "disabled", "quant_type": ple_quant_type,
           "layers": list(ple_specs)}
    graph = onnx.load(path, load_external_data=False).graph
    dtype_names = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.DOUBLE: "float64"}
    inputs = [
        {
            "name": v.name,
            "shape": [
                d.dim_value if d.HasField("dim_value") else d.dim_param or None for d in v.type.tensor_type.shape.dim
            ],
            "dtype": dtype_names.get(
                v.type.tensor_type.elem_type, onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type).lower()
            ),
        }
        for v in graph.input
    ]
    manifest = {
        "inputs": inputs,
        "outputs": [v.name for v in graph.output],
        "converted_gptq_modules": list(converted),
        "onnx_checker": "passed",
        "hmonnx_runtime_parity": parity,
        "ple_quantization": ple,
    }
    contract = getattr(wrapper, "cache_contract", None)
    if contract is not None:
        manifest["cache_contract"] = contract.to_dict()
    manifest_path = root / "qwen4_exp_hmonnx_io.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return Qwen4ExpStaticExportResult(str(path), str(path), str(manifest_path), converted, fixed_head)
