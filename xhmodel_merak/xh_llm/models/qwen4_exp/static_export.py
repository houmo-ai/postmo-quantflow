"""Standard TorchFX export of a Qwen4 static tensor wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .cache_contract import Qwen4ExpCacheContract, qwen4_exp_cache_input_specs, resolve_qwen4_exp_cache_shape
from .checkpoint import _adapt_gptq_linears
from .modeling import build_qwen4_exp_text_export_wrapper, extract_qwen4_exp_text_config, extract_qwen4_exp_text_model
from .static_runtime import _mrope_inputs as _sample_mrope_position_inputs


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
    calibration_inputs=None,
    validate_export_graph_parity=False,
    shared_quantized_head=None,
    capture_quantized_head=False,
):
    """Export one already-static wrapper with the standard SDK lifecycle.

    Nested cache tuples remain intact for TorchFX tracing. The SDK frontend
    flattens them; PTQ and serialization consume the same shared flat ABI.
    Publication retains QSA layout resources required by the existing graph
    contract; capture and quantization use only the standard SDK frontend.
    """
    import onnx

    from xhquant.api import (
        ConfigDict,
        FrontendType,
        PrecisionMode,
        ptq_quantize,
        to_export_graph,
        to_export_hmonnx_v2,
        to_frontend_graph,
        to_quant_graph,
    )

    from ...big_hf_model_helper import _copy_module_shared_params
    from ...mrope import MROPE_CURRENT_POSITION_NAMES, MROPE_POOL_POSITION_NAMES
    from ...shared_head import bind_quantized_head, quantized_head
    from ...utils import unfold_args
    from .modeling import Qwen4ExpExportableAttention, Qwen4ExpExportableGatedDeltaNet, adapt_qwen4_exp_rms_norms
    from .ple_export import export_ple_sefp4_hmonnx, extract_qwen4_exp_ple_specs
    from .precision import resolve_ple_quant_type
    from .qsa_host_controls import QSA_HOST_CONTROL_NAMES
    from .xhquant_extensions import qwen4_exp_xhquant_extension_scope

    ple_quant_type = resolve_ple_quant_type(ple_quant_type)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ple_specs = extract_qwen4_exp_ple_specs(native_model) if ple_quant_type == "sefp4" else ()
    converted = _adapt_gptq_linears(native_model)
    if export_wrapper is None:
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
        )
        inplace = {
            s.name
            for s in wrapper.cache_contract.tensors
            if s.state_kind == "delta_recurrent" and wrapper._linear_attention_uses_gdr_cache_state(s.layer_idx)
        }
        export_output_names = export_output_names or (
            "logits",
            *(("target_hidden_states",) if return_hidden_states else ()),
            *(
                f"present_{name}"
                for name, spec in qwen4_exp_cache_input_specs(wrapper.cache_contract).items()
                if spec.name not in inplace
            ),
        )
    else:
        wrapper = export_wrapper
    if export_input_names is None or export_output_names is None:
        raise ValueError("Static wrapper export requires explicit input and output names")
    flat = tuple(unfold_args(sample_inputs))
    if len(flat) != len(export_input_names):
        raise ValueError("Static wrapper input count does not match the declared ABI")
    # Shared tensor ownership helper keeps PTQ module replacement independent
    # from the caller's native model without copying its large parameter data.
    wrapper = _copy_module_shared_params(wrapper)
    from xhquant.core import CacheTensor

    contract = getattr(wrapper, "cache_contract", None)
    specs = qwen4_exp_cache_input_specs(contract) if contract is not None else {}
    inplace_names = {
        name
        for name, spec in specs.items()
        if spec.state_kind in {"kv_key", "kv_value", "qsa_indexer_key"}
        or (spec.state_kind == "delta_recurrent" and wrapper._linear_attention_uses_gdr_cache_state(spec.layer_idx))
    }
    flat = tuple(
        CacheTensor(value.detach().clone()) if name in inplace_names else value
        for name, value in zip(export_input_names, flat, strict=True)
    )
    with qwen4_exp_xhquant_extension_scope():
        frontend = to_frontend_graph(wrapper, FrontendType.TorchFX, sample_inputs)
        quanted = to_quant_graph(frontend, "xh2a", ConfigDict(dict(quant_config or {"quant_type": "w8a8h1_sefp"})))
        if shared_quantized_head is not None:
            bind_quantized_head(quanted, shared_quantized_head)
        calibration = [
            [
                CacheTensor(value.detach().clone()) if name in inplace_names else value.detach().clone()
                for name, value in zip(export_input_names, unfold_args(values), strict=True)
            ]
            for values in (calibration_inputs or (sample_inputs,))
        ]
        ptq_quantize(
            quanted,
            calibration,
            PrecisionMode.ALIGNED,
            [str(quant_device or "cpu")],
            auto_release_unused_parameters=True,
            infer_shape=False,
        )
        quanted.fixed()
        fixed_head = quantized_head(quanted) if capture_quantized_head else None
        exported = to_export_graph(quanted, list(flat))
        path = Path(
            to_export_hmonnx_v2(
                exported,
                flat,
                str(root / "qwen4_exp_text.onnx"),
                cfg=ConfigDict(input_names=list(export_input_names), output_names=list(export_output_names)),
                save_as_external_data=True,
            )
        )
        parity = []
        if validate_export_graph_parity:
            from .golden_runtime import _validate_hmonnx_runtime_parity

            parity = _validate_hmonnx_runtime_parity(
                path, flat, exported, device=quant_device or "cpu", reference_device="cpu"
            )
    onnx.checker.check_model(str(path))
    ple = {"status": "disabled", "quant_type": ple_quant_type}
    if ple_quant_type == "sefp4":
        ple = export_ple_sefp4_hmonnx(path, path, ple_specs=ple_specs)
        onnx.checker.check_model(str(path))
    if export_wrapper is None:
        from .serving_graph import publish_qwen4_exp_qsa_layout_resources

        publish_qwen4_exp_qsa_layout_resources(
            path,
            batch_size=batch_size,
            input_sequence_length=width,
            num_attention_heads=int(config.num_attention_heads),
            num_key_value_heads=int(config.num_key_value_heads),
            head_dim=int(config.head_dim),
        )
        onnx.checker.check_model(str(path))
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
