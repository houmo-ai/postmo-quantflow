"""Qwen4 decoder-block streaming export using the existing HMONNX assembler."""

from __future__ import annotations

import copy
import gc
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

import xhquant.nn as xhnn

from ...mrope import MROPE_CURRENT_POSITION_NAMES, MROPE_POOL_POSITION_NAMES
from .cache_contract import Qwen4ExpCacheContract, qwen4_exp_cache_input_specs, resolve_qwen4_exp_cache_shape
from .modeling import (
    Qwen4ExpExportableGatedDeltaNet,
    Qwen4ExpTextExportWrapper,
    _replace_native_modules_for_export,
    extract_qwen4_exp_text_config,
    extract_qwen4_exp_text_model,
)
from .qsa_host_controls import QSA_HOST_CONTROL_NAMES, build_qsa_host_controls


_BLOCK_INPUTS = (
    "hidden_states",
    "cos",
    "sin",
    "attention_mask",
    "conv_mask",
    "past_seq_length",
    "current_input_length",
    "current_position_indices",
    "ple_input_ids",
)


def _layer_contract(config, layer_idx):
    contract = Qwen4ExpCacheContract.from_text_config(config)
    return replace(contract, layers=(contract.layers[layer_idx],), model_tensors=())


def _cache_value(cache, spec):
    layer = cache.layers[spec.layer_idx]
    if spec.state_kind in {"delta_conv", "ple_conv", "ple_ngram_history"}:
        return layer.conv_states[spec.state_idx]
    if spec.state_kind == "delta_recurrent":
        return layer.recurrent_states[spec.state_idx]
    attribute = {
        "kv_key": "keys",
        "kv_value": "values",
        "qsa_indexer_key": "indexer_keys",
        "qsa_indexer_tail": "indexer_tail",
        "qsa_key_tail": "key_tail",
        "qsa_value_tail": "value_tail",
    }[spec.state_kind]
    return getattr(layer, attribute)


def _set_cache_value(cache, spec, value):
    layer = cache.layers[spec.layer_idx]
    if spec.state_kind in {"delta_conv", "ple_conv", "ple_ngram_history"}:
        layer.conv_states[spec.state_idx] = value
    elif spec.state_kind == "delta_recurrent":
        layer.recurrent_states[spec.state_idx] = value
    else:
        attribute = {
            "kv_key": "keys",
            "kv_value": "values",
            "qsa_indexer_key": "indexer_keys",
            "qsa_indexer_tail": "indexer_tail",
            "qsa_key_tail": "key_tail",
            "qsa_value_tail": "value_tail",
        }[spec.state_kind]
        setattr(layer, attribute, value)


def _state_sample(spec, config, sequence_length, cache_length, *, inplace_recurrent=False):
    shape = resolve_qwen4_exp_cache_shape(spec, batch_size=1, context_length=cache_length)
    dtype = torch.int32 if spec.dtype == "integer" else torch.float16
    value = torch.zeros(shape, dtype=dtype)
    if spec.state_kind == "ple_ngram_history":
        eos = config.eos_token_id
        value.fill_(int(eos[0] if isinstance(eos, (list, tuple)) else eos))
    if spec.state_kind == "delta_recurrent" and inplace_recurrent:
        from xhquant.core import CacheTensor

        value = CacheTensor(value)
    return value


class Qwen4ExpBlockExportWrapper(Qwen4ExpTextExportWrapper):
    """Flatten exactly one decoder's state while retaining its original index."""

    def __init__(self, layer, config, layer_idx, *, sequence_length, cache_length, ple_specs=()):
        nn.Module.__init__(self)
        self.layer = layer
        self.layer_idx = layer_idx
        self.text_config = config
        self.cache_contract = _layer_contract(config, layer_idx)
        self.input_sequence_length = sequence_length
        self.cache_length = cache_length
        self.batch_size = 1
        self.ple_specs = tuple(ple_specs)
        _replace_native_modules_for_export(
            layer, batch_size=1, sequence_length=sequence_length, cache_length=cache_length
        )
        self.has_qsa = getattr(layer, "self_attn", None) is not None

    def _linear_attention_uses_gdr_cache_state(self, layer_idx):
        module = getattr(self.layer, "linear_attn", None)
        return isinstance(module, Qwen4ExpExportableGatedDeltaNet) and (
            module.linear_attention_mode != "recurrent"
            and bool(getattr(getattr(module, "chunk_scan_op", None), "state_is_cache", False))
        )

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask,
        conv_mask,
        past_seq_length,
        current_input_length,
        current_position_indices,
        ple_input_ids,
        states,
    ):
        count = len(self.cache_contract.tensors)
        controls = tuple(states[count + len(self.ple_specs) + i] for i in range(5)) if self.has_qsa else None
        cache = self._build_dynamic_cache(
            {name: states[index] for index, name in enumerate(self.cache_contract.expected_keys())}
        )
        cache.past_seq_length = past_seq_length
        cache.current_input_length = current_input_length
        cache.current_position_indices = current_position_indices
        cache.qsa_host_controls = controls
        if self.ple_specs:
            from .streaming_ple import set_streamed_ple_inputs

            set_streamed_ple_inputs(
                self.layer, self.ple_specs, tuple(states[count + i] for i in range(len(self.ple_specs))), cache=cache
            )
        hidden_states = self.layer(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
            conv_mask=conv_mask,
            past_key_values=cache,
            ple_input_ids=ple_input_ids,
        )
        return (hidden_states, *self._flatten_dynamic_cache(cache))


class _BlockPlaceholder(nn.Module):
    """Weight-free boundary whose final ports come from the actual child graph."""

    def __init__(self, config, layer_idx, graph, output_specs, *, cache_length):
        super().__init__()
        self.layer_idx = layer_idx
        self.contract = _layer_contract(config, layer_idx)
        self.input_names = tuple(value.name for value in graph.input)
        self.output_names = tuple(value.name for value in graph.output)
        self.output_specs = tuple(output_specs)
        self.cache_input_specs = tuple(
            (name, spec)
            for name, spec in qwen4_exp_cache_input_specs(Qwen4ExpCacheContract.from_text_config(config)).items()
            if spec.layer_idx == layer_idx
        )
        shape = [[dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in graph.output]
        import onnx

        dtypes = [str(onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)) for value in graph.output]
        self.placeholder = xhnn.PlaceHolderModule(
            output_shapes=shape,
            output_dtypes=dtypes,
            num_outputs=len(shape),
            content=f"block_{layer_idx}",
        )

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        conv_mask=None,
        past_key_values=None,
        ple_input_ids=None,
        **kwargs,
    ):
        values = dict(
            zip(
                _BLOCK_INPUTS,
                (
                    hidden_states,
                    *position_embeddings,
                    attention_mask,
                    conv_mask,
                    past_key_values.past_seq_length,
                    past_key_values.current_input_length,
                    past_key_values.current_position_indices,
                    ple_input_ids,
                ),
                strict=True,
            )
        )
        if past_key_values.qsa_host_controls is not None:
            values.update(zip(QSA_HOST_CONTROL_NAMES, past_key_values.qsa_host_controls, strict=True))
        for name, spec in self.cache_input_specs:
            values[name] = _cache_value(past_key_values, spec)
        for name in self.input_names:
            if name.startswith("ple_embeddings_"):
                values[name] = past_key_values._streamed_ple_embeddings[int(name.removeprefix("ple_embeddings_"))]
        arguments = tuple(values[name] for name in self.input_names)
        outputs = self.placeholder(*arguments)
        for index, spec in enumerate(self.output_specs):
            _set_cache_value(past_key_values, spec, outputs[index + 1])
        return outputs[0]


class _StreamingSkeleton(Qwen4ExpTextExportWrapper):
    def __init__(self, *args, inplace_layers=(), ple_specs=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.inplace_layers = tuple(inplace_layers)
        self.ple_specs = tuple(ple_specs)
        # Child graph ports define the skeleton's actual tensor dependencies.
        # Avoid constructing masks or position indices that no block consumes.
        child_inputs = {name for layer in self.native_text_model.layers for name in layer.input_names}
        self.needs_current_position_indices = "current_position_indices" in child_inputs
        self.needs_causal_attention_mask = "attention_mask" in child_inputs

    def _linear_attention_uses_gdr_cache_state(self, layer_idx):
        return layer_idx in self.inplace_layers


def _sample_block(wrapper, rotary):
    c, length, capacity = wrapper.text_config, wrapper.input_sequence_length, wrapper.cache_length
    embeddings = torch.zeros((1, length, c.hidden_size), dtype=torch.float16)
    rotary_width = int(rotary.inv_freq.numel()) * 2
    cos = torch.ones((1, length + (length + 3) // 4, rotary_width), dtype=torch.float16)
    sin = torch.zeros_like(cos)
    offsets = torch.arange(length, dtype=torch.int32).reshape(1, length)
    mask = torch.arange(capacity).view(1, 1, 1, -1) <= offsets.view(1, 1, length, 1)
    state = tuple(
        _state_sample(
            spec, c, length, capacity, inplace_recurrent=wrapper._linear_attention_uses_gdr_cache_state(spec.layer_idx)
        )
        for spec in wrapper.cache_contract.tensors
    )
    ple = tuple(
        torch.zeros(
            (1, length, (int(spec["ngram_size"]) - 1) * int(spec["heads_per_ngram"]), int(spec["embedding_dim"])),
            dtype=torch.float16,
        )
        for spec in wrapper.ple_specs
    )
    controls = ()
    if wrapper.has_qsa:
        topk = min(int(c.indexer_budget) // 4, capacity // 4)
        controls = build_qsa_host_controls(0, length, length, capacity, topk, device="cpu", dtype=torch.float16)
    return (
        embeddings.repeat(1, 1, int(c.hc_count)),
        cos,
        sin,
        mask,
        torch.ones((1, length), dtype=torch.bool),
        torch.zeros(1, dtype=torch.int32),
        torch.full((1,), length, dtype=torch.int32),
        offsets,
        torch.zeros((1, length), dtype=torch.int32),
        (*state, *ple, *controls),
    )


def _module_prefix(model, selected):
    return next(name for name, module in model.named_modules() if module is selected)


def export_qwen4_exp_streamed_phase(
    loader,
    output_dir,
    *,
    input_sequence_length,
    cache_length,
    quant_config,
    quant_device="cpu",
    use_fused_gdr=True,
    linear_attention_mode=None,
    return_hidden_states=False,
    ple_quant_type="sefp4",
    shared_quantized_head=None,
    capture_quantized_head=False,
    ple_resources=None,
):
    """Load/export/release one real decoder at a time, then fill a small skeleton."""
    import onnx

    from ...big_hf_model_helper import _module_name_to_filename
    from .checkpoint import _adapt_gptq_linears
    from .static_export import export_qwen4_exp_hmonnx
    from .streaming_ple import PLEExportResources

    if not 1 <= input_sequence_length <= cache_length:
        raise ValueError("Require 1 <= streamed sequence length <= cache length")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ple_resources = ple_resources if ple_resources is not None else PLEExportResources(root)
    source_text = extract_qwen4_exp_text_model(loader.model)
    config = extract_qwen4_exp_text_config(loader.config)
    prefix = _module_prefix(loader.model, source_text)
    skeleton = copy.deepcopy(source_text)
    skeleton.embed_tokens = nn.Identity()
    paths, ple_specs, inplace_layers = {}, [], []
    block_load_audits = []
    converted_sources = []
    full_contract = Qwen4ExpCacheContract.from_text_config(config)
    cache_names = qwen4_exp_cache_input_specs(full_contract)
    for index in range(int(config.num_hidden_layers)):
        block_root = root / "placeholders"
        block_prefix = f"{prefix}.layers.{index}"
        from .streaming_ple import streamed_ple_tensor_names

        skip = streamed_ple_tensor_names(source_text.layers[index], block_prefix) if ple_quant_type == "sefp4" else ()
        layer = loader.load_module(block_prefix, skip_tensors=skip)
        audit = {
            "layer_idx": index,
            "loaded_parameter_bytes": sum(
                value.numel() * value.element_size() for value in layer.parameters() if not value.is_meta
            ),
        }
        block_load_audits.append(audit)
        try:
            specs = []
            if skip:
                from .streaming_ple import prepare_streamed_ple

                specs = prepare_streamed_ple(
                    loader, layer, block_prefix, block_root, quant_device=quant_device, resources=ple_resources,
                )
            converted_sources.extend(f"{block_prefix}.{name}" for name in _adapt_gptq_linears(layer))
            wrapper = Qwen4ExpBlockExportWrapper(
                layer,
                config,
                index,
                sequence_length=input_sequence_length,
                cache_length=cache_length,
                ple_specs=specs,
            )
            linear = getattr(layer, "linear_attn", None)
            if use_fused_gdr and isinstance(linear, Qwen4ExpExportableGatedDeltaNet):
                linear.enable_fused_gdr(linear_attention_mode=linear_attention_mode)
            inplace = wrapper._linear_attention_uses_gdr_cache_state(index)
            if inplace:
                inplace_layers.append(index)
            state_specs = wrapper.output_cache_specs()
            block_names = tuple(name for name, spec in cache_names.items() if spec.layer_idx == index)
            names = (*_BLOCK_INPUTS, *block_names, *(spec["input_name"] for spec in specs))
            outputs = (
                "hidden_states_out",
                *(f"present_{name}" for name, spec in cache_names.items() if spec in state_specs),
            )
            if wrapper.has_qsa:
                names += QSA_HOST_CONTROL_NAMES
            result = export_qwen4_exp_hmonnx(
                layer,
                block_root,
                text_config=config,
                sample_inputs=_sample_block(wrapper, skeleton.rotary_emb),
                quant_config=quant_config,
                quant_device=quant_device,
                input_sequence_length=input_sequence_length,
                cache_length=cache_length,
                use_fused_gdr=use_fused_gdr,
                linear_attention_mode=linear_attention_mode,
                export_wrapper=wrapper,
                export_input_names=names,
                export_output_names=outputs,
                ple_quant_type=ple_quant_type,
                model_name=f"qwen4_exp_block_{index}",
                ple_resources=ple_resources,
            )
            path = Path(result.onnx_path).rename(block_root / f"{_module_name_to_filename(f'block_{index}')}.onnx")
            part_model = onnx.load(path, load_external_data=False)
            graph = part_model.graph
            # Private block ports represent actual dependencies. TorchFX may
            # retain declared but unused arguments; forwarding them through a
            # placeholder would construct masks that the child never consumes.
            used_inputs = {name for node in graph.node for name in node.input} | {value.name for value in graph.output}
            required_inputs = [value for value in graph.input if value.name in used_inputs]
            del graph.input[:]
            graph.input.extend(required_inputs)
            onnx.save(part_model, path)
            # The SDK may prune unused inputs; every retained state output must
            # still have the expected semantic name and ordering.
            if tuple(value.name for value in graph.output) != outputs:
                raise ValueError("Streaming decoder output contract changed during export")
            skeleton.layers[index] = _BlockPlaceholder(
                config,
                index,
                graph,
                state_specs,
                cache_length=cache_length,
            )
            paths[f"block_{index}"] = path
            for spec in specs:
                ple_specs.append(spec)
        finally:
            loader.release(layer)
            audit["remaining_parameter_bytes_after_release"] = sum(
                value.numel() * value.element_size() for value in layer.parameters() if not value.is_meta
            )
            if "wrapper" in locals():
                del wrapper
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    mixer = loader.load_module(f"{prefix}.hyper_connection_mixer")
    head = None
    try:
        head = loader.load_module("lm_head")
        skeleton.hyper_connection_mixer = mixer
        wrapper = _StreamingSkeleton(
            skeleton,
            config,
            lm_head=head,
            input_sequence_length=input_sequence_length,
            cache_length=cache_length,
            return_hidden_states=return_hidden_states,
            inplace_layers=inplace_layers,
            ple_specs=ple_specs,
        )
        sample = (
            torch.zeros((1, input_sequence_length, config.hidden_size), dtype=torch.float16),
            torch.zeros(1, dtype=torch.int32),
            torch.full((1,), input_sequence_length, dtype=torch.int32),
            torch.zeros((1, input_sequence_length), dtype=torch.int32),
            *(torch.zeros((1, input_sequence_length), dtype=torch.int32) for _ in range(3)),
            *(torch.zeros((1, (input_sequence_length + 3) // 4), dtype=torch.int32) for _ in range(3)),
            tuple(
                _state_sample(
                    spec,
                    config,
                    input_sequence_length,
                    cache_length,
                    inplace_recurrent=spec.layer_idx in inplace_layers,
                )
                for spec in full_contract.tensors
            ),
        )
        output_specs = set(spec.name for spec in wrapper.output_cache_specs())
        outputs = (
            "logits",
            *(("target_hidden_states",) if return_hidden_states else ()),
            *(f"present_{name}" for name, spec in cache_names.items() if spec.name in output_specs),
        )
        has_qsa = any(spec.state_kind == "qsa_indexer_key" for spec in full_contract.tensors)
        input_names = (
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "ple_input_ids",
            *MROPE_CURRENT_POSITION_NAMES,
            *MROPE_POOL_POSITION_NAMES,
            *cache_names,
        )
        if has_qsa:
            topk = min(int(config.indexer_budget) // 4, cache_length // 4)
            controls = build_qsa_host_controls(
                0, input_sequence_length, input_sequence_length, cache_length, topk, device="cpu", dtype=torch.float16
            )
            sample = (*sample[:-1], (*sample[-1], *controls))
            input_names += QSA_HOST_CONTROL_NAMES
        if ple_specs:
            values = tuple(torch.zeros((1, input_sequence_length, len(spec["vocab_sizes"]),
                                        spec["embedding_dim"]), dtype=torch.float16) for spec in ple_specs)
            sample = (*sample[:-1], (*sample[-1], *values))
            input_names += tuple(spec["input_name"] for spec in ple_specs)
        result = export_qwen4_exp_hmonnx(
            wrapper,
            root,
            text_config=config,
            sample_inputs=sample,
            quant_config=quant_config,
            quant_device=quant_device,
            input_sequence_length=input_sequence_length,
            cache_length=cache_length,
            export_wrapper=wrapper,
            export_input_names=input_names,
            export_output_names=outputs,
            ple_quant_type=ple_quant_type,
            shared_quantized_head=shared_quantized_head,
            capture_quantized_head=capture_quantized_head,
            ple_resources=ple_resources,
        )
        loader._loader.replace_hmonnx_placeholders_with_subgraphs(result.onnx_path, cleanup_temporary_files=True)
        onnx.checker.check_model(result.onnx_path)
        info = json.loads(Path(result.io_manifest_path).read_text())
        final_graph = onnx.load(result.onnx_path, load_external_data=False).graph
        info["inputs"] = [
            {
                "name": value.name,
                "shape": [
                    dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None
                    for dim in value.type.tensor_type.shape.dim
                ],
                "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type).lower(),
            }
            for value in final_graph.input
        ]
        info["outputs"] = [value.name for value in final_graph.output]
        info["converted_gptq_modules"] = converted_sources
        if ple_specs:
            info["ple_quantization"] = {
                "status": "externalized",
                "layers": ple_specs,
                "source": "streamed_checkpoint_rows",
            }
        info.update(
            export_mode="low_memory",
            streamed_blocks=len(paths),
            block_load_audits=block_load_audits,
            inplace_recurrent_cache_inputs=[
                spec.name
                for spec in full_contract.tensors
                if spec.layer_idx in inplace_layers and spec.state_kind == "delta_recurrent"
            ],
        )
        Path(result.io_manifest_path).write_text(json.dumps(info, indent=2) + "\n")
        return replace(result, converted_gptq_modules=tuple(converted_sources))
    finally:
        loader.release(mixer)
        if head is not None:
            loader.release(head)


__all__ = ["Qwen4ExpBlockExportWrapper", "export_qwen4_exp_streamed_phase"]
