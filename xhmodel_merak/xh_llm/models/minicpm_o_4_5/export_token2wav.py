from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import torch
from onnx import numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator
from torch import Tensor, nn

from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

from .minicpmo_token2wav_modules import (
    FlowDecoderWrapper,
    FlowEstimatorStepWrapper,
    FlowFrontendWrapper,
    FlowStreamingFrontendWrapper,
    HiftF0Wrapper,
    HiftPart2Wrapper,
    _normalize_embedding,
)
from .streaming_token2wav import (
    FLOW_CACHE_NAMES,
    HiftStreamPart1Wrapper,
)


_RANDOM_ONNX_OPERATORS = frozenset({"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"})


def constantize_dynamic_expand_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fold constant-rooted Expand shape vectors into initializers, in place.

    The official Flow streaming internals build many Expand target shapes
    through chains of shape arithmetic whose leaves are all constants
    (`ConstantOfShape`/`Equal`/`Where`/...). ``torch.onnx.export`` leaves these
    as non-initializer tensors, so xhquant's Expand parser must resolve a
    dynamic shape even though it is provably constant. This pass evaluates any
    shape chain with no graph-input dependence up front, rewrites the Expand to
    consume an initializer, and drops the now-dead shape nodes. Semantics are
    unchanged (the folded value is the exact evaluated shape vector).
    """
    graph = model.graph
    node_list = list(graph.node)
    output_map: dict[str, onnx.NodeProto] = {}
    for node in node_list:
        for output in node.output:
            output_map.setdefault(output, node)
    initializer_map = {initializer.name: initializer for initializer in graph.initializer}
    graph_input_names = {value.name for value in graph.input}

    def _collect_constant_chain(shape_name: str):
        reachable: set[int] = set()
        leaves: set[str] = set()
        dynamic = [False]
        seen: set[str] = set()

        def walk(name: str) -> None:
            if name in seen:
                return
            seen.add(name)
            if name in initializer_map or name in graph_input_names:
                leaves.add(name)
                if name in graph_input_names:
                    dynamic[0] = True
                return
            node = output_map.get(name)
            if node is None:
                leaves.add(name)
                return
            if node.op_type == "Constant":
                leaves.add(name)
                return
            if id(node) in reachable:
                return
            reachable.add(id(node))
            for source in node.input:
                if source:
                    walk(source)

        walk(shape_name)
        return reachable, leaves, dynamic[0]

    shape_suffix = 0
    for expand in list(graph.node):
        if expand.op_type != "Expand" or len(expand.input) < 2:
            continue
        shape_name = expand.input[1]
        if shape_name in initializer_map:
            continue
        reachable, leaves, dynamic = _collect_constant_chain(shape_name)
        if dynamic or not reachable:
            continue
        topo_nodes = [candidate for candidate in node_list if id(candidate) in reachable]
        sub_initializers: list[onnx.TensorProto] = []
        for leaf in leaves:
            if leaf in initializer_map:
                sub_initializers.append(initializer_map[leaf])
            else:
                constant_node = output_map.get(leaf)
                if constant_node is not None and constant_node.op_type == "Constant":
                    value_attr = next((attr for attr in constant_node.attribute if attr.name == "value"), None)
                    if value_attr is not None:
                        sub_initializers.append(
                            copy.deepcopy(numpy_helper.from_array(numpy_helper.to_array(value_attr.t), leaf))
                        )
        sub_graph = onnx.helper.make_graph(
            topo_nodes,
            "expand_shape_fold",
            [],
            [onnx.helper.make_tensor_value_info(shape_name, onnx.TensorProto.INT64, None)],
            initializer=sub_initializers,
        )
        sub_model = onnx.helper.make_model(
            sub_graph,
            opset_imports=[onnx.helper.make_opsetid("", 17)],
            ir_version=model.ir_version,
        )
        reference = ReferenceEvaluator(sub_model)
        try:
            evaluated = reference.run(None, {})[0]
        except Exception:
            continue
        if evaluated.ndim != 1 or evaluated.dtype != np.int64:
            continue
        folded_name = f"expand_shape_folded_{shape_suffix}"
        shape_suffix += 1
        graph.initializer.append(numpy_helper.from_array(np.asarray(evaluated, dtype=np.int64), folded_name))
        expand.input[1] = folded_name
        for redundant in reachable:
            node = next(candidate for candidate in node_list if id(candidate) == redundant)
            for output in node.output:
                output_map.pop(output, None)
        for candidate in node_list:
            if id(candidate) in reachable:
                graph.node.remove(candidate)
    return model


def staticize_known_shape_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace ``Shape`` nodes whose source tensor has a fixed inferred shape.

    Streaming Flow is exported with fixed token and cache capacities. The
    official cache-buffer stores nevertheless lower their fixed extents as
    ``Shape(runtime_tensor)`` because the stores are expressed using runtime
    indexing. This pass specializes only the shape metadata; tensor values,
    cache reads, cache writes, and valid-length inputs remain graph inputs and
    therefore retain their runtime data dependence.
    """
    inferred = shape_inference.infer_shapes(model)
    shape_by_name: dict[str, tuple[int, ...]] = {}

    def add_shape(value: onnx.ValueInfoProto) -> None:
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            return
        dimensions = tuple(dimension.dim_value for dimension in tensor.shape.dim)
        if dimensions and all(dimension > 0 for dimension in dimensions):
            shape_by_name[value.name] = dimensions

    for value in inferred.graph.input:
        add_shape(value)
    for value in inferred.graph.value_info:
        add_shape(value)
    for value in inferred.graph.output:
        add_shape(value)
    for initializer in model.graph.initializer:
        shape_by_name[initializer.name] = tuple(initializer.dims)

    existing_initializers = {initializer.name for initializer in model.graph.initializer}
    shape_index = 0
    replacements: dict[str, str] = {}
    removable: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Shape" or not node.output:
            continue
        source_shape = shape_by_name.get(node.input[0])
        if source_shape is None:
            continue
        replacement = f"static_shape_{shape_index}"
        while replacement in existing_initializers:
            shape_index += 1
            replacement = f"static_shape_{shape_index}"
        shape_index += 1
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(source_shape, dtype=np.int64), name=replacement)
        )
        existing_initializers.add(replacement)
        replacements[node.output[0]] = replacement
        removable.add(node.name or node.output[0])

    if not replacements:
        return model
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[index] = replacements[input_name]
    kept_nodes = [node for node in model.graph.node if (node.name or node.output[0]) not in removable]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    return model


def remove_identity_expands(model: onnx.ModelProto) -> onnx.ModelProto:
    inferred = shape_inference.infer_shapes(model)
    shapes: dict[str, tuple[int, ...]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            continue
        dimensions = tuple(dimension.dim_value for dimension in tensor.shape.dim)
        if dimensions and all(dimension > 0 for dimension in dimensions):
            shapes[value.name] = dimensions
    rewrites: dict[str, str] = {}
    removable: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Expand" or shapes.get(node.input[0]) != shapes.get(node.output[0]):
            continue
        rewrites[node.output[0]] = node.input[0]
        removable.add(node.name or node.output[0])
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in rewrites:
                node.input[index] = rewrites[input_name]
    kept_nodes = [node for node in model.graph.node if (node.name or node.output[0]) not in removable]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    return model


def remove_static_broadcast_expands(model: onnx.ModelProto) -> onnx.ModelProto:
    inferred = shape_inference.infer_shapes(model)
    shapes: dict[str, tuple[int, ...]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            continue
        dimensions = tuple(dimension.dim_value for dimension in tensor.shape.dim)
        if dimensions and all(dimension > 0 for dimension in dimensions):
            shapes[value.name] = dimensions
    initializer_values = {
        initializer.name: tuple(int(value) for value in numpy_helper.to_array(initializer).reshape(-1).tolist())
        for initializer in model.graph.initializer
    }
    rewrites: dict[str, str] = {}
    removable: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Expand" or node.input[1] not in initializer_values:
            continue
        source_shape = shapes.get(node.input[0])
        if source_shape is None:
            continue
        target_shape = initializer_values[node.input[1]]
        if len(target_shape) < len(source_shape):
            continue
        aligned_source = (1,) * (len(target_shape) - len(source_shape)) + source_shape
        if aligned_source != target_shape:
            continue
        rewrites[node.output[0]] = node.input[0]
        removable.add(node.name or node.output[0])
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in rewrites:
                node.input[index] = rewrites[input_name]
    kept_nodes = [node for node in model.graph.node if (node.name or node.output[0]) not in removable]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    return model


@contextmanager
def _exportable_normalize():
    original = torch.nn.functional.normalize
    torch.nn.functional.normalize = _normalize_embedding
    try:
        yield
    finally:
        torch.nn.functional.normalize = original


def make_stream_export_flow(flow: nn.Module, *, cache_capacity: int, append_capacity: int = 56) -> nn.Module:
    export_flow = copy.deepcopy(flow)
    decoder = export_flow.decoder
    estimator = decoder.estimator
    decoder_att_cache = decoder.att_cache_buffer
    estimator_att_cache = estimator.att_cache_buffer
    decoder.register_buffer(
        "att_cache_buffer",
        decoder_att_cache.new_zeros(
            (*decoder_att_cache.shape[:-2], cache_capacity + append_capacity, decoder_att_cache.shape[-1])
        ),
        persistent=False,
    )
    estimator.register_buffer(
        "att_cache_buffer",
        estimator_att_cache.new_zeros(
            (*estimator_att_cache.shape[:-2], cache_capacity + append_capacity, estimator_att_cache.shape[-1])
        ),
        persistent=False,
    )
    if hasattr(estimator, "use_cuda_graph"):
        estimator.use_cuda_graph = False
    return export_flow


def require_deterministic_onnx(model: onnx.ModelProto) -> None:
    random_operators = sorted({node.op_type for node in model.graph.node if node.op_type in _RANDOM_ONNX_OPERATORS})
    if random_operators:
        raise RuntimeError(f"HiFT ONNX contains random operators: {', '.join(random_operators)}")


def prepare_deterministic_onnx(model: onnx.ModelProto) -> onnx.ModelProto:
    require_deterministic_onnx(model)
    return model


def load_token2wav_modules(model_dir: str) -> tuple[nn.Module, nn.Module]:
    import stepaudio2.token2wav as token2wav
    from hyperpyyaml import load_hyperpyyaml

    token2wav._setup_cosyvoice2_alias()
    asset_dir = Path(model_dir) / "assets" / "token2wav"
    with (asset_dir / "flow.yaml").open(encoding="utf-8") as stream:
        flow = load_hyperpyyaml(stream)["flow"]
    flow.load_state_dict(torch.load(asset_dir / "flow.pt", map_location="cpu", weights_only=True), strict=True)
    flow.eval()
    hift = token2wav.HiFTGenerator()
    state = torch.load(asset_dir / "hift.pt", map_location="cpu", weights_only=True)
    hift.load_state_dict({key.removeprefix("generator."): value for key, value in state.items()}, strict=True)
    hift.eval()
    return flow, hift


def _convert(
    module: nn.Module,
    inputs: Sequence[Tensor],
    names: tuple[Sequence[str], Sequence[str]],
    output_file: Path,
    target_device: str,
    quant_type: str,
    prepare: Callable[[onnx.ModelProto], onnx.ModelProto] | None = None,
) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="minicpmo45_token2wav_onnx_") as temporary_dir:
        source_file = Path(temporary_dir) / f"{output_file.stem}_source.onnx"
        with _exportable_normalize():
            torch.onnx.export(
                module,
                tuple(inputs),
                str(source_file),
                input_names=list(names[0]),
                output_names=list(names[1]),
                opset_version=17,
                do_constant_folding=True,
            )
        folded_model = staticize_known_shape_nodes(onnx.load(str(source_file)))
        folded_model = constantize_dynamic_expand_shapes(folded_model)
        folded_model = remove_identity_expands(folded_model)
        folded_model = remove_static_broadcast_expands(folded_model)
        if prepare is not None:
            folded_model = prepare(folded_model)
        onnx.save(folded_model, str(source_file))
        onnx.checker.check_model(str(source_file))
        if target_device != "XH2a":
            raise RuntimeError(f"Unsupported Token2Wav target device: {target_device}")
        quant_config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type))
        convert_onnx_to_hmonnx(
            str(source_file),
            [value.cpu() for value in inputs],
            DeviceType.XH2a,
            str(output_file),
            quant_config=quant_config,
            input_names=list(names[0]),
            output_names=list(names[1]),
        )
    return output_file


def prepare_hift_stream_onnx(model: onnx.ModelProto) -> onnx.ModelProto:
    """Materialize constant shape inputs in the streaming Mel/F0 graph."""
    constant_values: dict[str, np.ndarray] = {}
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for initializer in model.graph.initializer:
        constant_values[initializer.name] = numpy_helper.to_array(initializer)
    for node in model.graph.node:
        if node.op_type == "Constant":
            value = next((attribute.t for attribute in node.attribute if attribute.name == "value"), None)
            if value is not None:
                constant_values[node.output[0]] = numpy_helper.to_array(value)
    producers = {output: node for node in model.graph.node for output in node.output}

    def evaluate_constant(name: str) -> np.ndarray | None:
        if name in constant_values:
            return constant_values[name]
        node = producers.get(name)
        if node is None:
            return None
        values = [evaluate_constant(input_name) for input_name in node.input if input_name]
        if any(value is None for value in values):
            return None
        resolved = [value for value in values if value is not None]
        if node.op_type == "Cast":
            result = resolved[0].astype(np.int64)
        elif node.op_type == "Concat":
            axis = next(attribute.i for attribute in node.attribute if attribute.name == "axis")
            result = np.concatenate(resolved, axis=axis)
        elif node.op_type == "Unsqueeze":
            axes = resolved[1].reshape(-1).tolist()
            result = resolved[0]
            for axis in sorted((int(axis) for axis in axes)):
                result = np.expand_dims(result, axis=axis)
        elif node.op_type == "ConstantOfShape":
            shape = tuple(int(item) for item in resolved[0].reshape(-1))
            attribute = next((attribute for attribute in node.attribute if attribute.name == "value"), None)
            fill = numpy_helper.to_array(attribute.t).item() if attribute is not None else 0.0
            result = np.full(shape, fill)
        elif node.op_type == "Reshape":
            result = resolved[0].reshape(tuple(int(item) for item in resolved[1].reshape(-1)))
        elif node.op_type == "Transpose":
            permutation = next((list(attribute.ints) for attribute in node.attribute if attribute.name == "perm"), None)
            result = np.transpose(resolved[0], axes=permutation)
        elif node.op_type == "Slice":
            starts = resolved[1].reshape(-1)
            ends = resolved[2].reshape(-1)
            axes = resolved[3].reshape(-1) if len(resolved) > 3 else np.arange(starts.size)
            steps = resolved[4].reshape(-1) if len(resolved) > 4 else np.ones(starts.size)
            slices = [slice(None)] * resolved[0].ndim
            for start, end, axis, step in zip(starts, ends, axes, steps, strict=True):
                slices[int(axis)] = slice(int(start), int(end), int(step))
            result = resolved[0][tuple(slices)]
        else:
            return None
        constant_values[name] = result
        return result

    slice_index = 0
    for node in model.graph.node:
        if node.op_type != "Slice":
            continue
        for input_index in range(1, min(len(node.input), 5)):
            name = node.input[input_index]
            if not name or name in initializer_names:
                continue
            value = evaluate_constant(name)
            if value is None:
                continue
            initializer_name = f"hift_stream_slice_{slice_index}_{input_index}"
            slice_index += 1
            model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), initializer_name))
            initializer_names.add(initializer_name)
            constant_values[initializer_name] = np.asarray(value, dtype=np.int64)
            node.input[input_index] = initializer_name
    for node in model.graph.node:
        if (
            node.op_type not in {"CumSum", "Reshape", "Squeeze", "Unsqueeze"}
            or len(node.input) < 2
            or node.input[1] in initializer_names
        ):
            continue
        value = evaluate_constant(node.input[1])
        if value is None:
            continue
        initializer_name = f"hift_stream_unsqueeze_{slice_index}"
        slice_index += 1
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), initializer_name))
        initializer_names.add(initializer_name)
        constant_values[initializer_name] = np.asarray(value, dtype=np.int64)
        node.input[1] = initializer_name
    for node in model.graph.node:
        if node.op_type != "Pad" or len(node.input) < 2 or node.input[1] in initializer_names:
            continue
        value = evaluate_constant(node.input[1])
        if value is None:
            continue
        initializer_name = f"hift_stream_pad_{slice_index}"
        slice_index += 1
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), initializer_name))
        initializer_names.add(initializer_name)
        constant_values[initializer_name] = np.asarray(value, dtype=np.int64)
        node.input[1] = initializer_name
    for node in model.graph.node:
        if node.op_type != "Pow" or node.input[1] in initializer_names:
            continue
        value = evaluate_constant(node.input[1])
        if value is None:
            continue
        initializer_name = f"hift_stream_pow_{slice_index}"
        slice_index += 1
        array = np.asarray(value)
        model.graph.initializer.append(numpy_helper.from_array(array, initializer_name))
        initializer_names.add(initializer_name)
        constant_values[initializer_name] = array
        node.input[1] = initializer_name
    for node in model.graph.node:
        if node.op_type not in {"ReduceMean", "ReduceSum"} or len(node.input) < 2 or node.input[1] in initializer_names:
            continue
        value = evaluate_constant(node.input[1])
        if value is None:
            continue
        initializer_name = f"hift_stream_reduce_{slice_index}"
        slice_index += 1
        array = np.asarray(value, dtype=np.int64)
        model.graph.initializer.append(numpy_helper.from_array(array, initializer_name))
        initializer_names.add(initializer_name)
        constant_values[initializer_name] = array
        node.input[1] = initializer_name
    return model


def export_flow_frontend(
    flow: nn.Module,
    component_dir: Path,
    config: Mapping[str, object],
    target_device: str,
    name_prefix: str = "",
) -> dict[str, object]:
    export_mode = str(config["export_mode"])
    token_capacity = int(config["token_capacity"])
    mel_capacity = int(config["mel_capacity"])
    quant_type = str(config["quant_type"])
    inputs = (
        torch.zeros((1, token_capacity), dtype=torch.int32),
        torch.tensor([token_capacity], dtype=torch.int32),
        torch.zeros((1, mel_capacity, 80), dtype=torch.float32),
        torch.tensor([mel_capacity // 2], dtype=torch.int32),
        torch.zeros((1, 192), dtype=torch.float32),
    )
    graphs: dict[str, Path] = {}
    if export_mode == "offline":
        graphs["main"] = _convert(
            FlowFrontendWrapper(flow, mel_capacity),
            inputs,
            (
                ("tokens", "token_length", "prompt_feat", "prompt_feat_length", "embedding"),
                ("mu", "mask", "spks", "cond"),
            ),
            component_dir / f"{name_prefix}token2wav_flow_frontend_offline_{target_device}_{quant_type}.onnx",
            target_device,
            quant_type,
        )
    if export_mode in {"session", "stream"}:
        streaming = config["streaming"]
        prompt_token_capacity = int(streaming["prompt_token_capacity"])
        prompt_mel_capacity = int(streaming["prompt_mel_capacity"])
        stream_inputs = (
            torch.zeros((1, prompt_token_capacity), dtype=torch.int32),
            torch.zeros((1, prompt_mel_capacity, 80), dtype=torch.float32),
            torch.zeros((1, 192), dtype=torch.float32),
        )
        base_cache = flow.setup_cache(
            stream_inputs[0],
            stream_inputs[1],
            stream_inputs[2],
            n_timesteps=int(config["n_timesteps"]),
        )
        configured_base_shapes = {
            name: tuple(int(value) for value in streaming["base_cache_shapes"][name]) for name in FLOW_CACHE_NAMES
        }
        for name in FLOW_CACHE_NAMES:
            actual_shape = tuple(int(value) for value in base_cache[name].shape)
            if actual_shape != configured_base_shapes[name]:
                raise RuntimeError(
                    f"Token2Wav base cache {name} shape {actual_shape} != configured {configured_base_shapes[name]}"
                )
        stream_contract = {
            "pre_lookahead_len": int(streaming["pre_lookahead_len"]),
            "base_cache_valid_length": int(streaming["base_cache_valid_length"]),
            "base_cache_shapes": {name: list(shape) for name, shape in configured_base_shapes.items()},
        }
        all_cache_shapes = {
            name: tuple(int(value) for value in streaming["cache_shapes"][name]) for name in FLOW_CACHE_NAMES
        }
        cache_shapes = {name: all_cache_shapes[name] for name in ("conformer_cnn_cache", "conformer_att_cache")}
        frontend_input_names = tuple(streaming["frontend_input_names"])
        frontend_output_names = tuple(streaming["frontend_output_names"])
        chunk_tokens = int(streaming["chunk_token_capacity"])
        output_mel_capacities = {mode: int(capacity) for mode, capacity in streaming["output_mel_capacities"].items()}
        frontend_inputs = (
            torch.zeros((1, chunk_tokens), dtype=torch.int32),
            torch.tensor([chunk_tokens], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.zeros((1, 192), dtype=torch.float32),
            torch.zeros(cache_shapes["conformer_cnn_cache"], dtype=torch.float32),
            torch.zeros(cache_shapes["conformer_att_cache"], dtype=torch.float32),
            torch.tensor([int(streaming["base_cache_valid_length"])], dtype=torch.int32),
        )
        role = "stream_flow_frontend"
        graph_path = component_dir / f"{name_prefix}token2wav_flow_frontend_{role}_{target_device}_{quant_type}.onnx"
        graphs[role] = _convert(
            FlowStreamingFrontendWrapper(flow),
            frontend_inputs,
            (frontend_input_names, frontend_output_names),
            graph_path,
            target_device,
            quant_type,
        )
        stream_contract.update(
            {
                "frontend_input_names": list(frontend_input_names),
                "cache_shapes": {name: list(shape) for name, shape in all_cache_shapes.items()},
                "frontend_cache_shapes": {name: list(shape) for name, shape in cache_shapes.items()},
                "chunk_token_capacity": chunk_tokens,
                "base_conformer_layers": int(streaming["base_conformer_layers"]),
                "output_mel_capacities": output_mel_capacities,
            }
        )
    result = {
        "export_mode": export_mode,
        "quant_type": quant_type,
        "graphs": graphs,
        "token_capacity": token_capacity,
        "mel_capacity": mel_capacity,
        "up_rate": int(flow.up_rate),
    }
    if export_mode in {"session", "stream"}:
        result["stream_contract"] = stream_contract
    return result


def export_flow_decoder(
    flow: nn.Module,
    component_dir: Path,
    config: Mapping[str, object],
    target_device: str,
    name_prefix: str = "",
) -> dict[str, object]:
    component_dir.mkdir(parents=True, exist_ok=True)
    export_mode = str(config["export_mode"])
    mel_capacity = int(config["mel_capacity"])
    quant_type = str(config["quant_type"])
    inputs = (
        torch.zeros((2, 80, mel_capacity)),
        torch.ones((2, 1, mel_capacity)),
        torch.zeros((2, 80, mel_capacity)),
        torch.zeros((2,)),
        torch.zeros((2, 80)),
        torch.zeros((2, 80, mel_capacity)),
    )
    graphs: dict[str, Path] = {}
    if export_mode == "offline":
        graphs["main"] = _convert(
            FlowDecoderWrapper(flow.decoder.estimator),
            inputs,
            (("x", "mask", "mu", "t", "spks", "cond"), ("estimator_out",)),
            component_dir / f"{name_prefix}token2wav_flow_decoder_offline_{target_device}_{quant_type}.onnx",
            target_device,
            quant_type,
        )
    if export_mode in {"session", "stream"}:
        streaming = config["streaming"]
        step_shapes = streaming["estimator_step_cache_shapes"]
        stream_inputs = (
            torch.zeros((2, 80, int(streaming["append_capacity"]))),
            torch.zeros((2, 80, int(streaming["append_capacity"]))),
            torch.zeros((2,)),
            torch.zeros((2, 80)),
            torch.zeros((2, 80, int(streaming["append_capacity"]))),
            torch.zeros(tuple(step_shapes["input_cnn"])),
            torch.zeros(tuple(step_shapes["input_att"])),
            torch.tensor([int(streaming["base_cache_valid_length"])], dtype=torch.int32),
            torch.tensor([int(streaming["append_capacity"])], dtype=torch.int32),
        )
        input_names = tuple(streaming["estimator_step_input_names"])
        output_names = tuple(streaming["estimator_step_output_names"])
        graph_path = (
            component_dir
            / f"{name_prefix}token2wav_flow_decoder_stream_flow_estimator_step_{target_device}_{quant_type}.onnx"
        )
        graphs["stream_flow_estimator_step"] = _convert(
            FlowEstimatorStepWrapper(flow.decoder.estimator),
            stream_inputs,
            (input_names, output_names),
            graph_path,
            target_device,
            quant_type,
        )
        host_timestep_cache_banks = int(streaming["host_timestep_cache_banks"])
        stream_contract = {
            "estimator_step_cache_shapes": dict(step_shapes),
            "append_capacity": int(streaming["append_capacity"]),
            "base_cache_valid_length": int(streaming["base_cache_valid_length"]),
            "host_timestep_cache_banks": host_timestep_cache_banks,
            "prompt_cache_policy": dict(streaming["prompt_cache_policy"]),
        }
    result = {
        "export_mode": export_mode,
        "quant_type": quant_type,
        "graphs": graphs,
        "mel_capacity": mel_capacity,
        "n_timesteps": int(config["n_timesteps"]),
        "cfg_rate": float(flow.decoder.inference_cfg_rate),
        "noise_shape": [int(value) for value in flow.decoder.rand_noise.shape],
    }
    if export_mode in {"session", "stream"}:
        result["stream_contract"] = stream_contract
    return result


def export_hift(
    hift: nn.Module,
    component_dir: Path,
    config: Mapping[str, object],
    target_device: str,
    name_prefix: str = "",
) -> dict[str, object]:
    export_mode = str(config["export_mode"])
    frame_capacity = int(config["frame_capacity"])
    quant_type = str(config["quant_type"])
    source_length = frame_capacity * 480 - 480
    harmonic_count = int(hift.nb_harmonics) + 1
    graphs: dict[str, Path] = {}
    result: dict[str, object] = {
        "export_mode": export_mode,
        "quant_type": quant_type,
        "graphs": graphs,
        "frame_capacity": frame_capacity,
        "hop_length": 480,
        "source_length": source_length,
        "harmonic_count": harmonic_count,
        "stft": {
            "n_fft": int(hift.istft_params["n_fft"]),
            "hop_length": int(hift.istft_params["hop_len"]),
        },
        "source_generator": {
            "sampling_rate": int(hift.m_source.l_sin_gen.sampling_rate),
            "upsample_scale": int(hift.m_source.l_sin_gen.upsample_scale),
            "sine_amp": float(hift.m_source.l_sin_gen.sine_amp),
            "noise_std": float(hift.m_source.l_sin_gen.noise_std),
            "voiced_threshold": float(hift.m_source.l_sin_gen.voiced_threshold),
            "linear_weight": hift.m_source.l_linear.weight.detach().float().cpu().tolist(),
            "linear_bias": hift.m_source.l_linear.bias.detach().float().cpu().tolist(),
        },
    }
    if export_mode == "offline":
        graphs["part1"] = _convert(
            HiftF0Wrapper(hift),
            (torch.zeros((1, 80, frame_capacity)),),
            (("speech_feat",), ("f0",)),
            component_dir / f"{name_prefix}token2wav_hift_offline_part1_{target_device}_{quant_type}.onnx",
            target_device,
            quant_type,
        )
        stft_frames = frame_capacity * 480 // int(hift.istft_params["hop_len"]) + 1
        graphs["part2"] = _convert(
            HiftPart2Wrapper(hift, waveform_length=frame_capacity * 480),
            (
                torch.zeros((1, 80, frame_capacity)),
                torch.zeros((1, int(hift.istft_params["n_fft"]) + 2, stft_frames)),
            ),
            (("speech_feat", "source_stft"), ("waveform",)),
            component_dir / f"{name_prefix}token2wav_hift_offline_part2_{target_device}_{quant_type}.onnx",
            target_device,
            quant_type,
            prepare=prepare_deterministic_onnx,
        )
    if export_mode in {"session", "stream"}:
        streaming = config["streaming"]
        stream_frame_capacity = int(streaming["frame_capacity"])
        stream_source_length = (stream_frame_capacity + int(streaming["mel_cache_length"])) * 480
        source_cache_length = int(streaming["source_cache_length"])
        stream_inputs = (
            torch.zeros((1, 80, stream_frame_capacity)),
            torch.tensor([stream_frame_capacity], dtype=torch.int32),
            torch.zeros((1, int(streaming["mel_cache_length"]), 80)).transpose(1, 2),
            torch.tensor([0], dtype=torch.int32),
        )
        input_names = (
            "speech_feat",
            "speech_feat_valid_length",
            "past_mel",
            "past_mel_valid_length",
        )
        graphs["stream_hift_part1"] = _convert(
            HiftStreamPart1Wrapper(
                hift,
                mel_cache_length=int(streaming["mel_cache_length"]),
            ),
            stream_inputs,
            (input_names, ("prepared_speech_feat", "f0")),
            component_dir / f"{name_prefix}token2wav_hift_stream_part1_{target_device}_{quant_type}.onnx",
            target_device,
            quant_type,
            prepare=prepare_hift_stream_onnx,
        )
        stream_stft_frames = stream_source_length // int(hift.istft_params["hop_len"]) + 1
        graphs["stream_hift_part2"] = _convert(
            HiftPart2Wrapper(hift, waveform_length=stream_source_length),
            (
                torch.zeros((1, 80, stream_frame_capacity + int(streaming["mel_cache_length"]))),
                torch.zeros((1, int(hift.istft_params["n_fft"]) + 2, stream_stft_frames)),
            ),
            (("prepared_speech_feat", "source_stft"), ("raw_waveform",)),
            component_dir / f"{name_prefix}token2wav_hift_stream_part2_{target_device}_{quant_type}.onnx",
            target_device,
            quant_type,
            prepare=prepare_deterministic_onnx,
        )
        stream_contract = {
            "frame_capacity": stream_frame_capacity,
            "mel_cache_length": int(streaming["mel_cache_length"]),
            "source_cache_length": source_cache_length,
            "speech_cache_length": int(streaming["speech_cache_length"]),
            "harmonic_count": harmonic_count,
            "source_length": stream_source_length,
        }
    if export_mode in {"session", "stream"}:
        result["stream_contract"] = stream_contract
    return result


def export_minicpm_o_4_5_token2wav_flow_frontend(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, object],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, object]:
    del device
    flow, _ = load_token2wav_modules(model_dir)
    return export_flow_frontend(
        flow,
        work_dir / "token2wav_flow_frontend",
        component_cfg,
        target_device,
        name_prefix=f"{model_name or 'minicpm_o_4_5'}_",
    )


def export_minicpm_o_4_5_token2wav_flow_decoder(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, object],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, object]:
    del device
    flow, _ = load_token2wav_modules(model_dir)
    return export_flow_decoder(
        flow,
        work_dir / "token2wav_flow_decoder",
        component_cfg,
        target_device,
        name_prefix=f"{model_name or 'minicpm_o_4_5'}_",
    )


def export_minicpm_o_4_5_token2wav_hift(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, object],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, object]:
    del device
    _, hift = load_token2wav_modules(model_dir)
    return export_hift(
        hift,
        work_dir / "token2wav_hift",
        component_cfg,
        target_device,
        name_prefix=f"{model_name or 'minicpm_o_4_5'}_",
    )
