"""Artifact-only sharding for Qwen4-Exp's oversized PLE embedding table."""

from __future__ import annotations

import copy
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# ONNX exporters sanitize parameter dots to underscores.  Keep accepting the
# source-module spelling as well as the serialized spelling produced by the
# Qwen4 export graph.
_PLE_WEIGHT_SUFFIXES = ("ngram_embedding.weight", "ngram_embedding_weight")
_XH2A_DOMAIN = "ai.houmo.xh2a"


@dataclass(frozen=True)
class PLEGatherShardingResult:
    input_path: str
    output_path: str
    initializer_name: str
    gather_node_name: str
    shard_count: int
    shard_rows: tuple[int, ...]
    shard_nbytes: tuple[int, ...]


@dataclass(frozen=True)
class PLEShardTagGroupingResult:
    input_path: str
    output_path: str
    shard_count: int
    placement_group_size: int
    llm_boundary_shards: tuple[int, ...]


def _external_data_fields(initializer: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in initializer.external_data}


def _make_external_shard(
    initializer: TensorProto,
    *,
    name: str,
    rows: int,
    byte_offset: int,
    byte_length: int,
    source_model_path: Path,
    output_model_path: Path,
) -> TensorProto:
    fields = _external_data_fields(initializer)
    location = fields.get("location")
    if not location:
        raise ValueError(f"external initializer {initializer.name!r} has no location")
    source_data_path = (source_model_path.parent / location).resolve()
    output_location = os.path.relpath(source_data_path, output_model_path.parent.resolve())

    shard = copy.deepcopy(initializer)
    shard.name = name
    shard.dims[0] = rows
    shard.ClearField("external_data")
    shard.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", output_location),
        ("offset", str(byte_offset)),
        ("length", str(byte_length)),
    ):
        entry = shard.external_data.add()
        entry.key = key
        entry.value = value
    return shard


def shard_qwen4_exp_ple_gather(
    input_path: str | Path,
    output_path: str | Path,
    *,
    shard_count: int = 2,
) -> PLEGatherShardingResult:
    """Split the PLE axis-0 Gather without copying its external tensor bytes.

    Qwen4-Exp constructs PLE ids in ``[0, ngram_vocab_size)``. Within that
    model invariant, selecting valid local ids for each shard and then the
    matching Gather output is exactly equivalent to the original Gather. Keeping each
    shard as a separate external initializer lets HMONNX auto-offload place
    the shards on different GPUs.
    """

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if shard_count < 2:
        raise ValueError("Qwen4-Exp PLE serving requires at least two row shards")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    model = onnx.load(str(input_path), load_external_data=False)
    candidates = [
        initializer
        for initializer in model.graph.initializer
        if "ple" in initializer.name and initializer.name.endswith(_PLE_WEIGHT_SUFFIXES)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one Qwen4-Exp PLE initializer ending in {_PLE_WEIGHT_SUFFIXES!r}, "
            f"found {len(candidates)}"
        )
    initializer = candidates[0]
    if initializer.data_location != TensorProto.EXTERNAL:
        raise ValueError(f"PLE initializer {initializer.name!r} must use ONNX external data")
    if len(initializer.dims) < 2 or int(initializer.dims[0]) < shard_count:
        raise ValueError(f"invalid PLE initializer shape: {list(initializer.dims)}")

    consumers = [node for node in model.graph.node if initializer.name in node.input]
    if len(consumers) != 1 or consumers[0].op_type != "Gather":
        raise ValueError(
            f"PLE initializer must feed exactly one Gather, found {[(node.name, node.op_type) for node in consumers]}"
        )
    gather = consumers[0]
    gather_axis = next(
        (int(helper.get_attribute_value(attribute)) for attribute in gather.attribute if attribute.name == "axis"),
        0,
    )
    if gather_axis != 0 or len(gather.input) != 2 or len(gather.output) != 1:
        raise ValueError(f"unsupported PLE Gather signature at {gather.name!r}")

    fields = _external_data_fields(initializer)
    source_offset = int(fields.get("offset", "0"))
    source_length = int(fields.get("length", "0"))
    element_size = np.dtype(helper.tensor_dtype_to_np_dtype(initializer.data_type)).itemsize
    row_size = math.prod(int(dimension) for dimension in initializer.dims[1:]) * element_size
    total_rows = int(initializer.dims[0])
    expected_length = total_rows * row_size
    if source_length != expected_length:
        raise ValueError(f"PLE external length mismatch: metadata={source_length}, expected={expected_length}")

    rows_per_shard, remainder = divmod(total_rows, shard_count)
    shard_rows = tuple(rows_per_shard + (1 if index < remainder else 0) for index in range(shard_count))
    shard_nbytes = tuple(rows * row_size for rows in shard_rows)
    shard_names = tuple(f"{initializer.name}.row_shard_{index}_of_{shard_count}" for index in range(shard_count))
    existing_names = {value.name for value in model.graph.initializer}
    if any(name in existing_names for name in shard_names):
        raise ValueError("PLE graph already contains row-sharded initializer names")

    shard_initializers: list[TensorProto] = []
    running_offset = source_offset
    for name, rows, nbytes in zip(shard_names, shard_rows, shard_nbytes, strict=True):
        shard_initializers.append(
            _make_external_shard(
                initializer,
                name=name,
                rows=rows,
                byte_offset=running_offset,
                byte_length=nbytes,
                source_model_path=input_path,
                output_model_path=output_path,
            )
        )
        running_offset += nbytes

    prefix = f"{gather.name or 'ple_gather'}_row_sharded"
    zero_name = f"{prefix}.zero"
    axes_name = f"{prefix}.axes"
    scalar_initializers = [
        numpy_helper.from_array(np.asarray(0, dtype=np.int32), name=zero_name),
        numpy_helper.from_array(np.asarray([-1], dtype=np.int64), name=axes_name),
    ]

    indices_name = gather.input[1]
    replacement_nodes: list[onnx.NodeProto] = []
    combined_output: str | None = None
    row_start = 0
    for index, (shard_name, rows) in enumerate(zip(shard_names, shard_rows, strict=True)):
        row_end = row_start + rows
        start_name = f"{prefix}.shard_{index}.start"
        end_name = f"{prefix}.shard_{index}.end"
        if index > 0:
            scalar_initializers.append(numpy_helper.from_array(np.asarray(row_start, dtype=np.int32), name=start_name))
        if index < shard_count - 1:
            scalar_initializers.append(numpy_helper.from_array(np.asarray(row_end, dtype=np.int32), name=end_name))

        mask_name = f"{prefix}.shard_{index}.mask"
        if index == 0:
            replacement_nodes.append(
                helper.make_node(
                    "Less",
                    [indices_name, end_name],
                    [mask_name],
                    name=f"{prefix}.shard_{index}.mask_upper",
                )
            )
            shifted_name = indices_name
        else:
            lower_mask_name = f"{prefix}.shard_{index}.lower_mask"
            replacement_nodes.append(
                helper.make_node(
                    "GreaterOrEqual",
                    [indices_name, start_name],
                    [lower_mask_name],
                    name=f"{prefix}.shard_{index}.mask_lower",
                )
            )
            if index == shard_count - 1:
                mask_name = lower_mask_name
            else:
                upper_mask_name = f"{prefix}.shard_{index}.upper_mask"
                replacement_nodes.extend(
                    [
                        helper.make_node(
                            "Less",
                            [indices_name, end_name],
                            [upper_mask_name],
                            name=f"{prefix}.shard_{index}.mask_upper",
                        ),
                        helper.make_node(
                            "And",
                            [lower_mask_name, upper_mask_name],
                            [mask_name],
                            name=f"{prefix}.shard_{index}.mask",
                        ),
                    ]
                )
            shifted_name = f"{prefix}.shard_{index}.shifted_indices"
            replacement_nodes.append(
                helper.make_node(
                    "Sub",
                    [indices_name, start_name],
                    [shifted_name],
                    name=f"{prefix}.shard_{index}.shift_indices",
                )
            )

        local_indices_name = f"{prefix}.shard_{index}.local_indices"
        output_name = f"{prefix}.shard_{index}.output"
        replacement_nodes.extend(
            [
                helper.make_node(
                    "Where",
                    [mask_name, shifted_name, zero_name],
                    [local_indices_name],
                    name=f"{prefix}.shard_{index}.select_indices",
                ),
                helper.make_node(
                    "Gather",
                    [shard_name, local_indices_name],
                    [output_name],
                    name=f"{prefix}.shard_{index}.gather",
                    axis=0,
                ),
            ]
        )
        stage_input_name = output_name
        if combined_output is not None:
            expanded_mask_name = f"{prefix}.shard_{index}.expanded_mask"
            combined_name = f"{prefix}.shard_{index}.combined"
            replacement_nodes.extend(
                [
                    helper.make_node(
                        "Unsqueeze",
                        [mask_name, axes_name],
                        [expanded_mask_name],
                        name=f"{prefix}.shard_{index}.expand_mask",
                    ),
                    helper.make_node(
                        "Where",
                        [expanded_mask_name, output_name, combined_output],
                        [combined_name],
                        name=f"{prefix}.shard_{index}.combine",
                    ),
                ]
            )
            stage_input_name = combined_name

        is_last = index == shard_count - 1
        tagged_output_name = gather.output[0] if is_last else f"{prefix}.shard_{index}.tagged_output"
        replacement_nodes.append(
            helper.make_node(
                "Tag",
                [stage_input_name],
                [tagged_output_name],
                name=gather.name if is_last else f"{prefix}.shard_{index}.stage",
                domain=_XH2A_DOMAIN,
                tag_name=f"qwen4_exp_ple_shard_{index}",
                tag_type="LLM",
                content=f"qwen4_exp_ple_shard_{index}",
            )
        )
        combined_output = tagged_output_name
        row_start = row_end

    node_index = next(index for index, node in enumerate(model.graph.node) if node is gather)
    nodes = list(model.graph.node)
    nodes[node_index : node_index + 1] = replacement_nodes
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    initializers = [value for value in model.graph.initializer if value is not initializer]
    initializers.extend(shard_initializers)
    initializers.extend(scalar_initializers)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    if not any(opset.domain == _XH2A_DOMAIN for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid(_XH2A_DOMAIN, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        onnx.save_model(model, temporary_path)
        onnx.checker.check_model(str(temporary_path))
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return PLEGatherShardingResult(
        input_path=str(input_path),
        output_path=str(output_path),
        initializer_name=initializer.name,
        gather_node_name=gather.name,
        shard_count=shard_count,
        shard_rows=shard_rows,
        shard_nbytes=shard_nbytes,
    )


def group_qwen4_exp_ple_shard_placement_tags(
    input_path: str | Path,
    output_path: str | Path,
    *,
    placement_group_size: int = 2,
) -> PLEShardTagGroupingResult:
    """Group adjacent PLE shards into one auto-offload allocation unit.

    Every row shard remains separated by a functional Tag. Only the internal
    placement classification changes: the final shard in each group keeps an
    ``LLM`` boundary, while intermediate shard tags use ``PLE``. This avoids
    wasting GPU capacity when one shard is smaller than the runtime's balanced
    per-device soft limit but two shards fit comfortably.
    """

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if placement_group_size < 1:
        raise ValueError(f"placement_group_size must be positive, got {placement_group_size}")

    model = onnx.load(str(input_path), load_external_data=False)
    shard_tags: dict[int, onnx.NodeProto] = {}
    pattern = re.compile(r"^qwen4_exp_ple_shard_(\d+)$")
    for node in model.graph.node:
        if node.op_type != "Tag" or node.domain != _XH2A_DOMAIN:
            continue
        attributes = {attribute.name: attribute for attribute in node.attribute}
        tag_name_attribute = attributes.get("tag_name")
        if tag_name_attribute is None:
            continue
        tag_name = helper.get_attribute_value(tag_name_attribute)
        if isinstance(tag_name, bytes):
            tag_name = tag_name.decode("utf-8")
        match = pattern.fullmatch(str(tag_name))
        if match:
            shard_tags[int(match.group(1))] = node

    if not shard_tags:
        raise ValueError("no Qwen4-Exp PLE shard placement tags were found")
    shard_count = max(shard_tags) + 1
    if set(shard_tags) != set(range(shard_count)):
        raise ValueError(f"PLE shard tag indices must be contiguous, got {sorted(shard_tags)}")

    llm_boundary_shards = tuple(
        index for index in range(shard_count) if (index + 1) % placement_group_size == 0 or index == shard_count - 1
    )
    for index, node in shard_tags.items():
        tag_type = "LLM" if index in llm_boundary_shards else "PLE"
        attribute = next(attribute for attribute in node.attribute if attribute.name == "tag_type")
        attribute.s = tag_type.encode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        onnx.save_model(model, temporary_path)
        onnx.checker.check_model(str(temporary_path))
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return PLEShardTagGroupingResult(
        input_path=str(input_path),
        output_path=str(output_path),
        shard_count=shard_count,
        placement_group_size=placement_group_size,
        llm_boundary_shards=llm_boundary_shards,
    )
