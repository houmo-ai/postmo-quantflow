#!/usr/bin/env python3
"""Fail-closed preflight for the HMONNX FlashAttention output ABI.

The XHQuant FlashAttention operator returns BHSD.  Model wrappers used by the
Merak artifacts in this repository consume BSHD, so every exported attention
result must have a statically visible BHSD shape and an immediate
``Transpose(perm=[0, 2, 1, 3])`` consumer.

Gemma4 Unified artifacts receive additional checks from
``docs/gemma4_12b_unified_merak_deploy.md``.  This tool intentionally loads
ONNX protobufs without external tensor data, so it is safe to run before a
large model is loaded onto a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import onnx


class PreflightError(RuntimeError):
    """An artifact does not satisfy the runtime ABI."""


def _tensor_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
        for dimension in value.type.tensor_type.shape.dim
    ]


def _node_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _find_stage_graph(model_dir: Path, stage: str) -> Path:
    stage_dir = model_dir / stage
    candidates = sorted(stage_dir.glob("*.onnx"))
    if not candidates:
        raise PreflightError(f"{stage_dir}: no ONNX graph")
    with_act = [path for path in candidates if path.name.endswith("_with_act.onnx")]
    if len(with_act) == 1:
        return with_act[0]
    if len(candidates) != 1:
        raise PreflightError(
            f"{stage_dir}: expected one unambiguous ONNX graph, found {candidates}"
        )
    return candidates[0]


def _check_attention_graph(graph_path: Path) -> dict[str, Any]:
    model = onnx.load(str(graph_path), load_external_data=False)
    input_names = {value.name for value in model.graph.input}
    if "sliding_attention_mask" in input_names:
        raise PreflightError(
            f"{graph_path}: legacy sliding_attention_mask input is not allowed"
        )

    values = {
        value.name: value
        for value in (
            list(model.graph.input)
            + list(model.graph.value_info)
            + list(model.graph.output)
        )
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)

    attention_nodes = [
        node
        for node in model.graph.node
        if node.op_type in {"FlashAttention", "PageAttention"}
    ]
    if not attention_nodes:
        raise PreflightError(f"{graph_path}: no FlashAttention/PageAttention nodes")

    shapes: set[tuple[int | str, ...]] = set()
    for index, node in enumerate(attention_nodes):
        fact = node.name or f"{node.op_type}[{index}]"
        attributes = _node_attributes(node)
        num_heads = attributes.get("num_heads")
        if type(num_heads) is not int or num_heads <= 0:
            raise PreflightError(
                f"{graph_path}: {fact} has invalid num_heads={num_heads!r}"
            )
        if not node.output or node.output[0] not in values:
            raise PreflightError(f"{graph_path}: {fact} output has no static ValueInfo")
        output_name = node.output[0]
        shape = _tensor_shape(values[output_name])
        if len(shape) != 4:
            raise PreflightError(
                f"{graph_path}: {fact} output shape={shape}, expected rank-4 BHSD"
            )
        if shape[1] != num_heads:
            observed = "BSHD" if shape[2] == num_heads else "unknown"
            raise PreflightError(
                f"{graph_path}: {fact} output shape={shape} is {observed}, "
                f"expected BHSD with shape[1]=num_heads={num_heads}"
            )

        output_consumers = consumers.get(output_name, ())
        valid_consumer = (
            len(output_consumers) == 1
            and output_consumers[0].op_type == "Transpose"
            and _node_attributes(output_consumers[0]).get("perm")
            == [0, 2, 1, 3]
        )
        if not valid_consumer:
            actual = [
                {
                    "op_type": consumer.op_type,
                    "name": consumer.name,
                    "perm": _node_attributes(consumer).get("perm"),
                }
                for consumer in output_consumers
            ]
            raise PreflightError(
                f"{graph_path}: {fact} BHSD output must have exactly one immediate "
                f"Transpose[0,2,1,3] consumer, found {actual}"
            )
        shapes.add(tuple(shape))

    return {
        "graph": str(graph_path),
        "attention_nodes": len(attention_nodes),
        "output_layout": "BHSD",
        "output_shapes": [list(shape) for shape in sorted(shapes, key=str)],
        "bhsd_to_bshd_transposes": len(attention_nodes),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{path}: cannot load JSON") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{path}: JSON root must be an object")
    return value


def _is_gemma4_unified(model_dir: Path, meta: dict[str, Any]) -> bool:
    if meta.get("variant") == "12b_unified":
        return True
    hf_config_path = model_dir / "hf_config" / "config.json"
    if not hf_config_path.is_file():
        return False
    hf_config = _load_json(hf_config_path)
    architectures = hf_config.get("architectures") or ()
    return "Gemma4UnifiedForConditionalGeneration" in architectures


def _check_exact_side_inputs(
    model_dir: Path,
    stage: str,
    expected: list[tuple[tuple[str, ...], list[int]]],
) -> dict[str, Any]:
    graph_path = _find_stage_graph(model_dir, stage)
    model = onnx.load(str(graph_path), load_external_data=False)
    inputs = list(model.graph.input)
    if len(inputs) != len(expected):
        raise PreflightError(
            f"{graph_path}: expected {len(expected)} inputs, found "
            f"{[(value.name, _tensor_shape(value)) for value in inputs]}"
        )
    facts: list[dict[str, Any]] = []
    for value, (aliases, expected_shape) in zip(inputs, expected, strict=True):
        actual_shape = _tensor_shape(value)
        if value.name not in aliases or actual_shape != expected_shape:
            raise PreflightError(
                f"{graph_path}: input {(value.name, actual_shape)} does not match "
                f"aliases={aliases}, shape={expected_shape}"
            )
        facts.append({"name": value.name, "shape": actual_shape})
    return {"graph": str(graph_path), "inputs": facts}


def _check_gemma4_unified(model_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    required = (
        "merak_config.json",
        "golden_meta_info.json",
        "hf_config",
        "visual",
        "video_visual",
        "audio",
        "prefill",
        "decode",
    )
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise PreflightError(
            f"{model_dir}: Gemma4 Unified deployment files are missing: {missing}"
        )

    if meta.get("attention_contract_version") != 2:
        raise PreflightError(
            f"{model_dir}: Gemma4 Unified requires attention_contract_version=2, "
            f"got {meta.get('attention_contract_version')!r}"
        )
    merak = _load_json(model_dir / "merak_config.json")
    xh_model = merak.get("xh_model")
    if not isinstance(xh_model, dict) or xh_model.get("model_type") != "hmonnx":
        raise PreflightError(
            f"{model_dir / 'merak_config.json'}: xh_model.model_type must be hmonnx"
        )

    model_config = meta.get("model_config")
    if not isinstance(model_config, dict):
        raise PreflightError(f"{model_dir}: metadata has no model_config")
    if model_config.get("prefill_chunk_length") != 320:
        raise PreflightError(
            f"{model_dir}: Gemma4 Unified prefill_chunk_length must be 320"
        )
    modality = meta.get("modality_contract")
    if not isinstance(modality, dict):
        modality = model_config.get("modality_contract")
    expected_modality = {
        "image_soft_tokens": 280,
        "video_soft_tokens_per_frame": 70,
        "audio_soft_tokens": 750,
        "vision_patch_dim": 6912,
        "audio_feature_dim": 640,
    }
    if not isinstance(modality, dict):
        raise PreflightError(f"{model_dir}: metadata has no modality_contract")
    mismatches = {
        key: {"actual": modality.get(key), "expected": expected}
        for key, expected in expected_modality.items()
        if modality.get(key) != expected
    }
    if mismatches:
        raise PreflightError(
            f"{model_dir}: Gemma4 Unified modality contract mismatch: {mismatches}"
        )

    position_aliases = (
        "position_ids",
        "normalized_position_ids",
        "pixel_position_ids",
    )
    return {
        "attention_contract_version": 2,
        "prefill_chunk_length": 320,
        "modality_contract": expected_modality,
        "visual": _check_exact_side_inputs(
            model_dir,
            "visual",
            [
                (("pixel_values",), [1, 280, 6912]),
                (position_aliases, [1, 280, 2]),
            ],
        ),
        "video_visual": _check_exact_side_inputs(
            model_dir,
            "video_visual",
            [
                (("pixel_values",), [1, 70, 6912]),
                (position_aliases, [1, 70, 2]),
            ],
        ),
        "audio": _check_exact_side_inputs(
            model_dir,
            "audio",
            [(("input_features",), [1, 750, 640])],
        ),
    }


def check_model(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve()
    meta_path = model_dir / "golden_meta_info.json"
    if not meta_path.is_file():
        raise PreflightError(f"{meta_path}: file does not exist")
    meta = _load_json(meta_path)
    result: dict[str, Any] = {
        "model_dir": str(model_dir),
        "prefill": _check_attention_graph(_find_stage_graph(model_dir, "prefill")),
        "decode": _check_attention_graph(_find_stage_graph(model_dir, "decode")),
    }
    if _is_gemma4_unified(model_dir, meta):
        result["gemma4_unified"] = _check_gemma4_unified(model_dir, meta)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object per artifact instead of compact text.",
    )
    args = parser.parse_args()

    failed = False
    for model_dir in args.model_dirs:
        try:
            result = check_model(model_dir)
        except PreflightError as exc:
            failed = True
            if args.json:
                print(
                    json.dumps(
                        {
                            "model_dir": str(model_dir.expanduser().resolve()),
                            "status": "FAIL",
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                print(f"FAIL {model_dir}: {exc}")
            continue
        if args.json:
            print(
                json.dumps(
                    {"status": "PASS", **result},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            prefill = result["prefill"]
            decode = result["decode"]
            suffix = " Gemma4-Unified-contract-v2" if "gemma4_unified" in result else ""
            print(
                f"PASS {model_dir}: BHSD+Transpose "
                f"prefill={prefill['attention_nodes']} "
                f"decode={decode['attention_nodes']}{suffix}"
            )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
