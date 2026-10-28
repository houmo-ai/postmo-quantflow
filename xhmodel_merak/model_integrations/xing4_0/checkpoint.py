"""Xing-owned validation of its fixed mixed-precision checkpoint regions."""

import re
from collections.abc import Mapping
from pathlib import Path

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_build.checkpoint import (
    _direct_weight_bits, _find_base_quant_bits, _is_float_quant_entry, _read_quant_metadata,
)


REGION_WEIGHT_BITS = {"attention": 8, "dense_mlp": 8, "shared_expert": 8, "routed_expert": 4}


def _quant_key_matches_region(key: str, component: str) -> bool:
    normalized = key.lower().replace("\\", "")
    if component == "routed_expert":
        return "routed_expert" in normalized or re.search(r"(?:^|\.)experts(?:\.|$)", normalized) is not None
    if component == "dense_mlp":
        return "dense_mlp" in normalized or (
            re.search(r"(?:^|\.)mlp\.", normalized) is not None
            and "expert" not in normalized
            and any(name in normalized for name in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"))
        )
    if component == "attention":
        return "attention" in normalized or "attn" in normalized
    return "shared_expert" in normalized if component == "shared_expert" else False


def _collect_region_bits(value, evidence):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(child, Mapping) and not _is_float_quant_entry(child):
                bits = _direct_weight_bits(child)
                if bits is not None:
                    for component in evidence:
                        if _quant_key_matches_region(str(key), component):
                            evidence[component].add(bits)
            _collect_region_bits(child, evidence)
    elif isinstance(value, list):
        for child in value:
            _collect_region_bits(child, evidence)


def verify_quant_regions(payloads, *, model_id, routed_bits=4):
    expected = {**REGION_WEIGHT_BITS, "routed_expert": routed_bits}
    evidence = {component: set() for component in expected}
    for payload in payloads:
        _collect_region_bits(payload, evidence)
    invalid = [
        {"component": component, "expected_bits": bits, "actual_bits": sorted(evidence[component])}
        for component, bits in expected.items() if evidence[component] != {bits}
    ]
    if invalid:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            f"Xing checkpoint does not prove its W8 attention/dense/shared and W{routed_bits} routed-expert regions",
            model=model_id, invalid_regions=invalid,
        )
    return {component: {"weight_bits": bits, "evidence": "explicit"} for component, bits in expected.items()}


def inspect_quant_regions(path, *, model_id="xing4.0-29b-a4b", routed_bits=4):
    payloads = _read_quant_metadata(Path(path), model_id)
    if routed_bits is None:
        declared = {_find_base_quant_bits(payload) for _, payload in payloads} - {None}
        if len(declared) != 1 or not declared.issubset({4, 8}):
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH, "Xing checkpoint requires one explicit W4/W8 base precision",
                model=model_id, actual_bits=sorted(declared),
            )
        routed_bits = declared.pop()
    return verify_quant_regions([payload for _, payload in payloads], model_id=model_id, routed_bits=routed_bits)
