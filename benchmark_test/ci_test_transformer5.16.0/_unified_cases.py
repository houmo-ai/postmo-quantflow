# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Discover explicit cases without importing tests or production recipes."""
import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UnifiedCase:
    model_id: str
    variant: str
    fixture: str
    preparation: str
    owner: str

    @property
    def id(self):
        return f"{self.model_id}/{self.variant}"

    @property
    def node_id(self):
        return f"{self.owner}::test_public_build_relocate_and_inference[{self.id}]"


def discover_cases(directory):
    cases = []
    models = set()
    for path in sorted(Path(directory).glob("test_unified_model_*.py")):
        declarations = [node.value for node in ast.parse(path.read_text(), filename=str(path)).body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "CASE_SPEC" for target in node.targets)]
        if len(declarations) != 1:
            raise ValueError(f"{path}: expected one literal CASE_SPEC")
        spec = ast.literal_eval(declarations[0])
        if not isinstance(spec, tuple) or len(spec) != 4:
            raise ValueError(f"{path}: CASE_SPEC must be (model_id, fixture, preparation, variants)")
        model_id, fixture, preparation, variants = spec
        if (not all(isinstance(value, str) and value for value in spec[:3])
                or not isinstance(variants, tuple) or not variants
                or not all(isinstance(value, str) and value for value in variants)):
            raise ValueError(f"{path}: invalid CASE_SPEC")
        if model_id in models or len(set(variants)) != len(variants):
            raise ValueError(f"{path}: duplicate model or variant declaration")
        models.add(model_id)
        cases.extend(UnifiedCase(model_id, variant, fixture, preparation, path.name) for variant in variants)
    if not cases:
        raise ValueError(f"{directory}: no unified model cases")
    return tuple(cases)


CASES = discover_cases(Path(__file__).parent)
MODELS = {case.model_id: (case.fixture, tuple(c.variant for c in CASES if c.model_id == case.model_id))
          for case in CASES}


def cases_for(path):
    return tuple(case for case in CASES if case.owner == Path(path).name)
