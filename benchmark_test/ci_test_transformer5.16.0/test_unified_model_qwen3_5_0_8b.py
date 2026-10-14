# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Public build, relocation and inference for qwen3.5-0.8b."""
import pytest
from _unified_cases import cases_for
from _unified_runner import check_case_result

CASE_SPEC = ('qwen3.5-0.8b', 'qwen_dense', '_unified_fixtures.transformer', ('full',))


@pytest.mark.parametrize("case", cases_for(__file__), ids=lambda case: case.id)
def test_public_build_relocate_and_inference(case, unified_runs):
    check_case_result(case, unified_runs)
