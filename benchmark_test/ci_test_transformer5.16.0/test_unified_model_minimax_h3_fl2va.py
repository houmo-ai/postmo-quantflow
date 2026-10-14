# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Public build, relocation and inference for minimax-h3-fl2va."""
import pytest
from _unified_cases import cases_for
from _unified_runner import check_case_result

CASE_SPEC = ('minimax-h3-fl2va', 'minimax', '_unified_fixtures.minimax', ('full',))


@pytest.mark.parametrize("case", cases_for(__file__), ids=lambda case: case.id)
def test_public_build_relocate_and_inference(case, unified_runs):
    check_case_result(case, unified_runs)
