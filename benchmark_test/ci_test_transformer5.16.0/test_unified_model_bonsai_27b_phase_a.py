# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Public Bonsai build, relocation and inference from a complete Phase A artifact."""
import pytest
from _unified_cases import cases_for
from _unified_runner import check_case_result

CASE_SPEC = ('bonsai-27b-phase-a', 'bonsai', '_unified_fixtures.bonsai', ('full',))


@pytest.mark.parametrize('case', cases_for(__file__), ids=lambda case: case.id)
def test_public_build_relocate_and_inference(case, unified_runs):
    check_case_result(case, unified_runs)
