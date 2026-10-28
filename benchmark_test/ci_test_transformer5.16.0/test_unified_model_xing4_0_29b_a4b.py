# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Public build, relocation and inference for Xing4.0 MLA/mHC/MoE."""

import pytest
from _unified_cases import cases_for
from _unified_runner import check_case_result

CASE_SPEC = ("xing4.0-29b-a4b", "xing", "_unified_fixtures.xing", ("full",))


@pytest.mark.parametrize("case", cases_for(__file__), ids=lambda case: case.id)
def test_public_build_relocate_and_inference(case, unified_runs):
    check_case_result(case, unified_runs)
