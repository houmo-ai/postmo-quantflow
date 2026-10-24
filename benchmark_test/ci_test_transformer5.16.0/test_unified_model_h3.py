# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native H3 public build and source-free full/keep-LoRA inference."""

import pytest
from _unified_cases import cases_for
from _unified_runner import check_case_result


CASE_SPEC = ("minimax-h3", "h3", "_unified_fixtures.h3", ("full", "lora"))


@pytest.mark.parametrize("case", cases_for(__file__), ids=lambda case: case.id)
def test_public_build_relocate_and_inference(case, unified_runs):
    check_case_result(case, unified_runs)
