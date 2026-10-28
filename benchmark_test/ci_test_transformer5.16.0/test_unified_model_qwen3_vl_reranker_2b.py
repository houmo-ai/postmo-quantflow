# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Real small Qwen3-VL build, relocation and relevance scoring."""

import pytest
from _unified_cases import cases_for
from _unified_runner import check_case_result


CASE_SPEC = ("qwen3-vl-reranker-2b", "qwen3_vl_reranker", "_unified_fixtures.qwen3_vl_reranker", ("full",))


@pytest.mark.parametrize("case", cases_for(__file__), ids=lambda case: case.id)
def test_public_build_relocate_and_inference(case, unified_runs):
    check_case_result(case, unified_runs)
