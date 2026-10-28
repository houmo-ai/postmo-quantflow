"""Wall-OSS through public build, source deletion, relocation and inference."""

import pytest
from _unified_cases import cases_for
from _unified_runner import check_case_result


CASE_SPEC = ("wall-oss-0.5", "wall_oss", "_unified_fixtures.wall_oss", ("full",))


@pytest.mark.parametrize("case", cases_for(__file__), ids=lambda case: case.id)
def test_public_build_relocate_and_inference(case, unified_runs):
    check_case_result(case, unified_runs)
