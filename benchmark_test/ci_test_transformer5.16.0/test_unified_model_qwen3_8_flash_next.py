# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Qwen4 publication contracts; model export/runtime are explicit test boundaries.

Native numerics and SDK graph execution are tested by the Qwen4 model suites.
These CPU cases validate both public variants through publication and relocation.
"""
import pytest
import test_qwen4_exp_unified_api as qwen4_api
from _unified_cases import cases_for


source = qwen4_api.source

CASE_SPEC = ("qwen3.8-flash-next", "qwen4_exp", "test_qwen4_exp_unified_api", ("full", "mtp"))


@pytest.mark.parametrize("publication_case", cases_for(__file__), ids=lambda case: case.id)
def test_public_build_relocate_and_inference(publication_case, source, tmp_path, monkeypatch):
    qwen4_api.check_build_publish_relocate_and_dispatch(source, tmp_path, monkeypatch, variant=publication_case.variant)
