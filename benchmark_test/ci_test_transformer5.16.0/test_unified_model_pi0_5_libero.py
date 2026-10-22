# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Public build, relocation and inference for pi0.5-libero."""
import pytest
from _unified_cases import cases_for
from _unified_runner import check_case_result

CASE_SPEC = ('pi0.5-libero', 'pi05', '_unified_fixtures.pi05', ('lerobot-h50',))


@pytest.mark.parametrize("case", cases_for(__file__), ids=lambda case: case.id)
def test_public_build_relocate_and_inference(case, unified_runs):
    check_case_result(case, unified_runs)
    import onnx

    _, folder = unified_runs[case.id].result()
    protected = set()
    for path in (folder / "relocated").rglob("*.onnx"):
        graph = onnx.load(path, load_external_data=False)
        for node in graph.graph.node:
            if node.op_type != "Linear":
                continue
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if "hmfp_act_man_bit" not in attrs:
                continue
            action = next((name for name in ("action_in_proj", "action_out_proj") if name in path.name), None)
            first_down = "expert" in path.name and node.input[1] == "model.layers.0.mlp.down_proj.qweight"
            assert attrs["hmfp_act_man_bit"] == (16 if action or first_down else 8), (path, node.name)
            assert attrs["hmfp_weight_man_bit"] == 8, (path, node.name)
            if action or first_down:
                protected.add(action or "expert_first_down")
    assert protected == {"action_in_proj", "action_out_proj", "expert_first_down"}
