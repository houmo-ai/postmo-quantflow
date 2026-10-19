"""Exercise non-YAML precision producers and unified Workflow materialization."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import yaml

from xhmodel_merak.model_build.workflow import materialize_workflow_config
from xhmodel_merak.model_catalog.discovery import build_builtin_catalog

ROOT = Path(__file__).resolve().parents[2]
OPERANDS = {"q_bits", "k_bits", "v_bits", "p_bits"}


def _assert_no_score_bits(value):
    if isinstance(value, dict):
        assert not any(key.split(".")[-1] in {"s_bits", "s_cfg"} for key in value)
        for child in value.values():
            _assert_no_score_bits(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_score_bits(child)


@pytest.mark.parametrize("recipe", build_builtin_catalog().list_recipes(), ids=lambda recipe: recipe.model_id)
def test_every_unified_model_materializes_only_operand_precision(recipe, tmp_path):
    for variant in recipe.variants:
        configs = [variant.workflow_config, *(path for _, path in variant.context_workflows)]
        for index, relative in enumerate(configs):
            source = ROOT / relative
            plan = SimpleNamespace(
                model_id=recipe.model_id,
                config_path=str(source),
                source_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                config_replacements=dict(recipe.config_replacements),
            )
            output = tmp_path / f"{variant.name}-{index}.yaml"
            materialize_workflow_config(plan, output)
            _assert_no_score_bits(yaml.safe_load(output.read_text()))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("mtp", [False, True])
def test_qwen_next_cli_builds_main_and_draft_config_without_score_bits(bits, mtp):
    from examples_merak.llm.qwen3_next.qwen3_next_xh_export_hmonnx import build_config, build_parser

    argv = [
        "--model",
        "/unused/checkpoint",
        "--output-dir",
        "/unused/output",
        "--flash-attention",
        "--flash-attention-bits",
        str(bits),
    ]
    if mtp:
        argv.append("--mtp")
    config = build_config(build_parser().parse_args(argv))
    assert set(config.flash_attention) == OPERANDS | {"enable"}
    assert all(config.flash_attention[name] == bits for name in OPERANDS)
    if mtp:
        assert dict(config.mtp_config.flash_attention) == dict(config.flash_attention)


@pytest.mark.parametrize("bits", [8, 16])
def test_lingbot_cli_precision_override_has_only_four_operands(bits):
    from examples_merak.aigc.lingbot_video.lingbot_video_workflow import _config_overrides

    overrides = _config_overrides(SimpleNamespace(quant_type=f"w8a{bits}h1_sefp"))
    prefix = "export.components.transformer.flash_attention."
    actual = {key.removeprefix(prefix): value for key, value in overrides.items() if key.startswith(prefix)}
    assert actual == dict.fromkeys(OPERANDS, bits)


@pytest.mark.parametrize("bits", [8, 16])
def test_lingbot_embedded_comparison_defaults_use_four_operands(bits):
    # Execute the real embedded helper without importing the external LingBot checkpoint package.
    path = ROOT / "examples_merak/aigc/lingbot_video/compare_lingbot.py"
    tree = ast.parse(path.read_text())
    sources = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_LEGACY_SOURCES" for t in node.targets)
    )
    inner = ast.parse(sources["hmonnx_vs_front_blocks"])
    helper = next(node for node in inner.body if isinstance(node, ast.FunctionDef) and node.name == "_flash_bits")
    namespace = {"Any": object}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["_flash_bits"]({"quant_type": f"w8a{bits}h1_sefp"}) == dict.fromkeys(OPERANDS, bits)


def test_sefp16_audit_accepts_new_flash_attention_without_score_bits(tmp_path):
    import runpy

    audit = runpy.run_path(str(ROOT / "tools/audit_onnx_sefp16.py"))
    audit_model = audit["audit_model"]
    activation_attributes = audit["SEFP16_ACTIVATION_ATTRIBUTES"]

    nodes = [
        onnx.helper.make_node("Linear", ["x", "w"], ["q"], domain="ai.houmo.xh2a", **activation_attributes),
        onnx.helper.make_node(
            "FlashAttention", ["q", "k", "v", "table"], ["y"], domain="ai.houmo.xh2a", **dict.fromkeys(OPERANDS, 16)
        ),
    ]
    path = tmp_path / "attention.onnx"
    onnx.save(onnx.helper.make_model(onnx.helper.make_graph(nodes, "attention", [], [])), path)
    result = audit_model(path)
    assert result["passed"], result["violations"]
    assert "s_bits" not in result["flash_attention_nodes"][0]
