import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from vllm_merak.audit_logging import (
    AUDIT_LOG_DIR_ENV,
    AUDIT_LOG_MIDDLEWARE,
    AUDIT_LOG_MODE_ENV,
)
from vllm_merak.constants import MERAK_AUTO_OFFLOAD, MERAK_CUDA_GRAPH, MERAK_USE_V2


_SERVER_PATH = Path(__file__).resolve().parents[1] / "examples" / "vllm_chat_server.py"
_START_SCRIPT = _SERVER_PATH.parent / "vllm_server_start.sh"
_SERVER_SPEC = importlib.util.spec_from_file_location(
    "vllm_chat_server_under_test", _SERVER_PATH
)
assert _SERVER_SPEC is not None and _SERVER_SPEC.loader is not None
_SERVER_MODULE = importlib.util.module_from_spec(_SERVER_SPEC)
_SERVER_SPEC.loader.exec_module(_SERVER_MODULE)
apply_merak_runtime_flags = _SERVER_MODULE.apply_merak_runtime_flags
build_arg_parser = _SERVER_MODULE.build_arg_parser


def _run_start_script(
    tmp_path: Path, **overrides: str
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"${VLLM_CPU_KVCACHE_SPACE-}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.pop("CPU_KV", None)
    env.pop("VLLM_CPU_KVCACHE_SPACE", None)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.update(overrides)
    return subprocess.run(
        ["bash", str(_START_SCRIPT), "/tmp/model"],
        cwd=_SERVER_PATH.parents[1],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_vllm_chat_server_accepts_deploy_doc_arguments() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--model",
            "/tmp/model",
            "--host",
            "127.0.0.1",
            "--port",
            "18006",
            "--served-model-name",
            "qwen3_5_9b_merak",
            "--trust-remote-code",
            "--config-format",
            "merak_llm",
            "--load-format",
            "merak_llm",
            "--enforce-eager",
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "1",
            "--use-v2",
            "--auto-offload",
            "--cuda-graph",
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_xml",
            "--generation-config",
            "vllm",
            "--audit-log-mode",
            "both",
            "--audit-log-dir",
            "/tmp/vllm-audit",
        ]
    )

    assert args.model == "/tmp/model"
    assert args.port == 18006
    assert args.served_model_name == ["qwen3_5_9b_merak"]
    assert args.config_format == "merak_llm"
    assert args.load_format == "merak_llm"
    assert args.max_model_len == 2048
    assert args.max_num_seqs == 1
    assert args.enforce_eager is True
    assert args.use_v2 is True
    assert args.auto_offload is True
    assert args.cuda_graph is True
    assert args.enable_auto_tool_choice is True
    assert args.tool_call_parser == "qwen3_xml"
    assert args.reasoning_parser == "qwen3"
    assert args.generation_config == "vllm"
    assert args.default_chat_template_kwargs is None
    assert args.audit_log_mode == "both"
    assert args.audit_log_dir == "/tmp/vllm-audit"


def test_vllm_chat_server_audit_logging_is_off_by_default() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--model", "/tmp/model"])

    assert args.audit_log_mode == "off"
    assert args.audit_log_dir == "logs/audit"


def test_vllm_chat_server_defaults_to_qwen3_reasoning_parser() -> None:
    parser = build_arg_parser()
    args = SimpleNamespace(
        model="/tmp/model",
        model_tag=None,
        use_v2=False,
        auto_offload=False,
        cuda_graph=False,
        reasoning_parser="",
    )

    apply_merak_runtime_flags(args, parser)

    assert args.reasoning_parser == "qwen3"


def test_vllm_chat_server_does_not_apply_qwen_parser_to_gemma4(
    tmp_path: Path,
) -> None:
    (tmp_path / "merak_config.json").write_text(
        '{"xh_model":{"meta_info":"golden_meta_info.json"}}',
        encoding="utf-8",
    )
    (tmp_path / "golden_meta_info.json").write_text(
        '{"model_config":{"model_type":"Gemma4ForConditionalGeneration"}}',
        encoding="utf-8",
    )
    parser = build_arg_parser()
    args = SimpleNamespace(
        model=str(tmp_path),
        model_tag=None,
        use_v2=False,
        auto_offload=False,
        cuda_graph=False,
        reasoning_parser="",
    )

    apply_merak_runtime_flags(args, parser)

    assert args.reasoning_parser == ""


def test_vllm_chat_server_preserves_explicit_reasoning_parser() -> None:
    parser = build_arg_parser()
    args = SimpleNamespace(
        model="/tmp/model",
        model_tag=None,
        use_v2=False,
        auto_offload=False,
        cuda_graph=False,
        reasoning_parser="deepseek_r1",
    )

    apply_merak_runtime_flags(args, parser)

    assert args.reasoning_parser == "deepseek_r1"


def test_vllm_chat_server_preserves_explicit_chat_template_kwargs() -> None:
    parser = build_arg_parser()
    args = SimpleNamespace(
        model="/tmp/model",
        model_tag=None,
        use_v2=False,
        auto_offload=False,
        cuda_graph=False,
        default_chat_template_kwargs={"enable_thinking": True},
    )

    apply_merak_runtime_flags(args, parser)

    assert args.default_chat_template_kwargs == {"enable_thinking": True}


def test_vllm_chat_server_sets_merak_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_name in (MERAK_USE_V2, MERAK_AUTO_OFFLOAD, MERAK_CUDA_GRAPH):
        monkeypatch.delenv(env_name, raising=False)
    parser = build_arg_parser()
    args = SimpleNamespace(
        model="/tmp/model",
        model_tag=None,
        use_v2=True,
        auto_offload=True,
        cuda_graph=True,
    )

    apply_merak_runtime_flags(args, parser)

    assert args.model == "/tmp/model"
    assert {
        env_name: os.environ[env_name]
        for env_name in (MERAK_USE_V2, MERAK_AUTO_OFFLOAD, MERAK_CUDA_GRAPH)
    } == {
        MERAK_USE_V2: "1",
        MERAK_AUTO_OFFLOAD: "1",
        MERAK_CUDA_GRAPH: "1",
    }


def test_vllm_chat_server_installs_audit_middleware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(AUDIT_LOG_MODE_ENV, raising=False)
    monkeypatch.delenv(AUDIT_LOG_DIR_ENV, raising=False)
    parser = build_arg_parser()
    args = SimpleNamespace(
        model="/tmp/model",
        model_tag=None,
        use_v2=False,
        auto_offload=False,
        cuda_graph=False,
        audit_log_mode="both",
        audit_log_dir="/tmp/audit",
        middleware=[],
    )

    apply_merak_runtime_flags(args, parser)

    assert os.environ[AUDIT_LOG_MODE_ENV] == "both"
    assert os.environ[AUDIT_LOG_DIR_ENV] == "/tmp/audit"
    assert args.middleware == [AUDIT_LOG_MIDDLEWARE]


def test_vllm_chat_server_does_not_install_audit_middleware_when_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(AUDIT_LOG_MODE_ENV, raising=False)
    monkeypatch.delenv(AUDIT_LOG_DIR_ENV, raising=False)
    parser = build_arg_parser()
    args = SimpleNamespace(
        model="/tmp/model",
        model_tag=None,
        use_v2=False,
        auto_offload=False,
        cuda_graph=False,
        audit_log_mode="off",
        audit_log_dir="/tmp/audit",
        middleware=[],
    )

    apply_merak_runtime_flags(args, parser)

    assert AUDIT_LOG_MODE_ENV not in os.environ
    assert AUDIT_LOG_DIR_ENV not in os.environ
    assert args.middleware == []


def test_vllm_chat_server_rejects_auto_offload_cuda_graph_without_v2() -> None:
    parser = build_arg_parser()
    args = SimpleNamespace(
        model="/tmp/model",
        model_tag=None,
        use_v2=False,
        auto_offload=True,
        cuda_graph=True,
    )

    with pytest.raises(SystemExit):
        apply_merak_runtime_flags(args, parser)


def test_vllm_current_environment_registers_qwen3_xml_tool_parser() -> None:
    from vllm.tool_parsers import ToolParserManager

    registered = set(ToolParserManager.list_registered())
    assert "qwen3_xml" in registered
    assert "qwen3_coder" in registered
    assert ToolParserManager.get_tool_parser(
        "qwen3_xml"
    ) is ToolParserManager.get_tool_parser("qwen3_coder")


def test_vllm_current_environment_registers_qwen3_reasoning_parser() -> None:
    from vllm.reasoning import ReasoningParserManager

    registered = set(ReasoningParserManager.list_registered())
    assert "qwen3" in registered


def test_example_server_start_script_matches_deploy_doc_command() -> None:
    script = (_SERVER_PATH.parent / "vllm_server_start.sh").read_text(encoding="utf-8")
    docs = (_SERVER_PATH.parents[1] / "docs" / "qwen3_5_9b_merak_deploy.md").read_text(
        encoding="utf-8"
    )

    for required in (
        '--model "$model_id"',
        '--port "$PORT"',
        "--served-model-name qwen3_5_9b_merak",
        '--max-model-len "$MAX_MODEL_LEN"',
        '--max-num-seqs "$MAX_NUM_SEQS"',
        "--use-v2",
        "--auto-offload",
        "--cuda-graph",
        "--reasoning-parser qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser qwen3_xml",
        "--generation-config vllm",
        '--audit-log-mode "$AUDIT_LOG_MODE"',
        '--audit-log-dir "$AUDIT_LOG_DIR"',
    ):
        assert required in script
    assert "MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}" in script
    assert "AUDIT_LOG_MODE=${AUDIT_LOG_MODE:-off}" in script
    assert "AUDIT_LOG_DIR=${AUDIT_LOG_DIR:-logs/audit}" in script
    assert "VLLM_CPU_KVCACHE_SPACE=16" not in script
    assert 'if [[ -n "${CPU_KV+x}" ]]' in script
    assert 'elif [[ -n "${VLLM_CPU_KVCACHE_SPACE+x}" ]]' in script
    assert "conflicts with VLLM_CPU_KVCACHE_SPACE" in script
    assert "unset VLLM_CPU_KVCACHE_SPACE" not in script
    assert "--default-chat-template-kwargs" not in script
    assert "--enforce-eager" in script
    assert "mutually exclusive" not in script
    assert "--reasoning-parser qwen3" in docs
    assert "--generation-config vllm" in docs
    assert "AUDIT_LOG_MODE=both" in docs
    assert "audit-aggregate-YYYYMMDD-p<PID>.jsonl" in docs
    assert "audit-sse-YYYYMMDD-p<PID>.jsonl" in docs
    assert "不需要设置 `VLLM_LOGGING_LEVEL=DEBUG`" in docs
    assert "python -m vllm_merak.audit_log_viewer" in docs
    assert "响应重写 / alias 中间件" in docs
    assert "默认规划 4304 个全局 block" in docs
    assert "Phase 1 不允许把 `--max-num-seqs` 调到 4 或 8" in docs


def test_qwen36_start_script_exposes_audit_logging_modes() -> None:
    script = (_SERVER_PATH.parent / "vllm_server_start_qwen36_35b.sh").read_text(
        encoding="utf-8"
    )
    docs = (
        _SERVER_PATH.parents[1] / "docs" / "qwen3_6_35b_a3b_merak_deploy.md"
    ).read_text(encoding="utf-8")

    assert "AUDIT_LOG_MODE=${AUDIT_LOG_MODE:-off}" in script
    assert "AUDIT_LOG_DIR=${AUDIT_LOG_DIR:-logs/audit}" in script
    assert '--audit-log-mode "$AUDIT_LOG_MODE"' in script
    assert '--audit-log-dir "$AUDIT_LOG_DIR"' in script
    assert "VLLM_LOGGING_LEVEL=DEBUG" not in script
    assert "AUDIT_LOG_MODE=both" in docs
    assert "AUDIT_LOG_DIR=logs/audit_qwen36" in docs
    assert "python -m vllm_merak.audit_log_viewer" in docs
    assert "问题定位结束后应恢复 `AUDIT_LOG_MODE=off`" in docs


def test_example_server_start_preserves_standard_virtual_budget_override(
    tmp_path: Path,
) -> None:
    result = _run_start_script(tmp_path, VLLM_CPU_KVCACHE_SPACE="48")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "48"
    assert "VLLM_CPU_KVCACHE_SPACE=48GiB-explicit-virtual-override" in result.stderr


def test_example_server_start_rejects_conflicting_virtual_budget_aliases(
    tmp_path: Path,
) -> None:
    result = _run_start_script(
        tmp_path,
        CPU_KV="8",
        VLLM_CPU_KVCACHE_SPACE="48",
    )

    assert result.returncode == 2
    assert "CPU_KV (8) conflicts with VLLM_CPU_KVCACHE_SPACE (48)" in result.stderr
    assert result.stdout == ""
