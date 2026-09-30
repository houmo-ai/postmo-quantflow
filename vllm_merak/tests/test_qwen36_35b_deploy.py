import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "examples" / "vllm_server_start_qwen36_35b.sh"
_BENCHMARK = _ROOT / "tools" / "bench_openai_stream.py"
_DOC = _ROOT / "docs" / "qwen3_6_35b_a3b_merak_deploy.md"
_EXPORT_NAME = "hmquant_xh2_qwen3_6_35b_a3b_w4a8_256_256k_mpe256k_448x448_20260710"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "bench_openai_stream_under_test", _BENCHMARK
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_launcher(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "merak_config.json").write_text("{}\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'printf \'{"cuda":"%s","cpu_kv":"%s","argv":[\' '
        '"$CUDA_VISIBLE_DEVICES" "${VLLM_CPU_KVCACHE_SPACE-}"\n'
        "sep=''\n"
        'for arg in "$@"; do\n'
        "  escaped=$(printf '%s' \"$arg\" | sed 's/\\\\/\\\\\\\\/g; s/\"/\\\\\"/g')\n"
        '  printf \'%s"%s"\' "$sep" "$escaped"\n'
        "  sep=','\n"
        "done\n"
        "printf ']}\\n'\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.pop("CPU_KV", None)
    env.pop("VLLM_CPU_KVCACHE_SPACE", None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "MODEL_DIR": str(model_dir),
        }
    )
    env.update(overrides)
    return subprocess.run(
        ["bash", str(_SCRIPT)],
        cwd=_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_qwen36_launcher_defaults_to_two_cards_and_smoke_limits(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path, CUDA_VISIBLE_DEVICES="")

    assert result.returncode == 0, result.stderr
    invocation = json.loads(result.stdout)
    assert invocation["cuda"] == "0,1"
    assert invocation["cpu_kv"] == ""
    assert invocation["argv"][invocation["argv"].index("--model") + 1] == str(
        tmp_path / "model"
    )
    assert "--max-model-len" in invocation["argv"]
    assert invocation["argv"][invocation["argv"].index("--max-model-len") + 1] == "262144"
    assert invocation["argv"][invocation["argv"].index("--max-num-seqs") + 1] == "1"
    assert (
        invocation["argv"][invocation["argv"].index("--served-model-name") + 1]
        == "qwen3_6_35b_a3b_merak"
    )
    assert "--use-v2" in invocation["argv"]
    assert "--auto-offload" in invocation["argv"]
    assert "--cuda-graph" in invocation["argv"]
    assert "--tensor-parallel-size" not in invocation["argv"]
    assert "card_count=2" in result.stderr
    assert "VIRTUAL_KV=planner-default" in result.stderr


def test_qwen36_launcher_and_doc_target_latest_export() -> None:
    script = _SCRIPT.read_text(encoding="utf-8")
    docs = _DOC.read_text(encoding="utf-8")

    assert _EXPORT_NAME in script
    assert _EXPORT_NAME in docs
    assert 'cp -n data/merak_config.json "$MODEL_DIR/merak_config.json"' in docs
    assert "20260709" not in script
    assert "20260709" not in docs


def test_qwen36_doc_uses_phase1_planner_and_resource_evidence() -> None:
    docs = _DOC.read_text(encoding="utf-8")

    assert "Phase 1 支持恰好 1、2 或 3 张唯一 GPU" in docs
    assert "实际大小 cache backing" in docs
    assert "无论使用 1、2 还是 3 卡，都不要添加 `--tensor-parallel-size`" in docs
    assert "2k smoke 默认使用容量 planner" in docs
    assert "默认容量 planner 规划 4304 个全局 block" in docs
    assert "旧的 `CPU_KV=48` GiB 结果" in docs
    assert "不再是推荐默认值" in docs
    assert "CPU_KV=96" not in docs
    for resource in (
        "CUDA_VISIBLE_DEVICES",
        "card_count",
        "MAX_MODEL_LEN",
        "MAX_NUM_SEQS",
        "VIRTUAL_KV",
    ):
        assert resource in docs
    assert "HTTP 请求" in docs
    assert "排队串行执行" in docs
    assert "request-local" in docs
    assert "提升 `MAX_NUM_SEQS` 前" in docs


def test_qwen36_doc_distinguishes_final_benchmark_from_pre_fix_baseline() -> None:
    docs = _DOC.read_text(encoding="utf-8")

    assert "7.3265 tok/s" in docs
    assert "11.260519 tok/s" in docs
    assert "formal_tuned_35b_1800.jsonl" in docs
    assert "35B 使用双卡 `0,1`，9B 使用单卡 `6`" in docs
    assert "不能称同资源模型优势" in docs
    assert "8000 中途停止且没有有效 summary" in docs


def test_qwen36_launcher_honors_supported_environment_overrides(tmp_path: Path) -> None:
    result = _run_launcher(
        tmp_path,
        CUDA_VISIBLE_DEVICES="2,4,7",
        PORT="19001",
        MAX_MODEL_LEN="262144",
        MAX_NUM_SEQS="1",
        CPU_KV="48",
    )

    assert result.returncode == 0, result.stderr
    invocation = json.loads(result.stdout)
    assert invocation["cuda"] == "2,4,7"
    assert invocation["cpu_kv"] == "48"
    assert invocation["argv"][invocation["argv"].index("--port") + 1] == "19001"
    assert (
        invocation["argv"][invocation["argv"].index("--max-model-len") + 1] == "262144"
    )
    assert invocation["argv"][invocation["argv"].index("--max-num-seqs") + 1] == "1"


def test_qwen36_launcher_honors_served_model_name_override(tmp_path: Path) -> None:
    result = _run_launcher(
        tmp_path,
        SERVED_MODEL_NAME="qwen3_6_35b_a3b_custom",
    )

    assert result.returncode == 0, result.stderr
    invocation = json.loads(result.stdout)
    assert (
        invocation["argv"][invocation["argv"].index("--served-model-name") + 1]
        == "qwen3_6_35b_a3b_custom"
    )


def test_qwen36_launcher_accepts_one_visible_card(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path, CUDA_VISIBLE_DEVICES="5")

    assert result.returncode == 0, result.stderr
    invocation = json.loads(result.stdout)
    assert invocation["cuda"] == "5"
    assert "card_count=1" in result.stderr
    assert "--tensor-parallel-size" not in invocation["argv"]


def test_qwen36_launcher_preserves_standard_virtual_budget_override(
    tmp_path: Path,
) -> None:
    result = _run_launcher(
        tmp_path,
        CUDA_VISIBLE_DEVICES="2,4",
        VLLM_CPU_KVCACHE_SPACE="48",
    )

    assert result.returncode == 0, result.stderr
    invocation = json.loads(result.stdout)
    assert invocation["cpu_kv"] == "48"
    assert "VLLM_CPU_KVCACHE_SPACE=48GiB-explicit-virtual-override" in result.stderr


def test_qwen36_launcher_rejects_conflicting_virtual_budget_aliases(
    tmp_path: Path,
) -> None:
    result = _run_launcher(
        tmp_path,
        CUDA_VISIBLE_DEVICES="2,4",
        CPU_KV="8",
        VLLM_CPU_KVCACHE_SPACE="48",
    )

    assert result.returncode == 2
    assert "CPU_KV (8) conflicts with VLLM_CPU_KVCACHE_SPACE (48)" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("max_num_seqs", ("0", "2", "3", "01", "unbounded"))
def test_qwen36_launcher_rejects_non_serial_max_num_seqs(
    tmp_path: Path, max_num_seqs: str
) -> None:
    result = _run_launcher(tmp_path, MAX_NUM_SEQS=max_num_seqs)

    assert result.returncode != 0
    assert "MAX_NUM_SEQS must be exactly 1" in result.stderr
    assert "fixed-address CUDA Graph metadata staging" in result.stderr


@pytest.mark.parametrize(
    "devices",
    (
        "5,5",
        "0,,1",
        "gpu0,gpu1",
        "0,1,2,3",
        "-1",
        "01",
        "0,01",
        "1, 2",
        "1,",
        ",1",
    ),
)
def test_qwen36_launcher_rejects_invalid_visible_card_lists(
    tmp_path: Path, devices: str
) -> None:
    result = _run_launcher(tmp_path, CUDA_VISIBLE_DEVICES=devices)

    assert result.returncode != 0
    assert "exactly 1, 2, or 3 unique non-negative integers" in result.stderr


def test_qwen36_launcher_requires_merak_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    env = os.environ.copy()
    env.update({"MODEL_DIR": str(model_dir), "CUDA_VISIBLE_DEVICES": "0,1"})

    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        cwd=_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "merak_config.json" in result.stderr


def test_benchmark_parser_supports_formal_run_arguments() -> None:
    benchmark = _load_benchmark()

    args = benchmark.build_parser().parse_args(
        [
            "--base-url",
            "http://127.0.0.1:18036/v1",
            "--model",
            "qwen3_6_35b_a3b_merak",
            "--model-dir",
            "/models/qwen36",
            "--target-prompt-tokens",
            "2048",
            "--max-tokens",
            "128",
            "--warmup",
            "2",
            "--repeat",
            "5",
        ]
    )

    assert args.base_url == "http://127.0.0.1:18036/v1"
    assert args.model == "qwen3_6_35b_a3b_merak"
    assert args.model_dir == Path("/models/qwen36")
    assert args.tokenizer is None
    assert args.target_prompt_tokens == 2048
    assert args.max_tokens == 128
    assert args.warmup == 2
    assert args.repeat == 5


def test_benchmark_uses_explicit_tokenizer_or_model_hf_config() -> None:
    benchmark = _load_benchmark()

    assert benchmark.resolve_tokenizer_source("org/tokenizer", None) == "org/tokenizer"
    assert benchmark.resolve_tokenizer_source(None, Path("/models/qwen36")) == str(
        Path("/models/qwen36/hf_config")
    )


def test_benchmark_decode_rate_excludes_first_completion_token() -> None:
    benchmark = _load_benchmark()

    metrics = benchmark.calculate_metrics(
        started_at=10.0,
        first_token_at=12.0,
        last_token_at=16.0,
        finished_at=30.0,
        completion_tokens=21,
    )

    assert metrics["ttft_s"] == 2.0
    assert metrics["total_s"] == 20.0
    assert metrics["decode_tokens"] == 20
    assert metrics["decode_time_s"] == 4.0
    assert metrics["decode_tok_s"] == 5.0


def test_benchmark_prompt_builder_rejects_inexact_local_token_count() -> None:
    benchmark = _load_benchmark()

    class EvenLengthTokenizer:
        @staticmethod
        def encode(text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [0] * (len(text) * 2)

    with pytest.raises(ValueError, match="cannot build exactly 31 prompt tokens"):
        benchmark.build_prompt(EvenLengthTokenizer(), 31, "x")


def test_benchmark_request_ignores_eos_for_stable_completion_length() -> None:
    benchmark = _load_benchmark()

    body = benchmark.build_request_body(
        model="qwen3_6_35b_a3b_merak",
        prompt="test prompt",
        max_tokens=128,
    )

    assert body["ignore_eos"] is True
    assert body["max_tokens"] == 128
    assert body["stream_options"] == {"include_usage": True}


def test_benchmark_summary_reports_median_and_p95_for_measured_rounds() -> None:
    benchmark = _load_benchmark()
    runs = [
        {
            "ttft_s": 1.0,
            "decode_tok_s": 10.0,
            "target_prompt_tokens": 2048,
            "prompt_tokens_local": 2048,
            "prompt_tokens_usage": 2048,
        },
        {
            "ttft_s": 2.0,
            "decode_tok_s": 20.0,
            "target_prompt_tokens": 2048,
            "prompt_tokens_local": 2048,
            "prompt_tokens_usage": 2048,
        },
        {
            "ttft_s": 3.0,
            "decode_tok_s": 30.0,
            "target_prompt_tokens": 2048,
            "prompt_tokens_local": 2048,
            "prompt_tokens_usage": 2048,
        },
    ]

    summary = benchmark.summarize(runs)

    assert summary == {
        "type": "summary",
        "repeat": 3,
        "target_prompt_tokens": 2048,
        "prompt_tokens_local": 2048,
        "prompt_tokens_usage": 2048,
        "ttft_s": {"median": 2.0, "p95": 2.9},
        "decode_tok_s": {"median": 20.0, "p95": 29.0},
    }


def test_benchmark_jsonl_labels_warmups_and_excludes_them_from_summary() -> None:
    benchmark = _load_benchmark()
    args = SimpleNamespace(
        base_url="http://127.0.0.1:18036/v1",
        model="qwen3_6_35b_a3b_merak",
        target_prompt_tokens=64,
        max_tokens=16,
        warmup=2,
        repeat=3,
        unit="x",
        timeout=30.0,
    )

    class CharacterTokenizer:
        @staticmethod
        def encode(text: str, *, add_special_tokens: bool) -> list[str]:
            assert add_special_tokens is False
            return list(text)

    decode_rates = iter((100.0, 200.0, 10.0, 20.0, 30.0))

    def fake_benchmark_once(**kwargs):
        assert kwargs["max_tokens"] == 16
        rate = next(decode_rates)
        return {
            "ttft_s": rate / 10,
            "decode_tok_s": rate,
            "target_prompt_tokens": 64,
            "prompt_tokens_local": 64,
            "prompt_tokens_usage": 64,
        }

    output = io.StringIO()
    benchmark.run_benchmark(
        args,
        CharacterTokenizer(),
        benchmark_fn=fake_benchmark_once,
        output=output,
    )
    records = [json.loads(line) for line in output.getvalue().splitlines()]

    assert [(record["phase"], record["round"]) for record in records[:-1]] == [
        ("warmup", 1),
        ("warmup", 2),
        ("measured", 1),
        ("measured", 2),
        ("measured", 3),
    ]
    assert records[-1] == {
        "type": "summary",
        "repeat": 3,
        "target_prompt_tokens": 64,
        "prompt_tokens_local": 64,
        "prompt_tokens_usage": 64,
        "ttft_s": {"median": 2.0, "p95": 2.9},
        "decode_tok_s": {"median": 20.0, "p95": 29.0},
    }


class _CharacterTokenizer:
    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[str]:
        assert add_special_tokens is False
        return list(text)


class _FakeStreamingResponse:
    def __init__(self, events: list[str]) -> None:
        self._lines = [f"data: {event}\n".encode() for event in events]

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def __iter__(self):
        return iter(self._lines)


def _run_fake_stream(monkeypatch: pytest.MonkeyPatch, events: list[dict | str]):
    benchmark = _load_benchmark()
    encoded_events = [
        event if isinstance(event, str) else json.dumps(event) for event in events
    ]
    monkeypatch.setattr(
        benchmark,
        "urlopen",
        lambda request, timeout: _FakeStreamingResponse(encoded_events),
    )
    return benchmark.benchmark_once(
        base_url="http://127.0.0.1:18036/v1",
        model="qwen3_6_35b_a3b_merak",
        tokenizer=_CharacterTokenizer(),
        prompt="0123456789",
        prompt_tokens_local=10,
        target_prompt_tokens=10,
        max_tokens=4,
        timeout=30,
    )


def _text_event(text: str, finish_reason=None) -> dict:
    return {"choices": [{"text": text, "finish_reason": finish_reason}]}


def _usage_event(*, prompt_tokens: int = 10, completion_tokens: int = 4) -> dict:
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def test_benchmark_stream_records_complete_length_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_fake_stream(
        monkeypatch,
        [
            _text_event("a"),
            _text_event("b"),
            _text_event("", "length"),
            _usage_event(),
            "[DONE]",
        ],
    )

    assert result["finish_reason"] == "length"
    assert result["completion_tokens"] == 4
    assert result["prompt_tokens_usage"] == 10


def test_benchmark_stream_raises_on_error_after_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="server error.*failed"):
        _run_fake_stream(
            monkeypatch,
            [_text_event("a"), {"error": {"message": "failed"}}],
        )


def test_benchmark_stream_requires_done_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match=r"missing \[DONE\]"):
        _run_fake_stream(
            monkeypatch,
            [_text_event("a"), _text_event("", "length"), _usage_event()],
        )


def test_benchmark_stream_requires_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="missing usage"):
        _run_fake_stream(
            monkeypatch,
            [_text_event("a"), _text_event("", "length"), "[DONE]"],
        )


def test_benchmark_stream_rejects_short_ignore_eos_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="completion_tokens=3.*max_tokens=4"):
        _run_fake_stream(
            monkeypatch,
            [
                _text_event("a"),
                _text_event("", "length"),
                _usage_event(completion_tokens=3),
                "[DONE]",
            ],
        )


def test_benchmark_stream_rejects_non_length_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="unexpected finish_reason.*stop"):
        _run_fake_stream(
            monkeypatch,
            [
                _text_event("a"),
                _text_event("", "stop"),
                _usage_event(),
                "[DONE]",
            ],
        )


def test_benchmark_stream_rejects_prompt_token_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="prompt token mismatch"):
        _run_fake_stream(
            monkeypatch,
            [
                _text_event("a"),
                _text_event("", "length"),
                _usage_event(prompt_tokens=9),
                "[DONE]",
            ],
        )
