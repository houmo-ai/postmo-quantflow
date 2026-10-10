"""API evaluation regressions, without model weights or a live service."""

import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


EVAL_DIR = Path(__file__).resolve().parents[2] / "examples_merak/llm/xh_token_spark_2_5/eval"
sys.path.insert(0, str(EVAL_DIR))
import vllm_spark_tokens_eval as api  # noqa: E402


def test_sse_reasoning_and_tool_calls():
    events = [
        {"choices": [{"delta": {"reasoning": "think", "content": None}}]},
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "id": "call1", "function": {"name": "f", "arguments": "{"}}]}}
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}, "finish_reason": "tool_calls"}
            ]
        },
        {"choices": [], "usage": {"completion_tokens": 10}},
    ]
    data = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n"
    client = api.Client("http://localhost/v1")
    with patch.object(client, "open", return_value=io.BytesIO(data.encode())):
        result = client.generate({}, "test")
    assert result["reasoning"] == "think"
    assert result["tool_calls"][0]["function"] == {"name": "f", "arguments": "{}"}
    assert result["usage"]["completion_tokens"] == 10


def test_incomplete_sse_rejected():
    client = api.Client("http://localhost/v1")
    with patch.object(client, "open", return_value=io.BytesIO(b'data: {"choices": []}\n')):
        with pytest.raises(RuntimeError, match="Incomplete SSE"):
            client.generate({}, "test")


class FakeClient:
    base_url = "http://localhost/v1"
    finish = "stop"
    calls = 0
    input_tokens = 10

    def __init__(self, *a):
        pass

    def json(self, path, body=None):
        if path == "/models":
            return {"data": [{"id": "spark2_5", "root": "model", "max_model_len": 100}]}
        return {"count": self.input_tokens}

    def generate(self, body, label):
        type(self).calls += 1
        return dict(
            content="A", reasoning="thought", tool_calls=[], finish_reason=self.finish, usage={"completion_tokens": 1}
        )


@pytest.fixture
def run(tmp_path, monkeypatch):
    suite = tmp_path / "suite.yaml"
    suite.write_text("benchmarks:\n  - name: GPQA-Diamond\n    generation:\n      max_new_tokens: 50\n")
    (tmp_path / "gpqa_diamond.jsonl").write_text(json.dumps({"id": "one", "prompt": "test", "answer": "A"}) + "\n")
    args = ["--suite", str(suite), "--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")]
    monkeypatch.setattr(api, "Client", FakeClient)
    monkeypatch.setattr(FakeClient, "calls", 0)
    monkeypatch.setattr(FakeClient, "finish", "stop")
    monkeypatch.setattr(FakeClient, "input_tokens", 10)
    return args, tmp_path / "out"


def test_scoring_resume_and_mismatch(run):
    args, out = run
    assert api.main(args) == 0
    assert json.loads((out / "results.json").read_text())["results"][0]["score"] == 100
    config = out / "run_config.json"
    legacy = json.loads(config.read_text())
    legacy["precision"] = "w8a8h1_sefp"
    legacy["suite"]["benchmarks"][0]["generation"]["max_new_tokens"] = 1000
    config.write_text(json.dumps(legacy))
    assert api.main(args + ["--resume"]) == 0
    assert "precision" not in json.loads(config.read_text())
    assert FakeClient.calls == 1
    with pytest.raises(ValueError, match="Resume configuration changed"):
        api.main(args + ["--resume", "--seed", "42"])
    with pytest.raises(ValueError, match="Output already contains"):
        api.main(args)


def test_resume_ignores_unselected_benchmarks_and_legacy_labels(run):
    args, out = run
    assert api.main(args) == 0
    config = out / "run_config.json"
    saved = json.loads(config.read_text())
    saved.pop("generation_budget_policy")
    saved.pop("context_length")
    saved["suite"]["suite"] = {"name": "old-suite"}
    saved["suite"]["benchmarks"].append({"name": "Gaokao-2026"})
    config.write_text(json.dumps(saved))
    assert api.main(args + ["--resume"]) == 0
    assert FakeClient.calls == 1


@pytest.mark.parametrize("field,value", [
    ("generation_budget_policy", "fixed"),
    ("context_length", 200),
    ("served_model", {"id": "spark2_5", "root": "other", "max_model_len": 100}),
    ("datasets", {}),
])
def test_resume_rejects_inference_changes(run, field, value):
    args, out = run
    assert api.main(args) == 0
    config = out / "run_config.json"
    saved = json.loads(config.read_text())
    saved[field] = value
    config.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Resume configuration changed"):
        api.main(args + ["--resume"])
    assert FakeClient.calls == 1


def test_truncation_is_failed_and_retried(run, monkeypatch):
    args, out = run
    monkeypatch.setattr(FakeClient, "finish", "length")
    assert api.main(args) == 1
    result = json.loads((out / "results.json").read_text())["results"][0]
    assert result["score"] is None and result["failed"] == 1
    monkeypatch.setattr(FakeClient, "finish", "stop")
    assert api.main(args + ["--resume"]) == 0
    assert FakeClient.calls == 2


@pytest.mark.parametrize("input_tokens", [100, 101])
def test_overflow_never_calls_generation(run, monkeypatch, input_tokens):
    args, out = run
    monkeypatch.setattr(FakeClient, "input_tokens", input_tokens)
    assert api.main(args) == 1
    assert FakeClient.calls == 0
    row = json.loads((out / "gpqa_diamond/predictions.jsonl").read_text())
    assert "leaves no room" in row["error"]


def test_budget_override_and_tool_schema():
    args = api.build_parser().parse_args([])
    raw = {
        "id": "1",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    }
    generation = {
        "max_new_tokens": 10,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
    }
    _, body = api.request_body(raw, "BFCL-V4", args, generation, True)
    assert body["messages"] == raw["messages"] and body["tools"] == raw["tools"]
    assert body["top_k"] == -1 and body["chat_template_kwargs"]["enable_thinking"]


def test_bfcl_bare_function_keeps_original_schema():
    function = {"name": "math.sum", "parameters": {"type": "dict", "properties": {"x": {"type": "float"}}}}
    body = api.ChatRequest().apply_chat_template(
        [{"role": "user", "content": "test"}], tools=[function], enable_thinking=True
    )
    assert body["tools"] == [{"type": "function", "function": function}]
    assert "type" not in function  # Do not mutate dataset records.


@pytest.mark.parametrize(
    "budget,input_tokens,expected",
    [(50, 10, 50), (50, 50, 50), (50, 51, 49), (100, 10, 90), (131072, 99, 1)],
)
def test_dynamic_budget_and_resume(run, monkeypatch, budget, input_tokens, expected):
    args, out = run
    monkeypatch.setattr(FakeClient, "input_tokens", input_tokens)
    with patch.object(FakeClient, "generate", autospec=True, side_effect=FakeClient.generate) as generate:
        assert api.main(args + ["--max-new-tokens", str(budget)]) == 0
        assert generate.call_args.args[1]["max_tokens"] == expected
    row = json.loads((out / "gpqa_diamond/predictions.jsonl").read_text())
    assert row["input_tokens"] == input_tokens
    assert row["request"]["max_tokens"] == row["generation"]["max_new_tokens"] == expected
    config = json.loads((out / "run_config.json").read_text())
    assert config["generation"]["GPQA-Diamond"]["max_new_tokens"] == budget
    with pytest.raises(ValueError, match="Resume configuration changed"):
        api.main(args + ["--resume", "--max-new-tokens", "40"])


def test_default_suite_and_paths_belong_to_destination():
    import download_mini_eval_datasets as downloader

    repo = Path(__file__).resolve().parents[2]
    args = api.build_parser().parse_args([])
    assert api.common.REPO_ROOT == repo
    assert args.data_dir == repo / "data/evaluation/spark_x2_5"
    assert downloader.DEFAULT_OUTPUT_DIR == args.data_dir
    assert args.suite == downloader.DEFAULT_SUITE == EVAL_DIR / "mini_eval_vllm_128k.yaml"
    suite = api.common.load_suite(args.suite)
    args.do_sample = True
    budgets = {
        b["name"]: api.common.resolve_generation_config(args, suite, b)["max_new_tokens"] for b in suite["benchmarks"]
    }
    assert budgets == {
        "IFEval": 131072,
        "GPQA-Diamond": 131072,
        "AIME-2026": 131072,
        "Gaokao-2026": 131072,
        "AA-LCR": 131072,
        "BFCL-V4": 131072,
        "SciCode": 131072,
    }


def test_client_imports_without_inference_packages_from_other_cwd(tmp_path):
    # The migrated HTTP client must not load the old XH2 entry point or weights.
    script = """
import importlib.abc
import runpy
import sys

class RejectInferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {'torch', 'transformers', 'vllm', 'xhquant', 'xhmodel_merak', 'spark_tokens_xh2_eval'}
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Unexpected inference import: ' + fullname)

sys.meta_path.insert(0, RejectInferenceImports())
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(EVAL_DIR / "vllm_spark_tokens_eval.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--ifeval-scorer-dir" in result.stdout


def test_shared_prompt_scoring_and_unicode_resume(tmp_path):
    record = {
        "id": "gpqa-one",
        "Question": "Which option is correct?",
        "Correct Answer": "correct",
        "Incorrect Answer 1": "wrong one",
        "Incorrect Answer 2": "wrong two",
        "Incorrect Answer 3": "wrong three",
    }
    prepared = api.common.prepare_record(record, "GPQA-Diamond", 1024)
    repeated = api.common.prepare_record(record, "GPQA-Diamond", 1024)
    assert prepared == repeated
    assert prepared["choices"][ord(prepared["expected"]) - ord("A")] == "correct"
    assert api.common.score_prediction("GPQA-Diamond", prepared, prepared["expected"])[1] is True
    assert api.common.score_prediction("AIME-2026", {"answer": "42"}, r"\boxed{42}") == ("42", True)
    path = tmp_path / "predictions.jsonl"
    rows = [
        {"id": "one", "status": "failed", "finish_reason": "length"},
        {"id": "one", "status": "complete", "response": "text\u2028more", "finish_reason": "stop"},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    assert api.common.read_completed_predictions(path) == {"one": rows[-1]}


def test_ifeval_alignment_download_ids_and_failure_policy(tmp_path):
    import download_mini_eval_datasets as downloader
    import score_spark_tokens_ifeval as scorer

    dataset = [
        {"key": key, "prompt": f"Prompt {key}", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
        for key in (1000, 1001)
    ]
    source = tmp_path / "ifeval.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in dataset) + "\n")
    dataset = downloader.load_ifeval([source])
    assert [row["id"] for row in dataset] == ["1000", "1001"]
    predictions = [
        {"id": "1000", "status": "failed", "finish_reason": "length", "response": "old"},
        {"id": "1000", "status": "complete", "finish_reason": "stop", "response": "new"},
        {"id": "1001", "status": "failed", "finish_reason": "length", "response": "truncated"},
    ]
    _, responses, info = scorer.align_predictions(dataset, predictions)
    assert [row["response"] for row in responses] == ["new", ""]
    assert (info["scored_samples"], info["failed_samples"], info["truncated_samples"]) == (2, 1, 1)
    _, responses, _ = scorer.align_predictions(dataset, predictions, include_failed=True)
    assert responses[-1]["response"] == "truncated"
    with pytest.raises(ValueError, match="Missing 1 predictions"):
        scorer.align_predictions(dataset, predictions[:2])
