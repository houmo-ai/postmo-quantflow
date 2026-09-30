from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_merak.model_executor.model_loader.deepseek_v4 import (
    DeepseekV4MerakForCausalLM,
)
from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1


class _FakeCacheMixin:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset_kv_cache(self) -> None:
        self.reset_count += 1


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, data):
        self.calls.append(
            {
                "input_ids": data["input_ids"].clone(),
                "past_seq_length": int(data["past_seq_length"]),
            }
        )
        return data["input_ids"], torch.tensor([data["past_seq_length"]])


class _FakeRuntime:
    device = torch.device("cpu")

    def __init__(self, *, fail_at_past: int | None = None) -> None:
        self.cache_mixin = _FakeCacheMixin()
        self.processor = _FakeProcessor()
        self.modes = []
        self.forward_calls = []
        self.fail_at_past = fail_at_past

    def get_kvcache_mixin(self):
        return self.cache_mixin

    def get_data_preprocessor(self):
        return self.processor

    def set_prefill(self) -> None:
        self.modes.append("prefill")

    def set_decode(self) -> None:
        self.modes.append("decode")

    def forward(self, input_ids, past_length):
        past = int(past_length.item())
        if past == self.fail_at_past:
            raise RuntimeError("injected runtime failure")
        self.forward_calls.append((input_ids.clone(), past))
        return torch.full((1, 4), float(past + input_ids.numel()))


def _make_adapter(runtime: _FakeRuntime) -> DeepseekV4MerakForCausalLM:
    adapter = DeepseekV4MerakForCausalLM.__new__(DeepseekV4MerakForCausalLM)
    nn.Module.__init__(adapter)
    adapter.inference_model = runtime
    adapter.inference_device = torch.device("cpu")
    adapter.prefill_chunk_length = 256
    adapter._cache_owner_request_id = None
    adapter._cache_owner_next_past_len = 0
    adapter._extract_request_result = lambda _runtime, output, **_kwargs: SimpleNamespace(
        logits=output
    )
    return adapter


def test_internal_runtime_chunks_prefill_and_preserves_owner_for_decode() -> None:
    runtime = _FakeRuntime()
    adapter = _make_adapter(runtime)

    logits = adapter._runtime_forward(
        torch.arange(513),
        past_len=0,
        request_id="req-a",
    )

    assert runtime.cache_mixin.reset_count == 1
    assert runtime.modes == ["prefill", "prefill", "decode"]
    assert [call["input_ids"].numel() for call in runtime.processor.calls] == [
        256,
        256,
        1,
    ]
    assert [call["past_seq_length"] for call in runtime.processor.calls] == [
        0,
        256,
        512,
    ]
    assert adapter._cache_owner_request_id == "req-a"
    assert adapter._cache_owner_next_past_len == 513
    torch.testing.assert_close(logits, torch.full((4,), 513.0))

    adapter._runtime_forward(
        torch.tensor([99]),
        past_len=513,
        request_id="req-a",
    )

    assert runtime.cache_mixin.reset_count == 1
    assert runtime.modes[-1] == "decode"
    assert adapter._cache_owner_next_past_len == 514


def test_internal_runtime_rejects_missing_owner_and_nonmonotonic_resume() -> None:
    adapter = _make_adapter(_FakeRuntime())
    adapter._cache_owner_request_id = "req-a"
    adapter._cache_owner_next_past_len = 10

    with pytest.raises(RuntimeError, match="matching resident owner"):
        adapter._runtime_forward(
            torch.tensor([1]),
            past_len=10,
            request_id="req-b",
        )

    with pytest.raises(RuntimeError, match="not monotonic"):
        adapter._runtime_forward(
            torch.tensor([1]),
            past_len=9,
            request_id="req-a",
        )


def test_internal_runtime_failure_invalidates_owner() -> None:
    runtime = _FakeRuntime(fail_at_past=256)
    adapter = _make_adapter(runtime)

    with pytest.raises(RuntimeError, match="injected runtime failure"):
        adapter._runtime_forward(
            torch.arange(300),
            past_len=0,
            request_id="req-a",
        )

    assert adapter._cache_owner_request_id is None
    assert adapter._cache_owner_next_past_len == 0


def test_release_internal_cache_owner_is_request_scoped() -> None:
    adapter = _make_adapter(_FakeRuntime())
    adapter._cache_owner_request_id = "req-a"
    adapter._cache_owner_next_past_len = 8

    adapter.release_internal_cache_owner("req-b")
    assert adapter._cache_owner_request_id == "req-a"

    adapter.release_internal_cache_owner("req-a")
    assert adapter._cache_owner_request_id is None
    assert adapter._cache_owner_next_past_len == 0


def test_runner_routes_internal_cache_with_one_request_id() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.input_batch = SimpleNamespace(req_ids=["req-a"])
    runner.num_prompt_logprobs = {}
    captured = {}

    class _Model:
        uses_internal_hmonnx_cache = True

        @staticmethod
        def forward(**kwargs):
            captured.update(kwargs)
            return torch.tensor([1.0])

    runner.model = _Model()
    output = MerakModelRunnerV1._model_forward(
        runner,
        input_ids=torch.tensor([7]),
        positions=torch.tensor([3]),
    )

    assert output.tolist() == [1.0]
    assert captured["request_id"] == "req-a"


def test_runner_rejects_multiple_internal_cache_requests_and_prompt_logprobs() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.model = SimpleNamespace(uses_internal_hmonnx_cache=True)
    runner.num_prompt_logprobs = {}
    runner.input_batch = SimpleNamespace(req_ids=["req-a", "req-b"])

    with pytest.raises(RuntimeError, match="exactly one active request"):
        MerakModelRunnerV1._model_forward(runner, input_ids=torch.tensor([1, 2]))

    runner.input_batch = SimpleNamespace(req_ids=["req-a"])
    runner.num_prompt_logprobs = {"req-a": 2}
    with pytest.raises(RuntimeError, match="does not support prompt logprobs"):
        MerakModelRunnerV1._model_forward(runner, input_ids=torch.tensor([1]))
