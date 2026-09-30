from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import vllm.distributed.utils
import vllm_merak.worker_v1.merak_worker_v1 as worker_module
from vllm_merak.worker_v1.merak_worker_v1 import MerakWorkerV1


class _FakeProfiler:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = 0
        self.stepped = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def step(self):
        self.stepped += 1

    def stop(self):
        self.stopped += 1

    def annotate_context_manager(self, name):
        self.annotation = name
        return nullcontext()


def _worker(profiler: str | None) -> MerakWorkerV1:
    worker = MerakWorkerV1.__new__(MerakWorkerV1)
    worker.profiler_config = SimpleNamespace(profiler=profiler)
    worker.profiler = None
    worker.rank = 3
    worker.local_rank = 1
    return worker


def test_merak_worker_torch_profiler_lifecycle(monkeypatch):
    monkeypatch.setattr(worker_module, "TorchProfilerWrapper", _FakeProfiler)
    monkeypatch.setattr(
        vllm.distributed.utils,
        "get_worker_rank_suffix",
        lambda *, global_rank: f"rank-{global_rank}",
    )
    worker = _worker("torch")

    worker.profile(True, "w4-packed")
    assert isinstance(worker.profiler, _FakeProfiler)
    assert worker.profiler.kwargs["worker_name"] == "w4-packed_rank-3"
    assert worker.profiler.kwargs["local_rank"] == 1
    assert worker.profiler.kwargs["activities"] == ["CPU", "CUDA"]
    assert worker.profiler.started == 1

    worker.profile(False)
    assert worker.profiler.stopped == 1


def test_merak_worker_profile_requires_configuration():
    worker = _worker(None)
    with pytest.raises(RuntimeError, match="Profiling is not enabled"):
        worker.profile(True)


def test_merak_worker_steps_and_annotates_profiler_before_execution(
    monkeypatch,
):
    worker = _worker("torch")
    worker.profiler = _FakeProfiler()
    expected = object()
    worker.model_runner = SimpleNamespace(
        execute_model=lambda scheduler_output: expected
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    monkeypatch.setattr(
        worker_module,
        "compute_iteration_details",
        lambda output: SimpleNamespace(
            num_ctx_requests=1,
            num_ctx_tokens=256,
            num_generation_requests=0,
            num_generation_tokens=0,
        ),
    )

    assert worker.execute_model(scheduler_output) is expected
    assert worker.profiler.stepped == 1
    assert worker.profiler.annotation == "execute_context_1(256)_generation_0(0)"


def test_merak_worker_delegates_draft_token_retrieval():
    worker = _worker(None)
    expected = object()
    worker.model_runner = SimpleNamespace(take_draft_token_ids=lambda: expected)

    assert worker.take_draft_token_ids() is expected


def test_merak_worker_initializes_model_parallel_before_spec_runner(
    monkeypatch,
):
    events: list[str] = []
    model_config = SimpleNamespace(
        model="artifact",
        architecture="Gemma4UnifiedMerakForCausalLM",
        runner_type="generate",
    )
    cache_config = SimpleNamespace(cache_dtype="float16")
    profiler_config = SimpleNamespace(profiler=None)
    vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        profiler_config=profiler_config,
    )

    def fake_worker_base_init(self, **kwargs):
        self.vllm_config = kwargs["vllm_config"]
        self.model_config = model_config
        self.cache_config = cache_config

    class FakeRunner:
        def __init__(self, *_args, **_kwargs):
            events.append("runner")

    monkeypatch.setattr(
        worker_module.WorkerBase,
        "__init__",
        fake_worker_base_init,
    )
    monkeypatch.setattr(
        MerakWorkerV1,
        "init_distributed_environment",
        lambda _self: events.append("distributed"),
    )
    monkeypatch.setattr(worker_module, "MerakModelRunnerV1", FakeRunner)
    monkeypatch.setattr(worker_module.torch.cuda, "is_available", lambda: False)

    MerakWorkerV1(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:1",
    )

    assert events == ["distributed", "runner"]
