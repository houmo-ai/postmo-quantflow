from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch


def _write_worker_shard(
    output_dir,
    index,
    *,
    text_start,
    text_end,
    image_start,
    image_end,
    text,
    image,
):
    torch.save(
        {
            "text_start": text_start,
            "text_end": text_end,
            "image_start": image_start,
            "image_end": image_end,
            "text_embeddings": text,
            "image_embeddings": image,
        },
        output_dir / f"worker_{index:02d}.pt",
    )


def test_retrieval_metrics_perfect_ranking():
    from examples_merak.llm.wemm_embedding.evaluate_wemm_embedding import retrieval_metrics

    assert retrieval_metrics([[0.9, 0.1], [0.8, 0.2]], [0, 0]) == {
        "recall@1": 1.0,
        "mrr": 1.0,
        "ndcg@5": 1.0,
    }


def test_retrieval_metrics_uses_rank_for_mrr_and_ndcg():
    from examples_merak.llm.wemm_embedding.evaluate_wemm_embedding import retrieval_metrics

    metrics = retrieval_metrics([[0.1, 0.9, 0.2]], [0])
    assert metrics["recall@1"] == 0.0
    assert metrics["mrr"] == pytest.approx(1 / 3)
    assert metrics["ndcg@5"] == pytest.approx(1 / 2)


def test_flickr30k_utils_retrieval_metrics_and_flatten():
    from examples_merak.llm.wemm_embedding.flickr30k_utils import (
        flatten_rows,
        retrieval_metrics,
    )

    images, captions, owners, image_to_captions = flatten_rows([("a.jpg", ["cap-a1", "cap-a2"]), ("b.jpg", ["cap-b1"])])
    assert images == ["a.jpg", "b.jpg"]
    assert captions == ["cap-a1", "cap-a2", "cap-b1"]
    assert owners == [0, 0, 1]
    assert image_to_captions == [[0, 1], [2]]

    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    documents = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
    metrics = retrieval_metrics(query, documents, [[0], [1]], recall_ks=(1, 2), rank_limit=2)
    assert metrics["recall@1"] == 1.0
    assert metrics["mrr@2"] == 1.0


def test_flickr30k_utils_retrieval_metrics_rejects_0d_and_1d():
    from examples_merak.llm.wemm_embedding.flickr30k_utils import retrieval_metrics

    documents = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="must be rank-2"):
        retrieval_metrics(torch.tensor(1.0), documents, [[0]])
    with pytest.raises(ValueError, match="must be rank-2"):
        retrieval_metrics(torch.tensor([1.0, 0.0]), documents, [[0]])


def test_retrieval_dataset_registry_supports_future_datasets(tmp_path):
    from examples_merak.llm.wemm_embedding.retrieval_dataset import (
        RetrievalDataset,
        load_retrieval_dataset,
        register_retrieval_dataset,
    )

    dataset_name = f"ci-{tmp_path.name}"
    register_retrieval_dataset(
        dataset_name,
        lambda root: RetrievalDataset(
            name="custom",
            split="validation",
            rows=[(str(root) + "/image.jpg", ["caption one", "caption two"])],
            metadata={"source": "unit-test"},
        ),
    )
    dataset = load_retrieval_dataset(dataset_name, tmp_path)
    assert dataset.name == "custom"
    assert dataset.split == "validation"
    assert dataset.metadata == {"source": "unit-test"}
    assert dataset.flatten() == (
        [str(tmp_path) + "/image.jpg"],
        ["caption one", "caption two"],
        [0, 0],
        [[0, 1]],
    )


def test_retrieval_dataset_ordering_metadata_is_serializable():
    from examples_merak.llm.wemm_embedding.retrieval_dataset import RetrievalDataset

    dataset = RetrievalDataset(
        name="custom",
        split="test",
        rows=[("image.jpg", ["caption"])],
    )
    assert json.loads(json.dumps(dataset.ordering_metadata()))["dataset"] == "custom"


def test_resolve_hmonnx_meta_prefers_golden(tmp_path):
    from examples_merak.llm.wemm_embedding.runtime_utils import resolve_hmonnx_meta

    nested = tmp_path / "export" / "hmquant" / "golden_meta_info.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}", encoding="utf-8")
    assert resolve_hmonnx_meta(tmp_path / "export") == nested
    assert resolve_hmonnx_meta(nested) == nested


def test_parallel_eval_removes_worker_shards(tmp_path):
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import (
        _remove_worker_shards,
    )

    for index in range(2):
        (tmp_path / f"worker_{index:02d}.pt").write_bytes(b"shard")
    _remove_worker_shards(tmp_path, 2)
    assert not list(tmp_path.glob("worker_*.pt"))
    _remove_worker_shards(tmp_path, 2)


def test_parallel_eval_bounds_cover_full_range_without_overlap():
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import _bounds

    for length in (0, 1, 2, 5, 7):
        for workers in (1, 2, 3, 8):
            covered = []
            for index in range(workers):
                start, end = _bounds(length, index, workers)
                covered.extend(range(start, end))
            assert covered == list(range(length))


def test_parallel_eval_merge_reconstructs_ordered_embeddings(tmp_path):
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import _merge

    _write_worker_shard(
        tmp_path,
        1,
        text_start=2,
        text_end=4,
        image_start=1,
        image_end=2,
        text=torch.tensor([[2.0, 0.0], [3.0, 0.0]]),
        image=torch.tensor([[10.0, 1.0]]),
    )
    _write_worker_shard(
        tmp_path,
        0,
        text_start=0,
        text_end=2,
        image_start=0,
        image_end=1,
        text=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        image=torch.tensor([[9.0, 1.0]]),
    )
    text, image = _merge(tmp_path, workers=2, num_captions=4, num_images=2)
    assert torch.equal(text, torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    assert torch.equal(image, torch.tensor([[9.0, 1.0], [10.0, 1.0]]))


def test_parallel_eval_merge_keeps_empty_worker_ranges(tmp_path):
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import _merge

    _write_worker_shard(
        tmp_path,
        0,
        text_start=0,
        text_end=0,
        image_start=0,
        image_end=1,
        text=torch.empty((0, 0)),
        image=torch.tensor([[1.0, 0.0]]),
    )
    _write_worker_shard(
        tmp_path,
        1,
        text_start=0,
        text_end=2,
        image_start=1,
        image_end=1,
        text=torch.tensor([[2.0, 0.0], [3.0, 0.0]]),
        image=torch.empty((0, 0)),
    )
    text, image = _merge(tmp_path, workers=2, num_captions=2, num_images=1)
    assert torch.equal(text, torch.tensor([[2.0, 0.0], [3.0, 0.0]]))
    assert torch.equal(image, torch.tensor([[1.0, 0.0]]))


def test_parallel_eval_merge_fails_when_shard_is_missing(tmp_path):
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import _merge

    _write_worker_shard(
        tmp_path,
        0,
        text_start=0,
        text_end=1,
        image_start=0,
        image_end=1,
        text=torch.tensor([[1.0, 0.0]]),
        image=torch.tensor([[2.0, 0.0]]),
    )
    with pytest.raises(FileNotFoundError, match="shard is missing"):
        _merge(tmp_path, workers=2, num_captions=1, num_images=1)


def test_parallel_eval_merge_fails_on_overlapping_ranges(tmp_path):
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import _merge

    embeddings = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    image = torch.tensor([[3.0, 0.0]])
    _write_worker_shard(
        tmp_path,
        0,
        text_start=0,
        text_end=2,
        image_start=0,
        image_end=1,
        text=embeddings,
        image=image,
    )
    _write_worker_shard(
        tmp_path,
        1,
        text_start=0,
        text_end=2,
        image_start=0,
        image_end=1,
        text=embeddings,
        image=image,
    )
    with pytest.raises(ValueError, match="overlaps previous coverage"):
        _merge(tmp_path, workers=2, num_captions=2, num_images=1)


def test_parallel_eval_merge_fails_on_gapped_ranges(tmp_path):
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import _merge

    _write_worker_shard(
        tmp_path,
        0,
        text_start=0,
        text_end=1,
        image_start=0,
        image_end=1,
        text=torch.tensor([[1.0, 0.0]]),
        image=torch.tensor([[3.0, 0.0]]),
    )
    _write_worker_shard(
        tmp_path,
        1,
        text_start=2,
        text_end=3,
        image_start=1,
        image_end=2,
        text=torch.tensor([[2.0, 0.0]]),
        image=torch.tensor([[4.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="has a gap after previous coverage"):
        _merge(tmp_path, workers=2, num_captions=3, num_images=2)


def test_parallel_eval_half_open_ranges_detect_overlap_and_gap():
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import _advance_half_open_range

    assert _advance_half_open_range(0, 2, 0, "text") == 2
    assert _advance_half_open_range(2, 4, 2, "text") == 4
    with pytest.raises(ValueError, match="overlaps previous coverage"):
        _advance_half_open_range(1, 3, 2, "text")
    with pytest.raises(ValueError, match="has a gap after previous coverage"):
        _advance_half_open_range(3, 4, 2, "text")
    with pytest.raises(ValueError, match="inverted range"):
        _advance_half_open_range(4, 2, 2, "text")


def test_validate_shard_ranges_rejects_overlap_and_gap():
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import validate_shard_ranges

    assert validate_shard_ranges([(0, 2), (2, 4)], expected_count=4, name="text") == 4
    assert validate_shard_ranges([], expected_count=0, name="text") == 0
    with pytest.raises(ValueError, match="overlaps previous coverage"):
        validate_shard_ranges([(0, 3), (2, 4)], expected_count=4, name="text")
    with pytest.raises(ValueError, match="has a gap after previous coverage"):
        validate_shard_ranges([(0, 1), (2, 4)], expected_count=4, name="text")
    with pytest.raises(ValueError, match="does not match dataset size"):
        validate_shard_ranges([(0, 3)], expected_count=4, name="text")


def test_validate_shard_ranges_rejects_incomplete_dataset_coverage():
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import validate_shard_ranges

    with pytest.raises(ValueError, match="does not match dataset size 4"):
        validate_shard_ranges([(0, 3)], expected_count=4, name="text")
    with pytest.raises(ValueError, match="does not match dataset size 4"):
        validate_shard_ranges([(0, 5)], expected_count=4, name="text")
    with pytest.raises(ValueError, match="overlaps previous coverage"):
        validate_shard_ranges([(0, 3), (2, 4)], expected_count=4, name="text")


def test_parallel_eval_merge_accepts_adjacent_half_open_ranges(tmp_path):
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import (
        _advance_half_open_range,
        _merge,
    )

    assert _advance_half_open_range(2, 4, 2, "text") == 4
    _write_worker_shard(
        tmp_path,
        0,
        text_start=0,
        text_end=1,
        image_start=0,
        image_end=1,
        text=torch.tensor([[1.0, 0.0]]),
        image=torch.tensor([[3.0, 0.0]]),
    )
    _write_worker_shard(
        tmp_path,
        1,
        text_start=1,
        text_end=2,
        image_start=1,
        image_end=2,
        text=torch.tensor([[2.0, 0.0]]),
        image=torch.tensor([[4.0, 0.0]]),
    )
    text, image = _merge(tmp_path, workers=2, num_captions=2, num_images=2)
    assert torch.equal(text, torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
    assert torch.equal(image, torch.tensor([[3.0, 0.0], [4.0, 0.0]]))


def test_parallel_eval_merge_fails_when_row_count_mismatches_range(tmp_path):
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import _merge

    _write_worker_shard(
        tmp_path,
        0,
        text_start=0,
        text_end=2,
        image_start=0,
        image_end=1,
        text=torch.tensor([[1.0, 0.0]]),
        image=torch.tensor([[2.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="do not match range"):
        _merge(tmp_path, workers=1, num_captions=2, num_images=1)


class _FakeWorkerProcess:
    def __init__(self, *, exitcode=0, hang=False, join_error=None):
        self.exitcode = exitcode
        self._hang = hang
        self._join_error = join_error
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return self._hang and not self.terminated and not self.killed

    def join(self, timeout=None):
        if self._join_error is not None:
            raise self._join_error
        if self._hang and timeout is None:
            raise RuntimeError("test hang requires a timeout")

    def terminate(self):
        self.terminated = True
        self.exitcode = -15

    def kill(self):
        self.killed = True
        self.exitcode = -9


def test_parallel_eval_has_finite_worker_timeout_by_default(monkeypatch):
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_flickr30k_parallel.py",
            "--export-dir",
            "export",
            "--dataset-dir",
            "data",
            "--output-dir",
            "out",
            "--report",
            "report.json",
        ],
    )
    assert parse_args().worker_timeout == 7200.0


def test_wait_for_worker_processes_fails_closed_on_worker_error():
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import wait_for_worker_processes

    wait_for_worker_processes([_FakeWorkerProcess(), _FakeWorkerProcess()], timeout=0.1)
    with pytest.raises(RuntimeError, match="workers failed"):
        wait_for_worker_processes(
            [_FakeWorkerProcess(), _FakeWorkerProcess(exitcode=1)],
            timeout=0.1,
        )
    hanging_after_failure = _FakeWorkerProcess(hang=True)
    with pytest.raises(RuntimeError, match="exit code 1"):
        wait_for_worker_processes([_FakeWorkerProcess(exitcode=1), hanging_after_failure])
    assert hanging_after_failure.terminated is True
    with pytest.raises(RuntimeError, match="exit code None"):
        wait_for_worker_processes([_FakeWorkerProcess(exitcode=None)])
    with pytest.raises(RuntimeError, match="exit code -15"):
        wait_for_worker_processes([_FakeWorkerProcess(exitcode=-15)])
    hanging = _FakeWorkerProcess(hang=True)
    with pytest.raises(TimeoutError, match="still running"):
        wait_for_worker_processes([hanging], timeout=0.01)
    assert hanging.terminated is True
    with pytest.raises(RuntimeError, match="join raised"):
        wait_for_worker_processes(
            [_FakeWorkerProcess(join_error=RuntimeError("boom"))],
            timeout=0.1,
        )


def test_parallel_eval_reaps_killed_workers():
    from examples_merak.llm.wemm_embedding.eval_flickr30k_parallel import wait_for_worker_processes

    class StubbornWorker(_FakeWorkerProcess):
        def __init__(self):
            super().__init__(hang=True)
            self.join_count = 0

        def is_alive(self):
            return not self.killed

        def join(self, timeout=None):
            self.join_count += 1

    worker = StubbornWorker()
    with pytest.raises(TimeoutError, match="still running"):
        wait_for_worker_processes([worker], timeout=0.01)
    assert worker.killed is True
    assert worker.join_count == 3  # initial join, join after terminate, join after kill


def test_parallel_eval_cleans_worker_shards_after_failure(monkeypatch, tmp_path):
    from examples_merak.llm.wemm_embedding import eval_flickr30k_parallel as parallel

    class FailedWorker(_FakeWorkerProcess):
        def __init__(self):
            super().__init__(exitcode=1)

        def start(self):
            (tmp_path / "worker_00.pt").write_bytes(b"partial shard")

    worker = FailedWorker()
    dataset = SimpleNamespace(
        limited=lambda count: SimpleNamespace(flatten=lambda: (["image"], ["caption"], [0], [[0]]))
    )
    args = SimpleNamespace(
        workers=1,
        max_images=0,
        dataset_type="flickr30k",
        dataset_dir=str(tmp_path),
        output_dir=str(tmp_path),
        export_dir="unused",
        device="cpu",
        image_min_pixels=65536,
        image_max_pixels=1572864,
        worker_timeout=0.01,
        report=str(tmp_path / "report.json"),
    )
    monkeypatch.setattr(parallel, "parse_args", lambda: args)
    monkeypatch.setattr(parallel, "load_retrieval_dataset", lambda *args: dataset)
    monkeypatch.setattr(parallel.mp, "get_context", lambda method: SimpleNamespace(Process=lambda **kwargs: worker))
    with pytest.raises(RuntimeError, match="workers failed"):
        parallel.main()
    assert not (tmp_path / "worker_00.pt").exists()


def test_dynamic_resolution_audit_records_real_sizes_and_gears(tmp_path):
    from PIL import Image

    from examples_merak.llm.wemm_embedding.audit_dynamic_resolution import audit_images

    paths = []
    for index, size in enumerate(((640, 480), (1200, 300), (2000, 1600))):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", size).save(path)
        paths.append(path)

    result = audit_images(paths, min_pixels=65536, max_pixels=1572864)

    assert result["images"] == 3
    assert result["unique_original_sizes"] == 3
    assert result["unique_resized_sizes"] == 3
    assert sum(result["gear_counts"].values()) == 3
    assert result["visual_token_max"] <= 1536
