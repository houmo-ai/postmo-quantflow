from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from examples_merak.llm.wemm_embedding.mmeb_v3_vllm_merak import (
    build_vllm_request,
)
from examples_merak.llm.wemm_embedding.run_mmeb_v3_vllm_merak_full import (
    OFFICIAL_COMMIT,
    Job,
    Task,
    _balanced_groups,
    _command,
    _protocol_matches,
    _sha256,
)


def _task(name: str, seconds: float) -> Task:
    root = Path("/tmp/wemm-mmeb-test")
    return Task(
        modality="image",
        dataset_name=name,
        config=root / f"{name}.yaml",
        output=root / name,
        log=root / f"{name}.log",
        task_config={"dataset_parser": "image_cls"},
        estimated_seconds=seconds,
    )


def test_balanced_groups_keep_every_dataset_and_group_size() -> None:
    tasks = [_task(f"task-{index}", float(index + 1)) for index in range(19)]

    groups = _balanced_groups(tasks, group_size=8)

    assert len(groups) == 3
    assert all(len(group) <= 8 for group in groups)
    assert sorted(task.dataset_name for group in groups for task in group) == sorted(
        task.dataset_name for task in tasks
    )
    weights = [sum(task.estimated_seconds for task in group) for group in groups]
    assert max(weights) - min(weights) <= 19


def test_narrativeqa_is_isolated_from_other_datasets() -> None:
    tasks = [_task("text/LongEmbed/narrativeqa", 100.0), _task("other", 1.0)]

    groups = _balanced_groups(tasks, group_size=8)

    assert [[task.dataset_name for task in group] for group in groups] == [
        ["text/LongEmbed/narrativeqa"],
        ["other"],
    ]


def test_command_passes_vllm_concurrency_and_reuses_one_group_config() -> None:
    root = Path("/tmp/wemm-mmeb-test")
    args = Namespace(
        python=root / "python",
        repo=root / "repo",
        backend="vllm",
        vllm_max_num_seqs=4,
        gpu_memory_utilization=0.6,
        model=root / "float-model",
        source_model=root / "float-model",
        batch_size=8,
        dataloader_workers=4,
        data=root / "data",
        vllm_merak_source_sha256="unused",
    )
    job = Job(
        modality="image",
        group_name="group_00",
        dataset_names=("one", "two"),
        config=root / "group.yaml",
        output=root / "output",
        log=root / "group.log",
        estimated_seconds=2.0,
    )

    command = _command(args, job)

    assert command[command.index("--vllm-max-num-seqs") + 1] == "4"
    assert command[command.index("--dataloader_num_workers") + 1] == "4"
    assert command[command.index("--dataset_config") + 1] == str(job.config)


def test_long_text_reserves_one_token_below_vllm_context_limit() -> None:
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return "rendered"

        def encode(self, text, *_args, **_kwargs):
            if text == "<embedding>":
                return [99]
            return list(range(100))

    request = build_vllm_request(
        Tokenizer(),
        "text",
        None,
        None,
        image_min_pixels=1,
        image_max_pixels=1,
        video_min_pixels=1,
        video_max_pixels=1,
        max_model_len=32,
    )

    assert request["prompt_token_ids"] == [*range(30), 99]


def test_vllm_request_appends_embedding_pooling_anchor() -> None:
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return "rendered"

        def encode(self, text, *_args, **_kwargs):
            return [99] if text == "<embedding>" else [1, 2]

    request = build_vllm_request(
        Tokenizer(),
        "text",
        None,
        None,
        image_min_pixels=1,
        image_max_pixels=1,
        video_min_pixels=1,
        video_max_pixels=1,
        max_model_len=32,
    )

    assert request["prompt_token_ids"] == [1, 2, 99]


def test_video_frames_delivered_through_image_channel_are_not_capped_at_16(
    monkeypatch,
) -> None:
    import qwen_vl_utils
    from vllm.model_executor.models import wemm_embedding

    observed = {}

    def process_vision_info(messages, **_kwargs):
        observed["frames"] = messages[0]["content"][0]["video"]
        return None, [("processed-video", "metadata")], {"do_sample_frames": False}

    monkeypatch.setattr(qwen_vl_utils, "process_vision_info", process_vision_info)
    monkeypatch.setattr(
        wemm_embedding,
        "prepare_wemm_prompt_token_ids",
        lambda *_args, **_kwargs: [1, 2, 3],
    )
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *_args, **_kwargs: "rendered",
    )
    frames = [object() for _ in range(64)]

    request = build_vllm_request(
        tokenizer,
        "<|video_pad|> Describe the video.",
        frames,
        None,
        image_min_pixels=65536,
        image_max_pixels=1572864,
        video_min_pixels=4096,
        video_max_pixels=262144,
        max_model_len=262144,
    )

    assert observed["frames"] == frames
    assert request["multi_modal_data"]["video"] == ("processed-video", "metadata")


def test_result_reuse_requires_current_adapter_and_vllm_model_hash(tmp_path: Path) -> None:
    adapter = tmp_path / "examples_merak/llm/wemm_embedding/mmeb_v3_vllm_merak.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("# current adapter\n")
    config = tmp_path / "configs/image/task.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("task: {}\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "wemm_embedding"}\n')
    args = Namespace(
        repo=tmp_path,
        backend="vllm",
        model=model,
        source_model=model,
        vllm_max_num_seqs=16,
        vllm_version="test-vllm",
        transformers_version="test-transformers",
        vllm_wemm_model_sha256="current-vllm-model",
    )
    protocol = {
        "backend": "vllm",
        "official_repo_commit": OFFICIAL_COMMIT,
        "mmeb_v3_data_revision": "4a5560b2b64384204b6fea8a82ea986eba51f5aa",
        "mmeb_v2_media_revision": "e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1",
        "wemm_vllm_adapter_sha256": _sha256(adapter),
        "vllm_wemm_model_sha256": "current-vllm-model",
        "vllm_version": "test-vllm",
        "transformers_version": "test-transformers",
        "model_name": str(model),
        "hf_config_sha256": _sha256(model / "config.json"),
        "dataset_config_sha256": _sha256(config),
        "vllm": {"max_model_len": 262144, "max_num_seqs": 16, "max_images": 16},
        "image_preprocessing": {
            "mode": "dynamic",
            "min_pixels": 65536,
            "max_pixels": 1572864,
            "preserve_aspect_ratio": True,
        },
        "video_preprocessing": {
            "num_frames": 64,
            "frame_size": 224,
            "min_pixels": 4096,
            "max_pixels": 262144,
            "total_pixels": 8388608,
            "sample_fps": 1,
        },
        "upstream_args": [
            "--model_backbone",
            "wemm_embedding",
            "--pooling",
            "last",
            "--normalize",
            "true",
        ],
    }

    assert _protocol_matches(args, protocol, config)

    protocol["wemm_vllm_adapter_sha256"] = None
    assert not _protocol_matches(args, protocol, config)
    protocol["wemm_vllm_adapter_sha256"] = _sha256(adapter)
    protocol["vllm_wemm_model_sha256"] = None
    assert not _protocol_matches(args, protocol, config)
