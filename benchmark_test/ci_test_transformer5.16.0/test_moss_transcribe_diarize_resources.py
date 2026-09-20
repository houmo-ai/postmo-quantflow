import json
from pathlib import Path

import pytest

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.model_integrations.moss_transcribe_diarize.resources import resolve_config


def _write_bundle(root: Path) -> None:
    (root / "config.json").write_text(
        '{"model_type":"moss_transcribe_diarize","text_config":{"model_type":"qwen3"},'
        '"audio_config":{"model_type":"whisper"},"audio_merge_size":4,"adaptor_input_dim":4096}'
    )
    (root / "processor_config.json").write_text(
        '{"audio_tokens_per_second":12.5,"audio_merge_size":4,'
        '"time_marker_every_seconds":5,"enable_time_marker":true}'
    )
    (root / "preprocessor_config.json").write_text(
        '{"sampling_rate":16000,"feature_size":80,"hop_length":160,"n_fft":400,'
        '"n_samples":480000,"nb_max_frames":3000}'
    )
    (root / "generation_config.json").write_text('{"max_new_tokens":5120,"eos_token_id":151645}')
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}")
    for name in (
        "configuration_moss_transcribe_diarize.py",
        "modeling_moss_transcribe_diarize.py",
        "processing_moss_transcribe_diarize.py",
    ):
        (root / name).write_text("# remote-code fixture\n")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}})
    )
    (root / "model.safetensors").write_bytes(b"fixture")


def test_resource_resolver_accepts_complete_bundle(tmp_path: Path):
    _write_bundle(tmp_path)

    assert resolve_config(
        recipe=get_recipe("moss-transcribe-diarize-0.9b"),
        variant=get_recipe("moss-transcribe-diarize-0.9b").default_variant,
        model_dir=tmp_path,
        sources={},
        workflow={},
    ) == {}


def test_resource_resolver_uses_workflow_max_pe_length(tmp_path: Path):
    _write_bundle(tmp_path)

    assert resolve_config(
        recipe=get_recipe("moss-transcribe-diarize-0.9b"),
        variant=get_recipe("moss-transcribe-diarize-0.9b").default_variant,
        model_dir=tmp_path,
        sources={},
        workflow={"export": {"model": {"max_pe_length": 4096, "prefill_chunk_length": 512}}},
    ) == {"export.model.max_pe_length": 4096}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda root: (root / "processor_config.json").write_text(
            '{"audio_tokens_per_second":10,"audio_merge_size":4,'
            '"time_marker_every_seconds":5,"enable_time_marker":true}'
        ),
        lambda root: (root / "model.safetensors").unlink(),
    ],
)
def test_resource_resolver_rejects_incomplete_bundle(tmp_path: Path, mutate):
    _write_bundle(tmp_path)
    mutate(tmp_path)

    with pytest.raises(ModelError, match="valid local model bundle"):
        resolve_config(
            recipe=get_recipe("moss-transcribe-diarize-0.9b"),
            variant=get_recipe("moss-transcribe-diarize-0.9b").default_variant,
            model_dir=tmp_path,
            sources={},
            workflow={},
        )
