# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import json

from _unified_preparation import prepare_context


def prepare(case, root, monkeypatch):
    catalog, recipe, source, calibration, image, kwargs, inputs, replacements, family = prepare_context(case, root)
    # Release assets remain explicit and pinned; no model-hub downloads.
    if family == "kokoro":
        from _unified_assets import kokoro_source

        source = kokoro_source()
    else:
        from _unified_assets import moss_source

        source = moss_source()
    kwargs["model_dir"] = str(source)
    if family == "kokoro":
        from _unified_special_fixtures import configure_kokoro_minimal

        configure_kokoro_minimal(catalog, recipe, monkeypatch)
        inputs = {"text": "你好"}
    else:
        import numpy as np
        import soundfile as sf

        audio = root / "audio.wav"
        sf.write(audio, 0.03 * np.sin(np.arange(16000) * 2 * np.pi * 220 / 16000), 16000)
        inputs = {"audio": str(audio), "max_new_tokens": 64}
        kwargs.update(context_length=2048, prefill_chunk_length=512)
    return kwargs, inputs


def validate_output(case, output, result, moved, inputs, root, record):
    from pathlib import Path
    if case.fixture != "kokoro":
        return

    meta = json.loads(Path(output.meta_file).read_text())
    routes = meta["bucket_presets"]["routes"]
    assert [(r["token_max_length"], r["frame_max_length"]) for r in routes] == [(32, 160)]
    assert all(len(c["buckets"]) == 1 for c in meta["components"].values())
    assert output.segments[0]["route"] == "t0032_f0160"
    record["tested_routes"] = ["t0032_f0160"]
