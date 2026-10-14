# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""One isolated public build/load/inference regression process."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from _unified_cases import CASES


def fixture_module(case):
    from importlib import import_module

    return import_module(case.preparation)


def prepare(case, root, monkeypatch):
    return fixture_module(case).prepare(case, root, monkeypatch)


def validate_expert_lora_artifacts(meta, adapters):
    """Check final SDK operands after publication, including streamed graphs."""
    import hashlib

    import onnx
    from onnx import numpy_helper

    fingerprints = []
    for adapter in adapters:
        child_meta = meta.parent / adapter["meta_file"]
        description = json.loads(child_meta.read_text())
        mode_hashes = []
        for mode in ("prefill", "decode"):
            graph = onnx.load(child_meta.parent / description[mode + "_hmonnx"]).graph
            assert not any("placeholder" in node.op_type.lower() for node in graph.node)
            experts = [node for node in graph.node if node.op_type == "MoeBlock"]
            assert len(experts) == 2, "Both tiny decoder layers must retain expert LoRA"
            initializers = {value.name: value for value in graph.initializer}
            digest = hashlib.sha256()
            for node in experts:
                assert len(node.input) == 28 and all(node.input[16:]), "Incomplete expert LoRA ABI"
                attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
                assert attributes["lora_weight_man_bit"] == 8
                assert attributes["lora_scaling"] == 2
                for name in node.input[16:]:
                    digest.update(numpy_helper.to_array(initializers[name]).tobytes())
            mode_hashes.append(digest.hexdigest())
        assert mode_hashes[0] == mode_hashes[1], "Prefill/decode must use the same adapter weights"
        fingerprints.append(mode_hashes[0])
    assert len(set(fingerprints)) == len(adapters), "Adapters must not reuse another adapter's weights"
    return fingerprints


def run(case, root, *, low_memory=None):
    import pytest
    import torch

    from xhmodel_merak.model_build import build_model
    from xhmodel_merak.model_inference import inference_model

    torch.set_num_threads(4)
    root.mkdir(parents=True, exist_ok=True)
    record = {"case": case.id, "stage": "prepare", "passed": False}
    started = time.monotonic()
    try:
        with pytest.MonkeyPatch.context() as patch:
            kwargs, inputs = prepare(case, root, patch)
            if low_memory is not None:
                kwargs["low_memory"] = low_memory
            if case.variant == "lora_mtp":
                record["stage"] = "prepare_mtp_checkpoint"
                base = dict(kwargs)
                base.pop("lora")
                base.update(output_dir=str(root / "base_artifact"), quant_output_dir=str(root / "base_quant"))
                base_result = build_model(**base)
                assert base_result.quant_artifact is not None
                kwargs["quant_checkpoint"] = base_result.quant_artifact
            record["stage"] = "build"
            result = build_model(**kwargs)
            record["build"] = result.to_dict()
            assert result.loadable
            artifact = Path(result.output_dir)
            assert (artifact / "export_meta_info.json").is_file()
            assert (artifact / "environment.json").is_file()
            assert json.loads((artifact / "environment.json").read_text())["variant"] == case.variant
            fixture = fixture_module(case)
            if hasattr(fixture, "validate_build"):
                fixture.validate_build(case, result, record)
            moved = root / "relocated"
            artifact.rename(moved)
            # Generated checkpoints disappear before inference: no accidental
            # dependence on the build machine's source or quantization cache.
            for name in (
                "source",
                "checkpoint",
                "quant",
                "assistant",
                "draft",
                "native_draft",
                "base_quant",
                "base_artifact",
                "lora",
                "tokenizer",
            ):
                path = root / name
                if path.is_dir():
                    shutil.rmtree(path)
            record["stage"] = "inference"
            if hasattr(fixture, "check_inference"):
                fixture.check_inference(moved, inputs, record)
            else:
                output = inference_model(artifact_dir=str(moved), inputs=inputs, device="cuda:0")
                assert output.model_id == case.model_id
                value = output.to_dict()
                json.dumps(value, allow_nan=False)
                if case.fixture == "moss":
                    # ASR's public adapter does not expose a token count. Its
                    # structured transcript is validated inside inference_model.
                    assert isinstance(output.output_text, str) and output.output_text.strip()
                elif hasattr(output, "output_tokens"):
                    assert output.output_tokens > 0
                if case.variant in ("mtp", "lora_mtp", "dflash", "dflash2", "dspark"):
                    assert output.spec_decode_mode == (case.variant if case.variant in {"dflash", "dflash2", "dspark"} else "mtp")
                if hasattr(output, "actions"):
                    assert (output.horizon, output.action_dim) == (50, 7)
                    assert len(output.actions) == 50
                if hasattr(output, "num_samples"):
                    assert output.num_samples > 0
                if hasattr(fixture, "validate_output"):
                    fixture.validate_output(case, output, result, moved, inputs, root, record)
                if case.variant in ("lora", "lora_mtp"):
                    # The public API runs the root model. Exercise the exported
                    # adapter too, through the same runtime used by that API.
                    from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import quick_test_hmonnx
                    from xhmodel_merak.model_inference.inputs import scoped_environment

                    meta = Path(output.meta_file)
                    adapters = json.loads(meta.read_text())["lora_adapters"]
                    assert len(adapters) == (3 if case.fixture == "macaron" else 1)
                    if case.fixture == "macaron":
                        record["expert_lora_fingerprints"] = validate_expert_lora_artifacts(meta, adapters)
                    record["lora_inference_tokens"] = {}
                    for adapter in adapters:
                        with scoped_environment({"ENABLE_HMINFERENCE_V2": "1"}):
                            child = quick_test_hmonnx(
                                str(meta.parent / adapter["meta_file"]),
                                prompt="hello world",
                                device="cuda:0",
                                max_new_tokens=2,
                                stream_output=False,
                            )
                        assert child.output_tokens > 0
                        if case.variant == "lora_mtp":
                            assert child.spec_decode_mode == "mtp"
                        record["lora_inference_tokens"][adapter["name"]] = child.output_tokens
                record.update(passed=True, stage="complete", inference=value)
    except BaseException as exc:
        from _unified_reporting import record_error

        record_error(record, exc)
        raise
    finally:
        from _unified_reporting import write_report

        write_report(root, record, started)


if __name__ == "__main__":
    case = next(c for c in CASES if c.id == sys.argv[1])
    run(case, Path(sys.argv[2]).resolve(), low_memory=True if "--low-memory" in sys.argv[3:] else None)
