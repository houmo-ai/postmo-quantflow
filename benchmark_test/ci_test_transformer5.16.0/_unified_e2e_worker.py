# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""One isolated public build/load/inference regression process."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

from _unified_cases import CASES
from _unified_small_model import write_auxiliary, write_calibration, write_native_checkpoint


def prepare(case, root, monkeypatch):
    from xhmodel_merak.model_catalog.registry import get_catalog

    catalog = get_catalog()
    recipe = catalog._recipes[case.model_id]
    source = root / "source"
    calibration, image = write_calibration(root)
    kwargs = {"model_id": case.model_id, "model_dir": str(source),
              "output_dir": str(root / "artifact"), "device": "cuda:0"}
    inputs = {"prompt": "hello world", "max_new_tokens": 2}
    replacements = dict(recipe.config_replacements)
    family = case.fixture
    if family == "pi05":
        from _pi05_small_fixture import write_pi05
        tokenizer = root / "tokenizer"
        write_pi05(source, tokenizer, monkeypatch)
        monkeypatch.setitem(catalog._recipes, recipe.model_id, replace(
            recipe, config_replacements=(*recipe.config_replacements, ("export.other.hidden_size", 64)),
        ))
        kwargs["resources"] = {"tokenizer": str(tokenizer)}
        inputs = {"task": "pick up the object", "observation.images.image": str(image),
                  "observation.images.image2": str(image), "observation.state": [0.] * 8, "seed": 1}
        return kwargs, inputs
    if family in ("kokoro", "moss"):
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

    if family == "glm":
        from _glm53_packed_fixture import write_native_glm53
        write_native_glm53(source)
        kwargs["quant_output_dir"] = str(root / "quant")
        config = json.loads((source / "config.json").read_text())
        calibration.write_text("".join(
            json.dumps({"text": " ".join(f"t{2 + (i + row) % 40}" for i in range(128))}) + "\n"
            for row in range(2)
        ))
        inputs["prompt"] = "t2 t3 t4"
        replacements.update({"export.model.linear_chunk_size": 8,
                             "export.model.placeholder_export_workers": 1,
                             "export.model.vision_max_patches": 32})
    elif family == "ling":
        from _unified_special_fixtures import write_ling
        config = write_ling(source)
        # Match the tiny numeric vocabulary; natural language would become
        # repeated UNK tokens and be rejected by AutoRound's sample filter.
        calibration.write_text("".join(
            json.dumps({"text": " ".join(str(4 + (i + row) % 6) for i in range(128))}) + "\n"
            for row in range(2)
        ))
        inputs["prompt"] = "4 5 6"
    else:
        config = write_native_checkpoint(source, "qwen_moe" if family == "macaron" else family)
    gemma = family.startswith("gemma")
    context = 320 if gemma else 256 if family == "paddleocr" else 64
    prefill = 320 if gemma else 64 if family in ("paddleocr", "ling") else 8
    kwargs.update(context_length=context, prefill_chunk_length=prefill)
    if "calibration" in recipe.required_resources:
        kwargs["resources"] = {"calibration": str(calibration)}
    if family == "paddleocr":
        kwargs["resources"] = {"calibration": str(calibration)}
        inputs["image"] = str(image)
        inputs["prompt"] = "OCR:"
    if case.model_id.startswith("wemm"):
        inputs = {"text": "hello world"}
    if case.variant in ("mtp", "lora_mtp"):
        kwargs["speculative"] = {"mode": "mtp"}
        if gemma:
            from _unified_special_fixtures import write_gemma_assistant
            assistant = root / "assistant"
            replacements.update(write_gemma_assistant(assistant, config, ordered=family in ("gemma_e2b", "gemma_e4b")))
            kwargs["sources"] = {"mtp": str(assistant)}
        else:
            write_auxiliary(source, family + "_mtp")
            replacements.update({
                "export.model.mtp_config.hidden_size": 64,
                "export.model.mtp_config.num_key_value_heads": 2,
                "export.model.mtp_config.head_dim": 32,
                "export.model.mtp_config.input_sequence_length": 8,
                "export.model.mtp_config.max_pe_length": 128,
            })
    if case.variant in {"dflash", "dflash2"}:
        draft = root / "draft"
        write_auxiliary(draft, case.variant, target_config=config)
        kwargs.update(speculative={"mode": case.variant}, sources={case.variant: str(draft)})
    if case.variant in ("lora", "lora_mtp") and family != "macaron":
        adapter = root / "lora"
        write_auxiliary(adapter, "qwen_lora")
        kwargs["lora"] = {"paths": [str(adapter)]}
    if family == "macaron" and case.variant == "lora":
        from _unified_special_fixtures import write_macaron_loras
        write_macaron_loras(source)
        kwargs["lora"] = {"paths": [str(source / "loras" / f"L{i}") for i in range(3)]}
    constraints = (("model_type", config["model_type"]),)
    if family in ("minicpm", "ling"):
        constraints = (("architectures.0", config["architectures"][0]),)
    monkeypatch.setitem(catalog._recipes, case.model_id, replace(
        recipe, source_constraints=constraints, max_pe_length=512 if gemma else max(128, context),
        config_replacements=tuple(replacements.items()),
    ))
    configure_calibration(case, calibration, monkeypatch)
    return kwargs, inputs


def configure_calibration(case, calibration, monkeypatch):
    """Change test data and calibration budget, never bypass quant/export calls."""
    import xhmodel_merak.model_build.api as api

    if case.fixture == "minicpm":
        import datasets

        def local_wikitext(name, config, *, split):
            assert (name, config, split) == ("Salesforce/wikitext", "wikitext-2-raw-v1", "train")
            return datasets.Dataset.from_dict({"text": ["hello world " * 32] * 2})

        # The legacy workflow owns a fixed dataset request. Replace only that
        # external data boundary; quantization itself executes unchanged.
        monkeypatch.setattr(datasets, "load_dataset", local_wikitext)
        return

    original = api.resolve_build_plan

    def resolve(**kwargs):
        plan = original(**kwargs)
        replacements = copy.deepcopy(plan.config_replacements)
        quant = replacements.get("quant")
        fields = {"calibration.dataset": str(calibration), "calibration.nsamples": 2,
                  "calibration.seqlen": 64, "runtime.batch_size": 1}
        if case.fixture in ("qwen_next", "glm", "ling"):
            fields["calibration.jsonl"] = fields.pop("calibration.dataset")
        if case.fixture == "qwen_next" and plan.quant_method == "gptq":
            # A random tiny checkpoint is not a WikiText benchmark. Keep real
            # quantization/reload and public inference, without hub downloads
            # or the example script's default 128-token demonstration.
            fields.update({"validation.skip_ppl": True, "validation.max_new_tokens": 2})
        if plan.quant_method == "autoround":
            fields["iters"] = 4
        if case.fixture == "paddleocr":
            fields = {"nsamples": 2, "iters": 4}
        for key, value in fields.items():
            # A resource binding may already occupy this dotted path after the
            # whole quant mapping. Ensure the fixture data remains the last write.
            replacements.pop("quant." + key, None)
            if isinstance(quant, dict):
                node = quant
                parts = key.split(".")
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = value
            else:
                replacements["quant." + key] = value
        if isinstance(quant, dict) and quant.get("mtp"):
            quant["mtp"].update(validation_nsamples=1, iters=4)
        return replace(plan, config_replacements=replacements)

    monkeypatch.setattr(api, "resolve_build_plan", resolve)


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
            if case.fixture == "glm":
                assert result.quant_artifact is not None
                reports = list(Path(result.quant_artifact).rglob("glm53_quantization_recipe.json"))
                assert len(reports) == 1, "GLM checkpoint must preserve real quantization evidence"
                evidence = json.loads(reports[0].read_text())
                provenance = evidence["provenance"]
                assert evidence["method"] == "autoround" and provenance["dry_run"] is False
                assert provenance["iters"] == 4 and provenance["tuning"]["loss_evaluations"] > 0
                assert evidence["calibration"]["actual_nsamples"] == 2
                assert evidence["calibration"]["actual_tokens"] == 128
                assert all(provenance["audit"][key] > 0 for key in ("w4_modules", "w8_modules"))
                record["quantization_evidence"] = evidence
            moved = root / "relocated"
            artifact.rename(moved)
            # Generated checkpoints disappear before inference: no accidental
            # dependence on the build machine's source or quantization cache.
            for name in ("source", "checkpoint", "quant", "assistant", "draft", "base_quant", "base_artifact", "lora", "tokenizer"):
                path = root / name
                if path.is_dir():
                    shutil.rmtree(path)
            record["stage"] = "inference"
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
            if case.variant in ("mtp", "lora_mtp", "dflash", "dflash2"):
                assert output.spec_decode_mode == (case.variant if case.variant in {"dflash", "dflash2"} else "mtp")
            if hasattr(output, "actions"):
                assert (output.horizon, output.action_dim) == (50, 7)
                assert len(output.actions) == 50
            if hasattr(output, "num_samples"):
                assert output.num_samples > 0
            if case.fixture == "kokoro":
                meta = json.loads(Path(output.meta_file).read_text())
                routes = meta["bucket_presets"]["routes"]
                assert [(r["token_max_length"], r["frame_max_length"]) for r in routes] == [(32, 160)]
                assert all(len(c["buckets"]) == 1 for c in meta["components"].values())
                assert output.segments[0]["route"] == "t0032_f0160"
                record["tested_routes"] = ["t0032_f0160"]
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
                            str(meta.parent / adapter["meta_file"]), prompt="hello world",
                            device="cuda:0", max_new_tokens=2, stream_output=False,
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
    run(case, Path(sys.argv[2]), low_memory=True if "--low-memory" in sys.argv[3:] else None)
