# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from _unified_preparation import prepare_context


def prepare(case, root, monkeypatch):
    catalog, recipe, source, calibration, image, kwargs, inputs, replacements, family = prepare_context(case, root)
    # Packaging-only unified case: stub the heavy vision+LLM converter,
    # keep public build/relocate/validation, and assert unsupported inference.
    configure_minicpm_v47_packaging_stub(
        source, monkeypatch, recipe=recipe, variant=case.variant,
    )
    if case.variant == "dspark":
        kwargs.update(
            speculative={"mode": "dspark", "num_draft_tokens": 3},
            sources={"dspark": str(source / "draft")},
        )
    inputs = {
        "prompt": "describe the image briefly",
        "image": str(image),
        "max_new_tokens": 2,
        "downsample_mode": "4x",
    }
    return kwargs, inputs


def check_inference(moved, inputs, record):
    import pytest
    from xhmodel_merak.model_inference import inference_model

    from xhmodel_merak.errors import ErrorCode, ModelError
    with pytest.raises(ModelError) as exc_info:
        inference_model(artifact_dir=str(moved), inputs=inputs, device='cuda:0')
    error = exc_info.value
    values = error.details.get('values', {})
    assert error.code == ErrorCode.UNSUPPORTED_COMBINATION
    assert values.get('image') == inputs['image']
    assert values.get('downsample_mode') == inputs['downsample_mode']
    record.update(passed=True, stage='complete', inference={'status': 'unsupported_combination', 'code': error.code.value, 'prompt': values.get('prompt'), 'image': values.get('image'), 'downsample_mode': values.get('downsample_mode')})


def configure_minicpm_v47_packaging_stub(source, monkeypatch, *, recipe, variant="full"):
    """Keep public packaging/validation, stub only the heavy MiniCPM-V-4.7 export.

    Unified CI cannot run the real vision+LLM converter. Emit a complete root-
    metadata fixture so build/relocate still exercise the registered adapters,
    then force unified inference to assert UNSUPPORTED_COMBINATION.
    """
    import json
    from pathlib import Path

    import onnx
    import torch
    from onnx import TensorProto, helper

    import xhmodel_merak.model_build.executor as build_executor
    import xhmodel_merak.model_build.workflow as build_workflow
    import xhmodel_merak.model_inference.api as inference_api
    import xhmodel_merak.model_integrations.minicpm_v_4_7.inference as minicpm_inference
    from xhmodel_merak.errors import unsupported_combination
    from xhmodel_merak.model_inference.inputs import require_local_file, validate_inputs
    from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult

    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    # Nested-dict form is accepted by source_constraint_mismatches for dotted
    # paths such as architectures.0 and text_config.hidden_size.
    config: dict = {}
    for dotted, value in recipe.source_constraints:
        cursor = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    config["text_config"]["vocab_size"] = 8
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if variant == "dspark":
        draft = source / "draft"
        draft.mkdir()
        (draft / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3DSparkModel"],
            "hidden_size": config["text_config"]["hidden_size"],
            "vocab_size": 8,
            "num_target_layers": config["text_config"]["num_hidden_layers"],
            "target_layer_ids": [3], "mask_token_id": 4, "block_size": 3,
            "markov_rank": 8, "markov_head_type": "vanilla",
        }), encoding="utf-8")

    def write_minimal_hmonnx(path: Path) -> None:
        input_value = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8])
        output_value = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 8])
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            "minicpm_v47_ci_fixture",
            [input_value],
            [output_value],
        )
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)

    merak_model_type = (
        "MiniCPMV47MoeForConditionalGeneration"
        if "35b" in recipe.model_id
        else "MiniCPMV47ForConditionalGeneration"
    )
    hidden_size = next(
        (
            value
            for dotted, value in recipe.source_constraints
            if dotted.endswith("hidden_size") and isinstance(value, int)
        ),
        None,
    )
    if hidden_size is None:
        raise ValueError(f"MiniCPM recipe {recipe.model_id!r} is missing a hidden_size constraint")

    def create_workflow(model_dir, config_path):
        class StubWorkflow:
            def quant(self, output_dir, device, config_overrides=None):
                del output_dir, device, config_overrides
                return QuantResult(raw_model_dir=str(source), skipped=True, meta={"reason": "unified-ci-stub"})

            def export(self, quant_result, output_dir, device, config_overrides=None):
                del quant_result, device, config_overrides
                out = Path(output_dir)
                config_file = out / Path(config_path).name
                config_file.write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

                vision = {}
                for mode in ("4x", "16x"):
                    vision[mode] = {
                        "schema_version": 2,
                        "downsample_mode": mode,
                        "token_unit": "post_spatial_merge",
                        "image_token_gears": ([96, 196, 384, 704, 1536] if mode == "4x" else [24, 49, 96, 176, 384]),
                        "routing_policy": "smallest_fit",
                        "overflow_policy": "reject",
                        "gear_manifest": f"visual_{mode}_gears.json",
                        "gears": [],
                    }
                    for gear in vision[mode]["image_token_gears"]:
                        relative = f"visual_{mode}_m{gear}/hmonnx/visual.onnx"
                        graph_path = out / relative
                        graph_path.parent.mkdir(parents=True, exist_ok=True)
                        write_minimal_hmonnx(graph_path)
                        vision[mode]["gears"].append({
                            "downsample_mode": mode,
                            "image_token_capacity": gear,
                            "token_capacity": gear * (4 if mode == "4x" else 16),
                            "positions_per_side": 8,
                            "quant_type": "w8a8h1_sefp",
                            "hmonnx": relative,
                        })
                    (out / vision[mode]["gear_manifest"]).write_text(json.dumps(vision[mode]), encoding="utf-8")

                artifact_dir = Path("llm/hmquant_minicpm_v47_ci")
                llm_root = out / artifact_dir
                (llm_root / "prefill").mkdir(parents=True, exist_ok=True)
                (llm_root / "decode").mkdir(parents=True, exist_ok=True)
                (llm_root / "hf_config").mkdir(parents=True, exist_ok=True)
                metadata_rel = str(artifact_dir / "golden_meta_info.json")
                prefill_rel = str(artifact_dir / "prefill" / "prefill_with_act.onnx")
                decode_rel = str(artifact_dir / "decode" / "decode_with_act.onnx")
                embedding_rel = str(artifact_dir / "quant_embedding.pt")
                hf_config_rel = str(artifact_dir / "hf_config")
                (out / metadata_rel).write_text(json.dumps({"model_name": "minicpm_v47_ci"}), encoding="utf-8")
                write_minimal_hmonnx(out / prefill_rel)
                write_minimal_hmonnx(out / decode_rel)
                # Match packaged hf_config text_config.hidden_size for embedding normalization.
                torch.save({"weight": torch.zeros(8, hidden_size)}, out / embedding_rel)
                (out / hf_config_rel / "config.json").write_text(
                    (source / "config.json").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                processor_root = out / "hf_config"
                processor_root.mkdir()
                (processor_root / "config.json").write_text(json.dumps(config), encoding="utf-8")
                meta = {
                    "schema_version": 1,
                    "model_type": "MiniCPM-V-4.7",
                    "merak_model_type": merak_model_type,
                    "hf_model": "hf_config",
                    "target_device": "XH2a",
                    "config": config_file.name,
                    "components": ["vision_4x", "vision_16x", "llm"],
                    "vision": vision,
                    "llm": {
                        "quant_type": "w8a8h1_sefp",
                        "artifact_dir": str(artifact_dir),
                        "metadata": metadata_rel,
                        "prefill_hmonnx": prefill_rel,
                        "decode_hmonnx": decode_rel,
                        "quant_embedding": embedding_rel,
                        "hf_config": hf_config_rel,
                        "context_max_length": 8192,
                        "prefill_chunk_length": 256,
                    },
                }
                if variant == "dspark":
                    draft_contract = {"noise_token_id": 4, "architecture": "Qwen3DSparkModel"}
                    for name in ("context", "context_decode", "decode"):
                        graph = out / f"dspark_{name}" / "draft.onnx"
                        graph.parent.mkdir()
                        write_minimal_hmonnx(graph)
                        draft_contract[f"{name}_hmonnx"] = str(graph.relative_to(out))
                    meta["llm"]["spec_decode"] = {
                        "mode": "dspark", "num_draft_tokens": 3, "verify_length": 4,
                        "draft": draft_contract,
                    }
                    meta["components"].append("dspark")
                (out / "export_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")
                return ExportResult(work_dir=str(out), config_file=str(config_file), meta=meta)

        return StubWorkflow()

    def stub_run(
        self,
        *,
        recipe,
        root,
        manifest,
        meta_file,
        inputs,
        device,
        components,
        selected_golden,
        golden_artifacts,
    ):
        del self, manifest, selected_golden, golden_artifacts
        normalized = validate_inputs(recipe, inputs)
        image = require_local_file(recipe, "image", normalized["image"])
        raise unsupported_combination(
            recipe.model_id,
            (
                "MiniCPM-V-4.7 unified CI uses a packaging stub; "
                "full multimodal HMONNX inference remains outside this matrix"
            ),
            device=device,
            components=list(components),
            prompt=normalized["prompt"],
            image=str(image),
            downsample_mode=normalized.get("downsample_mode", "4x"),
            meta_file=str(meta_file),
            artifact_dir=str(root),
        )

    monkeypatch.setattr(build_workflow, "create_workflow", create_workflow)
    monkeypatch.setattr(build_executor, "validate_worker_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_executor, "validate_worker_capabilities", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference_api, "validate_worker_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference_api, "validate_worker_capabilities", lambda *args, **kwargs: None)
    monkeypatch.setattr(minicpm_inference.adapter, "run", stub_run.__get__(minicpm_inference.adapter, type(minicpm_inference.adapter)))
