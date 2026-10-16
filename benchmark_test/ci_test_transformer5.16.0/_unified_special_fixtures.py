# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native hybrid, assistant and multi-adapter fixture factories."""

import copy
import json
from pathlib import Path


def prepare_minicpmo(catalog, recipe, root, calibration, monkeypatch, variant):
    """Keep all eight native components and real GPTQ/export at small shapes."""
    import shutil
    import subprocess
    from dataclasses import replace

    import soundfile as sf
    import torch
    from _unified_assets import minicpmo_source

    torch.manual_seed(4891)
    source = root / "source"
    shutil.copytree(minicpmo_source(), source)
    audio, sample_rate = sf.read(source / "assets/system_ref_audio.wav", dtype="float32")
    sf.write(root / "reference.wav", audio[:sample_rate // 2], sample_rate)
    sf.write(root / "audio.wav", audio[:sample_rate * 2], sample_rate)
    shutil.copyfile(source / "assets/fossil.png", root / "image.png")
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-loop", "1", "-i", str(root / "image.png"),
        "-i", str(root / "audio.wav"), "-t", "0.5" if variant == "offline" else "2",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-threads", "2", "-c:a", "aac", str(root / "video.mp4"),
    ], check=True)
    calibration.write_text("".join(json.dumps({"text": "hello world " * 160}) + "\n" for _ in range(2)))
    overrides = dict(recipe.config_replacements)
    overrides.update({
        "quant.nsamples": 2, "quant.seqlen": 256, "quant.offload_to_disk": False,
        "quant.calibration_jsonl": str(calibration),
        "export.model.context_max_length": 1024 if variant == "offline" else 2048,
        "export.model.prefill_chunk_length": 256,
        "export.components.llm.wrap_cfg.max_sequence_length": 1024 if variant == "offline" else 2048,
        "export.components.llm.wrap_cfg.input_sequence_length": 256,
        "export.components.llm.wrap_cfg.image_slice_max_size": [2, 2],
        "export.components.tts.wrap_cfg.max_sequence_length": 512,
        "export.components.tts.wrap_cfg.image_slice_max_size": [2, 2],
        "export.components.vision.image_slice_max_size": [2, 2],
        "export.components.audio.static_batch_size": 1,
        "export.components.audio.max_audio_frames": 100,
        "export.components.token2wav_flow_frontend.token_capacity": 128,
        "export.components.token2wav_flow_frontend.mel_capacity": 256,
        "export.components.token2wav_flow_decoder.mel_capacity": 256,
        "export.components.token2wav_hift.frame_capacity": 256,
        "export.components.speaker.sequence_length": 64,
        "export.components.speaker.feats_length": 100,
    })
    if variant != "offline":
        from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import export_token2wav

        original_load = export_token2wav.load_token2wav_modules

        def load_small_token2wav(model_dir):
            flow, hift = original_load(model_dir)
            # Upstream nonpersistent buffers hardcode the full-size architecture.
            # Match their geometry to the real tiny layers, preserving ten CFM steps.
            flow.decoder.estimator.cnn_cache_buffer = torch.zeros(2, 2, 64, 2)
            flow.decoder.estimator.att_cache_buffer = torch.zeros(2, 2, 2, 214, 32)
            flow.decoder.cnn_cache_buffer = torch.zeros(10, 2, 2, 64, 2)
            flow.decoder.att_cache_buffer = torch.zeros(10, 2, 2, 2, 214, 32)
            return flow, hift

        monkeypatch.setattr(export_token2wav, "load_token2wav_modules", load_small_token2wav)
        base_shapes = {
            "conformer_cnn_cache": [1, 32, 6], "conformer_att_cache": [2, 1, 2, 58, 32],
            "estimator_cnn_cache": [10, 2, 2, 64, 2], "estimator_att_cache": [10, 2, 2, 2, 58, 32],
        }
        cache_shapes = copy.deepcopy(base_shapes)
        cache_shapes["conformer_att_cache"][3] = 158
        cache_shapes["estimator_att_cache"][4] = 158
        overrides.update({
            "export.components.token2wav_flow_frontend.streaming.prompt_token_capacity": 32,
            "export.components.token2wav_flow_frontend.streaming.prompt_mel_capacity": 58,
            "export.components.token2wav_flow_frontend.streaming.base_cache_valid_length": 58,
            "export.components.token2wav_flow_frontend.streaming.base_conformer_layers": 1,
            "export.components.token2wav_flow_frontend.streaming.base_cache_shapes": base_shapes,
            "export.components.token2wav_flow_frontend.streaming.cache_shapes": cache_shapes,
            "export.components.token2wav_flow_decoder.streaming.base_cache_valid_length": 58,
            "export.components.token2wav_flow_decoder.streaming.estimator_step_cache_shapes.input_cnn": [2, 2, 64, 2],
            "export.components.token2wav_flow_decoder.streaming.estimator_step_cache_shapes.input_att": [2, 2, 2, 158, 32],
            "export.components.token2wav_flow_decoder.streaming.estimator_step_cache_shapes.host_cnn_banks": [10, 2, 2, 64, 2],
        })
    monkeypatch.setitem(catalog._recipes, recipe.model_id, replace(
        recipe, config_replacements=tuple(overrides.items()),
    ))
    return {
        "model_id": recipe.model_id, "model_dir": str(source), "output_dir": str(root / "artifact"),
        "quant_output_dir": str(root / "quant"), "device": "cuda:0", "profile": variant,
    }, {
        "prompt": "Describe briefly.", "video": str(root / "video.mp4"),
        "max_new_tokens": 2 if variant == "offline" else 16,
        "generate_audio": True, "reference_audio": str(root / "reference.wav"),
        **({"language": "en"} if variant == "offline" else {}),
    }


def configure_kokoro_minimal(catalog, recipe, monkeypatch):
    """Reduce only CI profile geometry, keeping build/load validation active.

    Like the other native fixtures, this profile is local to the worker. The
    public four-route profile is still checked by test_kokoro_unified_api.
    """
    from dataclasses import replace
    from xhmodel_merak.model_integrations.kokoro import artifact

    profile = "bucketed-t32f160"
    monkeypatch.setattr(artifact, "_ROUTES", ((32, 160),))
    overrides = dict(recipe.config_replacements)
    overrides.update({"export.token_buckets": [32], "export.audio_seconds_buckets": [4]})
    monkeypatch.setitem(
        catalog._recipes,
        recipe.model_id,
        replace(
            recipe,
            default_profile=profile,
            profiles=(profile,),
            config_replacements=tuple(overrides.items()),
            fixed_artifact_name="hmquant_xh2_kokoro-82m-v1.1-zh_w16a16_t32_f160",
        ),
    )


def configure_minimax_packaging_stub(source, monkeypatch):
    """Keep public packaging/validation, stub only the heavy MiniMax export path.

    Unified CI cannot run the real MiniMax converter. Emit a complete seven-
    component HMONNX fixture so build/relocate still exercise the registered
    aigc adapters, then let inference assert UNSUPPORTED_COMBINATION.
    """
    import json
    from pathlib import Path

    import onnx
    from onnx import TensorProto, helper

    import xhmodel_merak.model_build.executor as build_executor
    import xhmodel_merak.model_build.workflow as build_workflow
    import xhmodel_merak.model_inference.api as inference_api
    from xhmodel_merak.model_integrations.minimax_h3.artifact import COMPONENTS
    from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult

    def write_minimal_hmonnx(path: Path) -> None:
        input_value = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8])
        output_value = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 8])
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            "minimax_ci_fixture",
            [input_value],
            [output_value],
        )
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)

    def create_workflow(model_dir, config_path):
        class StubWorkflow:
            def quant(self, output_dir, device, config_overrides=None):
                del output_dir, device, config_overrides
                return QuantResult(raw_model_dir=str(source), skipped=True, meta={"reason": "unified-ci-stub"})

            def export(self, quant_result, output_dir, device, config_overrides=None):
                del quant_result, device, config_overrides
                out = Path(output_dir)
                config_file = out / "minimax_h3.yaml"
                config_file.write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
                components = {}
                for name in COMPONENTS:
                    relative = f"{name}/{name}.hmonnx"
                    graph_path = out / relative
                    graph_path.parent.mkdir(parents=True, exist_ok=True)
                    write_minimal_hmonnx(graph_path)
                    components[name] = {
                        "hmonnx_file": relative,
                        "input_names": ["x"],
                        "output_names": ["y"],
                        "inputs": [{"name": "x", "shape": [1, 8], "dtype": "float16"}],
                    }
                meta = {
                    "model_type": "XHMiniMaxH3Model",
                    "target_device": "XH2a",
                    "components": components,
                    "config": config_file.name,
                }
                (out / "export_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")
                return ExportResult(work_dir=str(out), config_file=str(config_file), meta=meta)

        return StubWorkflow()

    # Local CI pins can drift from the recipe's blocked_distribution pins.
    # Bypass only the worker pin gate; packaging/validation still run.
    monkeypatch.setattr(build_workflow, "create_workflow", create_workflow)
    monkeypatch.setattr(build_executor, "validate_worker_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_executor, "validate_worker_capabilities", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference_api, "validate_worker_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference_api, "validate_worker_capabilities", lambda *args, **kwargs: None)


def configure_minicpm_v47_packaging_stub(source, monkeypatch, *, recipe):
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
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")

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
                    relative = f"vision_{mode}/hmonnx/vision_{mode}.onnx"
                    graph_path = out / relative
                    graph_path.parent.mkdir(parents=True, exist_ok=True)
                    write_minimal_hmonnx(graph_path)
                    vision[mode] = {
                        "downsample_mode": mode,
                        "token_capacity": 64,
                        "positions_per_side": 8,
                        "quant_type": "w8a8h1_sefp",
                        "hmonnx": relative,
                    }

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
                meta = {
                    "schema_version": 1,
                    "model_type": "MiniCPM-V-4.7",
                    "merak_model_type": merak_model_type,
                    "hf_model": str(source),
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


def write_ling(source):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    from xhmodel_merak.xh_llm.models.ling_3_flash._compat import (
        load_ling_config,
        patch_ling_remote_code_compatibility,
    )

    bundled = Path(__file__).with_name("fixtures") / "ling3_flash"
    source.mkdir()
    for name in ("configuration_bailing_moe_v3.py", "modeling_bailing_moe_v3.py"):
        (source / name).write_bytes((bundled / name).read_bytes())
    config = json.loads((bundled / "config.json").read_text())
    config.update(num_hidden_layers=2, layer_group_size=2)
    (source / "config.json").write_text(json.dumps(config))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4891)
        native = AutoModelForCausalLM.from_config(
            load_ling_config(str(source)), trust_remote_code=True, attn_implementation="eager"
        )
        # The upstream KDA constructor leaves dt_bias as torch.empty because
        # deployed models load it from a checkpoint. This synthetic checkpoint
        # must supply a defined value instead of saving allocator contents.
        with torch.no_grad():
            for name, parameter in native.named_parameters():
                if name.endswith(".attention.dt_bias"):
                    parameter.zero_()
    patch_ling_remote_code_compatibility()
    native.half().save_pretrained(source)
    vocab = {str(i): i for i in range(128)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="3"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="0", bos_token="1", eos_token="2", unk_token="3"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.save_pretrained(source)
    return config


def write_gemma_assistant(source, target, *, ordered):
    import torch
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model import _resolve_assistant_text_types

    family = target["model_type"]
    text = copy.deepcopy(target["text_config"])
    text.update(
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        enable_moe_block=False,
        num_experts=None,
        top_k_experts=None,
        moe_intermediate_size=None,
        use_double_wide_mlp=False,
    )
    config_cls, model_cls, _ = _resolve_assistant_text_types({"model_type": family + "_assistant"})
    config = config_cls(**text)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4891)
        model = model_cls(config)
    generator = torch.Generator().manual_seed(4891)
    from safetensors.torch import save_file

    state = {
        "model." + k: v
        for k, v in model.state_dict().items()
        if not any(part in k for part in (".k_proj.", ".v_proj.", ".k_norm."))
    }
    state.update(
        {
            "pre_projection.weight": torch.randn(64, 128, generator=generator) * 0.02,
            "post_projection.weight": torch.randn(64, 64, generator=generator) * 0.02,
        }
    )
    if ordered:
        state.update(
            {
                "masked_embedding.centroids.weight": torch.randn(8, 64, generator=generator) * 0.02,
                "masked_embedding.token_ordering": torch.randperm(512, generator=generator),
            }
        )
    source.mkdir()
    save_file(
        {k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous() for k, v in state.items()},
        source / "model.safetensors",
    )
    raw = {
        "model_type": family + "_assistant",
        "text_config": config.to_dict(),
        "backbone_hidden_size": 64,
        "use_ordered_embeddings": ordered,
        "num_centroids": 8,
        "centroid_intermediate_top_k": 2,
    }
    (source / "config.json").write_text(json.dumps(raw))
    return {
        "export.model.mtp_config." + k: v
        for k, v in {
            "assistant_num_hidden_layers": 2,
            "assistant_layer_pattern": ["sliding_attention", "full_attention"],
            "assistant_hidden_size": 64,
            "assistant_num_attention_heads": 4,
            "assistant_num_key_value_heads": config.per_layer_config[0].num_key_value_heads,
            "assistant_num_global_key_value_heads": config.to_dict()
            .get("per_layer_config", {})
            .get("1", {})
            .get("num_key_value_heads", 0),
            "head_dim": 16,
        }.items()
    }


def write_macaron_loras(source):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((source / "config.json").read_text())["text_config"]
    with safe_open(source / "model.safetensors", framework="pt") as checkpoint:
        shapes = {key: checkpoint.get_slice(key).get_shape() for key in checkpoint.keys()}
    rank = 2
    for index in range(4):
        tensors = {}
        for key, shape in shapes.items():
            if len(shape) != 2 or not key.endswith(".weight") or ".layers." not in key:
                continue
            if not any(part in key for part in (".linear_attn.", ".self_attn.", ".mlp.shared_expert.")):
                continue
            module = key.removeprefix("model.language_model.").removesuffix(".weight")
            prefix = f"base_model.model.model.{module}"
            tensors[prefix + ".lora_A.weight"] = torch.full((rank, shape[1]), 0.01 * (index + 1))
            tensors[prefix + ".lora_B.weight"] = torch.full((shape[0], rank), 0.02 * (index + 1))
        for layer in range(config["num_hidden_layers"]):
            e, h, m = config["num_experts"], config["hidden_size"], config["moe_intermediate_size"]
            for suffix, a, b in (("base_layer", (e * rank, h), (2 * m, e * rank)), ("", (e * rank, m), (h, e * rank))):
                prefix = f"base_model.model.model.layers.{layer}.mlp.experts" + ("." + suffix if suffix else "")
                tensors[prefix + ".lora_A.weight"] = torch.full(a, 0.01 * (index + 1))
                tensors[prefix + ".lora_B.weight"] = torch.full(b, 0.02 * (index + 1))
        folder = source / "loras" / f"L{index}"
        folder.mkdir(parents=True)
        (folder / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "bias": "none",
                    "r": rank,
                    "lora_alpha": rank * 2,
                    "target_parameters": ["experts.gate_up_proj", "experts.down_proj"],
                }
            )
        )
        save_file(tensors, folder / "adapter_model.safetensors")
