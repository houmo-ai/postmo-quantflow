"""Canonical mixed GPTQ loading and the public, portable streamed build."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from _glm53_packed_fixture import write_packed_glm53
from safetensors import safe_open

from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.xh_llm.models.glm5_3_flash.checkpoint import Glm53Checkpoint
from xhmodel_merak.xh_llm.models.glm5_3_flash.contract import DSA
from xhquant.xhonnxruntime.parsers.llm_cache import LLMCache


@pytest.fixture
def packed_checkpoint(tmp_path):
    source, checkpoint = tmp_path / "source", tmp_path / "checkpoint"
    reference, names = write_packed_glm53(source, checkpoint)
    return source, checkpoint, reference, names


@pytest.fixture
def autoround_packed_checkpoint(tmp_path):
    source, checkpoint = tmp_path / "source", tmp_path / "checkpoint"
    reference, names = write_packed_glm53(source, checkpoint, autoround_format=True)
    return source, checkpoint, reference, names


def test_packed_loading_preserves_words_and_clamped_experts(packed_checkpoint):
    from xhquant.nn.modules.gptq_packed import GPTQPackedLinear
    from xhmodel_merak.xh_llm.models.glm5_3_flash.modeling import StaticMoE

    _, checkpoint, reference, names = packed_checkpoint
    reader = Glm53Checkpoint(checkpoint)
    loaded = [reader.load_layer(i) for i in range(2)]
    with safe_open(checkpoint / "model.safetensors", framework="pt") as tensors:
        for name in names:
            index, relative = name.removeprefix("model.language_model.layers.").split(".", 1)
            layer = loaded[int(index)].get_submodule(relative)
            assert isinstance(layer, GPTQPackedLinear)
            packed = layer.packed_weight
            assert packed.bits == (4 if ".experts." in name else 8)
            assert packed.group_size == 64 and packed.qzero_format == 2
            assert layer.weight is None
            torch.testing.assert_close(packed.qweight, tensors.get_tensor(name + ".qweight"), rtol=0, atol=0)
            torch.testing.assert_close(packed.dequantize(dtype=torch.float16), reference.get_submodule(name).weight)
    x = torch.randn(1, 8, 128).half() * 20
    with torch.no_grad():
        expected = reference.model.language_model.layers[1].mlp(x)
        actual = StaticMoE(loaded[1].mlp)(x)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("field,value", [("bits", 4), ("group_size", 128), ("sym", False), ("desc_act", True)])
def test_rejects_incompatible_checkpoint_profile(packed_checkpoint, field, value):
    _, checkpoint, _, _ = packed_checkpoint
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"][field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="W8 / routed W4 G64"):
        Glm53Checkpoint(checkpoint)


@pytest.mark.parametrize("value", [float("nan"), 0.0, -0.01])
def test_rejects_invalid_packed_scales(packed_checkpoint, value):
    from safetensors.torch import load_file, save_file

    _, checkpoint, _, names = packed_checkpoint
    tensors = load_file(checkpoint / "model.safetensors")
    key = names[0] + ".scales"
    tensors[key] = tensors[key].clone()
    tensors[key][0, 0] = value
    temporary = checkpoint / "invalid.safetensors"
    save_file(tensors, temporary)
    temporary.replace(checkpoint / "model.safetensors")
    with pytest.raises(ValueError, match="Invalid GPTQ scales"):
        Glm53Checkpoint(checkpoint).load_layer(0)


def test_public_recipe_and_quant_parameter_validation(tmp_path):
    from xhmodel_merak.model_build import describe_model
    from xhmodel_merak.xh_llm.models.glm5_3_flash.quant_adapter import quantize_glm53

    recipe = get_recipe("glm-5.3-flash")
    assert recipe.modalities == ("text", "image", "video")
    assert recipe.default_precision.components["routed_expert"]["weight"] == "w4"
    description = describe_model(recipe.model_id, full=True)
    assert description["model_id"] == recipe.model_id
    assert description["quant_capabilities"] == {
        "autoround": ["gptqmodel.recipes.glm5_3_flash_autoround:quantize_glm5_3_flash_autoround"]
    }
    assert recipe.quant_checkpoint_methods == ("autoround",)
    assert recipe.checkpoint_auxiliary_layers_field == "num_nextn_predict_layers"
    with pytest.raises(ValueError, match="not consumed"):
        quantize_glm53(
            model_dir=str(tmp_path),
            output_dir=str(tmp_path / "q"),
            device="cpu",
            config={"runtime": {"misspelled_knob": True}},
        )
    with pytest.raises(ValueError, match="shared experts must be W8"):
        quantize_glm53(
            model_dir=str(tmp_path),
            output_dir=str(tmp_path / "q"),
            device="cpu",
            config={"moe": {"shared_expert_bits": 4}},
        )


def test_public_glm53_plan_requires_low_memory_export(tmp_path):
    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.model_build.resolver import resolve_build_plan

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {"hidden_size": 4096, "num_hidden_layers": 45},
    }))
    kwargs = dict(model_id="glm-5.3-flash", model_dir=str(source), output_dir=str(tmp_path / "output"))
    plan = resolve_build_plan(**kwargs)
    assert plan.low_memory is True
    assert plan.environment["HUGE_MODEL_EXPORT_ENABLED"] == "1"
    with pytest.raises(ModelError, match="Export mode is not registered"):
        resolve_build_plan(**kwargs, low_memory=False)


def test_workflow_quant_calls_public_recipe_and_preserves_owned_root(tmp_path, monkeypatch):
    import gptqmodel.recipes.glm5_3_flash_autoround as producer
    from xhmodel_merak.workflows import AutoWorkflow

    execute = producer.quantize_glm5_3_flash_autoround
    # Only the external expensive producer execution is replaced by its real
    # dry-run contract; AutoRound numerical behavior is exercised in its quant lane.
    monkeypatch.setattr(producer, "quantize_glm5_3_flash_autoround", lambda **kw: execute(**kw, dry_run=True))
    source, output = tmp_path / "source", tmp_path / "owned_quant_root"
    source.mkdir()
    output.mkdir()
    identity = output.stat().st_ino
    workflow = AutoWorkflow.from_config(
        model_dir=str(source), config_path=get_recipe("glm-5.3-flash").default_variant.workflow_config
    )
    result = workflow.quant(output_dir=str(output), device="cuda:0")
    assert result.quanted_model_dir == str(output / "checkpoint")
    assert output.stat().st_ino == identity
    assert result.meta["method"] == "autoround" and result.meta["precision"]["routed_non_shared_experts"] == "W4 G64"
    assert result.meta["provenance"]["iters"] == 200
    assert result.meta["provenance"]["device_map"] == "0"
    assert result.meta["provenance"]["nblocks"] == 1
    assert result.meta["provenance"]["batch_size"] == 1
    assert result.meta["provenance"]["gradient_accumulate_steps"] == 2
    assert result.meta["calibration"]["nsamples"] == 128
    assert Path(result.meta["calibration"]["jsonl"]).is_file()


@pytest.mark.parametrize("legacy_selection", [False, True], ids=["pooled", "legacy"])
def test_public_streamed_build_move_load_and_generate(
    autoround_packed_checkpoint, tmp_path, monkeypatch, legacy_selection
):
    from test_glm53_multimodal import processor_inputs
    from xhmodel_merak.model_build import build_model
    from xhmodel_merak.model_build import checkpoint as checkpoint_api, resolver
    from xhmodel_merak.model_integrations.glm53.artifact import audit_glm53_hmonnx, validate_glm53_bundle
    from xhmodel_merak.xh_llm.hmonnx.auto_llm_hmonnx_model import AutoLLMHONNXModel

    if legacy_selection:
        import xhquant.nn as xhnn

        from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import LatentCacheGather
        from xhmodel_merak.xh_llm.models.glm5_3_flash.glm53_model import XHGlm53Model
        from xhmodel_merak.xh_llm.models.glm5_3_flash.modeling import StaticDSA

        class LegacySelection(torch.nn.Module):
            """The pre-fusion MatMul/ReLU/TopK/Gather graph and its state ABI."""

            def __init__(self, spec, scale, length):
                super().__init__()
                self.scale, self.length, self.ratio = scale, length, spec.pool_size
                self.topk = xhnn.TopK(spec.selected_pools, axis=-1)
                self.gather_validity = xhnn.GatherElements(axis=2)
                self.gather = LatentCacheGather(feature_dim=spec.latent_dim)
                self.register_buffer("offsets", torch.arange(self.ratio, dtype=torch.int32), persistent=False)

            def forward(self, query, pool, payload, validity, weights):
                scores = torch.relu(query.float() @ pool.float().transpose(1, 2).unsqueeze(1) * self.scale)
                scores = (weights.float() @ scores).squeeze(-2)
                scores = scores * validity + (validity - 1.0) * 65504.0
                selected = self.topk(scores)[1]
                selected_valid = self.gather_validity(validity, selected)
                indices = (selected.unsqueeze(-1) * self.ratio + self.offsets).flatten(-2)
                mask = selected_valid.unsqueeze(-1).expand(1, self.length, -1, self.ratio).flatten(-2)
                return self.gather(payload, indices), mask

        initialize = StaticDSA.__init__
        extra_metadata = XHGlm53Model._extra_export_metadata

        def initialize_legacy(self, source, spec, input_length):
            initialize(self, source, spec, input_length)
            self.select_kv = LegacySelection(spec, source.indexer.softmax_scale, input_length)

        def legacy_metadata(self, output_dir, meta_info):
            result = extra_metadata(self, output_dir, meta_info)
            del result.glm53_dsa_selection
            return result

        monkeypatch.setattr(StaticDSA, "__init__", initialize_legacy)
        monkeypatch.setattr(XHGlm53Model, "_extra_export_metadata", legacy_metadata)

    source, checkpoint, reference, _ = autoround_packed_checkpoint
    # Relax only release dimensions for this synthetic native model. All real
    # resolver, checkpoint validation, export and packaging hooks still execute.
    recipe = replace(
        get_recipe("glm-5.3-flash"),
        source_constraints=(("model_type", "glm5_next"),),
        max_pe_length=128,
        config_replacements=(
            ("export.model.linear_chunk_size", 8),
            ("export.model.placeholder_export_workers", 1),
            ("export.model.vision_max_patches", 32),
        ),
    )
    monkeypatch.setattr(resolver, "get_recipe", lambda _: recipe)
    monkeypatch.setattr(checkpoint_api, "get_recipe", lambda _: recipe)
    result = build_model(
        model_id=recipe.model_id,
        model_dir=str(source),
        output_dir=str(tmp_path / "build"),
        quant_checkpoint=str(checkpoint),
        device="cuda:0",
        context_length=64,
        prefill_chunk_length=8,
    )
    assert result.loadable and result.components == ("prefill", "decode", "vision")
    moved = tmp_path / "relocated"
    Path(result.output_dir).rename(moved)
    shutil.rmtree(source)
    shutil.rmtree(checkpoint)
    meta_path = moved / "golden_meta_info.json"
    meta = json.loads(meta_path.read_text())
    validate_glm53_bundle(moved, meta)
    assert all(meta[f"{stage}_hmonnx"].startswith(f"{stage}/") for stage in ("prefill", "decode"))
    assert (moved / "prefill/weights.bin").samefile(moved / "decode/weights.bin")
    export_index = json.loads((moved / "export_meta_info.json").read_text())
    for stage in ("prefill", "decode"):
        assert export_index[stage] == {"path": meta[f"{stage}_hmonnx"], "type": stage}
    for component in meta["glm53_vision"]["components"]:
        assert export_index[f"visual_{component['name']}"] == {
            "path": component["hmonnx"], "type": "hmonnx",
        }
    assert export_index["quant_embedding_bin"]["type"] == "embedding"
    assert export_index["golden_meta_info"] == {"path": "golden_meta_info.json", "type": "json"}
    report = audit_glm53_hmonnx(moved, meta)
    expected_report = {
        "w4g64_moe_blocks": 1,
        "w8_linears": 26,
        "floating_gate_linears": 3,
        "w8_constant_matmuls": 2,
        "w8_convolutions": 1,
        "mhc_sinkhorn_blocks": 4,
    }
    if not legacy_selection:
        expected_report["sparse_kv_select_blocks"] = 1
    assert report["prefill"] == expected_report
    assert report["decode"] == report["prefill"] and report["vision_components"] == 3
    if legacy_selection:
        assert "glm53_dsa_selection" not in meta
        for stage in ("prefill", "decode"):
            import onnx

            graph = onnx.load(moved / meta[f"{stage}_hmonnx"], load_external_data=False)
            assert not any(node.op_type == "SparseAttentionSelectKV" for node in graph.graph.node)
            assert any(node.op_type == "TopK" for node in graph.graph.node)
        monkeypatch.delattr(xhnn, "SparseAttentionSelectKVPool")
    else:
        assert meta["glm53_dsa_selection"] == "hmfp_pool"
    # A saved constant projection silently left at W16 used to escape the
    # Linear/MoE-only audit. Corrupt its real graph attribute, then restore it.
    import onnx

    graph_path = moved / meta["decode_hmonnx"]
    original_graph = graph_path.read_bytes()
    graph = onnx.load(graph_path, load_external_data=False)
    initializers = {value.name for value in graph.graph.initializer}
    projection = next(n for n in graph.graph.node if n.op_type == "MatMul" and n.input[1] in initializers)
    next(a for a in projection.attribute if a.name == "hmfp_weight_man_bit").i = 16
    try:
        onnx.save(graph, graph_path)
        with pytest.raises(ValueError, match="constant RHS MatMul precision"):
            audit_glm53_hmonnx(moved, meta)
    finally:
        graph_path.write_bytes(original_graph)
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    monkeypatch.setenv("XHQUANT_PACKED_W4", "1")
    monkeypatch.setenv("XHQUANT_DISABLE_INITIALIZER_DIGEST_CACHE", "1")
    model = AutoLLMHONNXModel.from_pretrained(str(meta_path), device_map=["cuda:0"])
    for graph_model in (model.prefill_model, model.decode_model):
        writers = [
            module
            for module in graph_model.hmonnx_session.graph_module.modules()
            if isinstance(module, LLMCache) and not module.only_handle_old_cache and module.attention_max_length < 0
        ]
        assert len(writers) == 2 * model.cache_spec.layer_types.count(DSA)
        assert {module.input_names[3] for module in writers} == {
            f"layer_{i}_{suffix}_input"
            for i, kind in enumerate(model.cache_spec.layer_types)
            if kind == DSA
            for suffix in ("latent", "pool")
        }
        assert all(module.reuse_output_storage for module in writers)
    reference = reference.cuda().eval()
    processed = processor_inputs(model.get_processor()).to(model.device)
    ids = processed["input_ids"].cuda()
    media = {k: v.cuda() for k, v in processed.items() if isinstance(v, torch.Tensor) and k != "input_ids"}
    with torch.no_grad():
        expected = reference(input_ids=ids, **media).logits[:, -1:]
        output = model.generate(**processed, max_new_tokens=3, do_sample=False, eos_token_id=None)
        assert output.shape == (1, ids.shape[1] + 3)
        torch.testing.assert_close(
            model.generate(**processed, max_new_tokens=3, do_sample=False, eos_token_id=None), output
        )
        # A separately reset prefill verifies logits, not merely graph shape.
        from xhmodel_merak.xh_llm.models.glm5_3_flash.vision import fuse_multimodal_embeddings

        embed = fuse_multimodal_embeddings(
            ids,
            model.embed_tokens,
            model.vision_encoder,
            reference.config,
            **{k: v for k, v in media.items() if k.startswith(("pixel_values", "image_grid", "video_grid"))},
        )
        model.set_prefill()
        model.get_kvcache_mixin().clear_kv_cache()
        for start in range(0, ids.shape[1], 8):
            actual = model.forward(
                *model.get_data_preprocessor()({"inputs_embeds": embed[:, start : start + 8], "past_seq_length": start})
            )
        cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
        assert torch.isfinite(actual).all() and cosine > 0.98
    model.release_inference_model()
    del model, reference
    import gc

    gc.collect()
    # The public inference API validates the manifest, portable paths and the
    # GLM adapter, and exercises image and actual MP4 input through the demo.
    import av
    import numpy as np
    from PIL import Image
    from xhmodel_merak.model_inference import api as inference_api

    monkeypatch.setattr(inference_api, "get_recipe", lambda _: recipe)
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 12), color=(180, 60, 20)).save(image)
    video = tmp_path / "video.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width, stream.height, stream.pix_fmt = 8, 12, "yuv420p"
        for value in (30, 180, 60, 210):
            frame = av.VideoFrame.from_ndarray(np.full((12, 8, 3), value, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    for media_name, path in (("image", image), ("video", video)):
        result = inference_api.inference_model(
            artifact_dir=str(moved),
            device="cuda:0",
            inputs={"prompt": "t3", media_name: str(path), "max_new_tokens": 2},
        )
        assert 0 < result.output_tokens <= 2 and result.latency_s > 0
        assert result.components == ("prefill", "decode", "vision")
    # Corrupt component identity after successful load: publication validation
    # must fail closed instead of silently accepting an incompatible vision ABI.
    meta["glm53_vision"]["components"][0]["hmonnx"] = "../outside.onnx"
    from xhmodel_merak.errors import ModelError

    with pytest.raises(ModelError, match="Invalid GLM vision path"):
        validate_glm53_bundle(moved, meta)


def test_pooled_build_requires_frontend_only_when_building(monkeypatch):
    import xhquant.nn as xhnn

    from xhmodel_merak.xh_llm.models.glm5_3_flash.glm53_model import XHGlm53Model

    monkeypatch.delattr(xhnn, "SparseAttentionSelectKVPool")
    # Capability failure precedes configuration/checkpoint access.
    with pytest.raises(RuntimeError, match="export requires xhquant.nn.SparseAttentionSelectKVPool"):
        XHGlm53Model(None)
