"""Reranker semantics with real small tensors and native Qwen3-VL modules."""

import pytest
import torch
from torch import nn


def test_binary_head_preserves_official_difference_and_does_not_mutate_tied_weights():
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.model import binary_score_head

    torch.manual_seed(4)
    head = nn.Linear(128, 256, bias=False)
    before = head.weight.detach().clone()
    hidden = torch.randn(1, 1, 128)
    score_head = binary_score_head(head, 17, 31)
    expected = torch.nn.functional.linear(hidden, (before[17] - before[31])[None])
    actual = score_head(hidden)
    assert actual.shape == (1, 1, 64)
    torch.testing.assert_close(actual[..., :1], expected)
    assert torch.count_nonzero(actual[..., 1:]) == 0
    torch.testing.assert_close(head.weight, before)
    assert score_head.weight.data_ptr() != head.weight.data_ptr()


@pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 128, 129])
def test_chunk_inputs_preserve_positions_deepstack_and_final_valid_token(length):
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import prefill_chunks

    embeds = torch.arange(length * 8).reshape(1, length, 8).float()
    positions = [torch.arange(length) + offset for offset in [0, 7, 19]]
    deep = [embeds + offset for offset in [3, 5, 7]]
    keys, values = [torch.zeros(1)], [torch.zeros(1)]
    full = (embeds, *positions, torch.tensor([0]), torch.tensor([length]), *deep, keys, values)
    chunks = list(prefill_chunks(full, chunk_length=64, context_length=256))
    assert len(chunks) == (length + 63) // 64
    for index, chunk in enumerate(chunks):
        offset = index * 64
        valid = min(64, length - offset)
        assert chunk[0].shape == (1, 64, 8)
        assert chunk[4].item() == offset
        assert chunk[5].item() == valid
        for slot in [0, 6, 7, 8]:
            torch.testing.assert_close(chunk[slot][:, :valid], full[slot][:, offset : offset + valid])
            assert torch.count_nonzero(chunk[slot][:, valid:]) == 0
        for slot in [1, 2, 3]:
            torch.testing.assert_close(chunk[slot][:valid], full[slot][offset : offset + valid])
        assert chunk[9] is keys and chunk[10] is values


def test_chunk_capacity_and_empty_input_fail_before_execution():
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import prefill_chunks

    embeds = torch.zeros(1, 65, 8)
    full = (embeds, *[torch.arange(65)] * 3, torch.tensor([0]), torch.tensor([65]), *[embeds] * 3, [], [])
    with pytest.raises(ValueError, match="context"):
        list(prefill_chunks(full, chunk_length=64, context_length=64))


def test_reranker_messages_preserve_query_document_order_and_reject_video():
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import build_messages

    messages = build_messages(
        {"text": "question", "image": "query.png"},
        {"text": "answer", "image": ["one.png", "two.png"]},
        instruction="Find a relevant document.",
    )
    content = messages[1]["content"]
    assert [item["image"] for item in content if item["type"] == "image"] == ["query.png", "one.png", "two.png"]
    texts = [item["text"] for item in content if item["type"] == "text"]
    assert texts == ["<Instruct>: Find a relevant document.", "<Query>:", "question", "\n<Document>:", "answer"]
    with pytest.raises(ValueError, match="video"):
        build_messages({"video": "clip.mp4"}, {"text": "answer"})
    with pytest.raises(ValueError, match="text or image"):
        build_messages({}, {"text": "answer"})


def test_visual_features_merge_media_axis_without_treating_it_as_batch():
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import merge_visual_features

    groups = [
        (torch.full((1, 4, 8), float(i)), [torch.full((1, 4, 8), float(i + j)) for j in range(3)]) for i in range(2)
    ]
    images, deep = merge_visual_features(groups)
    assert images.shape == (8, 8)
    assert all(item.shape == (8, 8) for item in deep)
    torch.testing.assert_close(images[:4], groups[0][0][0])
    torch.testing.assert_close(images[4:], groups[1][0][0])


def test_recipe_is_discoverable_and_only_advertises_implemented_components():
    from xhmodel_merak.model_catalog import get_recipe

    recipe = get_recipe("qwen3-vl-reranker-2b")
    assert recipe.task == "multimodal-reranking"
    assert recipe.modalities == ("text", "image")
    assert recipe.components == ("visual", "prefill")
    assert recipe.quant_methods == ("autoround",)
    assert ("text_config.num_hidden_layers", 28) in recipe.source_constraints


@pytest.mark.parametrize(
    ("precision", "deployment", "base_type", "local_a16", "quantized"),
    [
        (None, None, "w8a8h1_sefp", False, False),
        ({"weight": "w4", "activation": "a8"}, None, "w8a8h1_sefp", True, True),
        ({"weight": "w4", "activation": "a16"}, None, "w8a16h1_sefp", False, True),
        (None, {"prefill": "w8a16h1_sefp"}, "w8a16h1_sefp", False, False),
        (
            {"weight": "w4", "activation": "a8"},
            {"prefill": "w8a16h1_sefp"},
            "w8a16h1_sefp",
            True,
            True,
        ),
    ],
)
def test_public_recipe_materializes_maintained_precision_without_native_calibration(
    tmp_path, precision, deployment, base_type, local_a16, quantized
):
    import json

    import yaml

    from xhmodel_merak.model_build.resolver import resolve_build_plan
    from xhmodel_merak.model_build.workflow import materialize_workflow_config

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "text_config": {"hidden_size": 2048, "num_hidden_layers": 28},
                "vision_config": {"out_hidden_size": 2048},
            }
        )
    )
    plan = resolve_build_plan(
        model_id="qwen3-vl-reranker-2b",
        model_dir=str(source),
        output_dir=str(tmp_path / "artifact"),
        precision=precision,
        deployment_precision=deployment,
    )
    effective_path = tmp_path / "effective.yaml"
    materialize_workflow_config(plan, effective_path)
    effective = yaml.safe_load(effective_path.read_text())
    assert plan.quant_method == ("autoround" if quantized else None)
    if quantized:
        assert effective["quant"]["method"] == "autoround"
        assert effective["quant"]["bits"] == 4
        assert effective["quant"]["group_size"] == 64
    else:
        assert plan.precision.weight == "w16"
        assert effective["quant"] is None
        assert not any(requirement.startswith("auto-round") for requirement in plan.worker_requirements)
    scheme = effective["export"]["model"]["quant_scheme"]
    assert scheme["quant_type"] == base_type
    assert plan.deployment_precision == {"visual": "w8a8h1_sefp", "prefill": base_type}
    expected_nodes = {"lm_head": {"quant_type": "w8a16h1_sefp"}}
    if local_a16:
        expected_nodes.update(
            {
                "model_language_model_layers_26_mlp_up_proj": {"act_scheme": {"bits": 16}},
                "model_language_model_layers_27_mlp_gate_proj": {"act_scheme": {"bits": 16}},
            }
        )
    assert scheme["nodes"] == expected_nodes
    assert effective["export"]["model"]["visual_config"]["quant_scheme"] == {"quant_type": "w8a8h1_sefp"}


def test_official_legacy_mrope_checkpoint_identity_keeps_every_geometry_value():
    from copy import deepcopy

    from xhmodel_merak.model_build.runtime_assets import behavior_config_mismatches
    from xhmodel_merak.model_catalog import get_recipe

    source = {
        "text_config": {
            "rope_theta": 5000000,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
            },
        },
        "vision_config": {"model_type": "qwen3_vl"},
    }
    checkpoint = {
        "text_config": {"rope_parameters": {"rope_theta": 5000000, **source["text_config"]["rope_scaling"]}},
        "vision_config": {"model_type": "qwen3_vl_vision"},
    }
    aliases = get_recipe("qwen3-vl-reranker-2b").config_type_aliases
    assert behavior_config_mismatches(source, checkpoint, type_aliases=aliases) == []
    for key, wrong in [("rope_theta", 1000000), ("mrope_section", [16, 24, 24]), ("mrope_interleaved", False)]:
        changed = deepcopy(checkpoint)
        changed["text_config"]["rope_parameters"][key] = wrong
        assert behavior_config_mismatches(source, changed, type_aliases=aliases)


@pytest.mark.parametrize("precision", ["w4a8_a16_nodes2", "w4a16"])
def test_w4_workflow_rejects_exporting_native_weights_without_autoround(tmp_path, precision):
    from pathlib import Path

    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.workflow import XHQwen3VLRerankerWorkflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    (tmp_path / "config.json").write_text('{"model_type":"qwen3_vl"}')
    config = Path(__file__).resolve().parents[2] / (
        f"configs_merak/workflows/xh2a/llm_models/qwen3_vl_reranker/2b/qwen3_vl_reranker_2b_xh2a_{precision}.yaml"
    )
    workflow = XHQwen3VLRerankerWorkflow(str(tmp_path), str(config))
    with pytest.raises(ValueError, match="AutoRound"):
        workflow.export(QuantResult(raw_model_dir=str(tmp_path), skipped=True), str(tmp_path / "export"), "cpu")


@pytest.mark.parametrize("quant_type", ["w8a8h1_sefp", "w4a8h0_ssfp", "w8a16h1_sefp", "w4a16h0_ssfp"])
def test_autoround_checkpoint_requires_256_chunk_even_with_w8_base_scheme(tmp_path, quant_type):
    import json
    from pathlib import Path

    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.workflow import XHQwen3VLRerankerWorkflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3_vl", "quantization_config": {"provider": "auto-round", "bits": 4, "group_size": 64}}
        )
    )
    config = Path(__file__).resolve().parents[2] / (
        "configs_merak/workflows/xh2a/llm_models/qwen3_vl_reranker/2b/qwen3_vl_reranker_2b_xh2a_w4a8_a16_nodes2.yaml"
    )
    workflow = XHQwen3VLRerankerWorkflow(str(tmp_path), str(config))
    with pytest.raises(ValueError, match="prefill_chunk_length=256"):
        workflow.export(
            QuantResult(raw_model_dir=str(tmp_path), skipped=True),
            str(tmp_path / "export"),
            "cpu",
            config_overrides={
                "quant": None,
                "export.model.quant_scheme.quant_type": quant_type,
                "export.model.prefill_chunk_length": 512,
            },
        )
    assert not (tmp_path / "export").exists()


def test_score_artifact_rejects_changed_output_anchor_and_missing_runtime_files(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    from xhmodel_merak.model_integrations.qwen3_vl_reranker.artifact import validate_score_contract

    graph = helper.make_graph(
        [], "score_abi", [], [helper.make_tensor_value_info("score_logits", TensorProto.FLOAT16, [1, 1, 64])]
    )
    onnx.save(helper.make_model(graph), tmp_path / "prefill.onnx")
    (tmp_path / "visual.onnx").touch()
    (tmp_path / "quant_embedding.pt").touch()
    contract = {
        "version": 1,
        "output": "score_logits",
        "score_index": 0,
        "head_width": 64,
        "activation": "sigmoid",
        "projection": "yes_minus_no",
    }
    meta = {
        "reranker": contract,
        "prefill_hmonnx": "prefill.onnx",
        "quant_embedding": "quant_embedding.pt",
        "visual_config": {"hmonnx": "visual.onnx"},
    }
    validate_score_contract(tmp_path, meta)
    with pytest.raises(ValueError, match="score contract"):
        validate_score_contract(tmp_path, {**meta, "reranker": {**contract, "score_index": 1}})
    with pytest.raises(ValueError, match="score contract"):
        validate_score_contract(tmp_path, {**meta, "decode_hmonnx": "prefill.onnx"})
    (tmp_path / "quant_embedding.pt").unlink()
    from xhmodel_merak.errors import ModelError

    with pytest.raises(ModelError):
        validate_score_contract(tmp_path, meta)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_logits_to_keep": 0},
        {"output_pre_norm_hidden_states": True},
        {"output_hidden_states_for_export": True},
        {"batch_size": 2},
        {"use_cache": False},
    ],
)
def test_configuration_rejects_embedding_semantics_and_unsupported_execution(overrides):
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.model import XHQwen3VLRerankerModelConfig

    with pytest.raises(ValueError):
        XHQwen3VLRerankerModelConfig(
            model_name="bad", visual_config={"max_size_w": 448, "max_size_h": 448}, **overrides
        )


@pytest.mark.parametrize("flash", [False, True])
@pytest.mark.parametrize("length", [7, 8, 9, 17])
def test_real_wrapped_language_model_chunking_matches_native_and_resets(tmp_path, length, flash):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from xhmodel_merak.xh_llm.infer_mixin import LLMInferenceContextManager
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.model import (
        XHQwen3VLRerankerModel,
        XHQwen3VLRerankerModelConfig,
    )
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import prefill_chunks

    torch.manual_seed(23)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            max_position_embeddings=64,
            rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20]},
            eos_token_id=2,
        ),
        vision_config=dict(
            depth=3,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            out_hidden_size=256,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0, 1, 2],
        ),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    source = Qwen3VLForConditionalGeneration(config).eval().half()
    source.save_pretrained(tmp_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<pad>": 0, "<unk>": 1, "<eos>": 2, "yes": 3, "no": 4}, unk_token="<unk>")
        ),
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(tmp_path)
    wrapped = XHQwen3VLRerankerModel(
        XHQwen3VLRerankerModelConfig(
            model_name="tiny_reranker",
            hf_model=str(tmp_path),
            context_max_length=32,
            prefill_chunk_length=8,
            max_pe_length=64,
            flash_attention={"enable": flash, "q_bits": 8, "k_bits": 8, "v_bits": 8, "p_bits": 8},
            quant_scheme={"quant_type": "w8a8h1_sefp"},
            visual_config={"max_size_w": 64, "max_size_h": 64},
        )
    )
    wrapped.to(dtype=torch.float16, device="cpu")
    wrapped.to_wrap()
    from xhquant.nn import FlashAttention

    assert any(isinstance(module, FlashAttention) for module in wrapped.wrap_model.modules()) == flash
    ids = torch.arange(length)[None] % 20 + 5
    with torch.no_grad():
        hidden = source.model.language_model(input_ids=ids, use_cache=False).last_hidden_state[:, -1:]
        expected = torch.nn.functional.linear(hidden, (source.lm_head.weight[3] - source.lm_head.weight[4])[None])
    results = []
    for _ in range(2):
        wrapped.set_prefill()
        with LLMInferenceContextManager(wrapped), torch.no_grad():
            processor = wrapped._get_data_preprocessor()
            processor.input_sequence_length = length
            full = processor({"input_ids": ids, "past_seq_length": 0})
            for chunk in prefill_chunks(full, chunk_length=8, context_length=32):
                actual = wrapped.forward(*chunk)
            results.append(actual[..., :1].clone())
    torch.testing.assert_close(results[0], expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)


@pytest.mark.parametrize("precision", ["w4a16", "w4a8_a16_nodes2"])
def test_workflows_export_32k_flash_and_dynamic_image_gears(precision):
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / f"configs_merak/workflows/xh2a/llm_models/qwen3_vl_reranker/2b/qwen3_vl_reranker_2b_xh2a_{precision}.yaml"
    )
    workflow = yaml.safe_load(path.read_text())
    if precision.startswith("w4"):
        quant = workflow["quant"]
        assert quant["method"] == "autoround" and quant["iters"] > 0
        assert quant["bits"] == 4 and quant["group_size"] == 64
        assert quant["calibration"]["dataset"].startswith("xh2modelzoo://")
    else:
        assert workflow["quant"] is None
    model = workflow["export"]["model"]
    assert model["context_max_length"] == 32768
    assert model["prefill_chunk_length"] == 256
    assert model["flash_attention"] == dict(enable=True, q_bits=8, k_bits=8, v_bits=8, p_bits=8)
    base_precision = precision.split("_", 1)[0]
    expected = {"w4a8": "w8a8h1_sefp", "w4a16": "w8a16h1_sefp"}.get(base_precision, base_precision + "h1_sefp")
    assert model["quant_scheme"]["quant_type"] == expected
    assert model["quant_scheme"]["nodes"]["lm_head"]["quant_type"] == "w8a16h1_sefp"
    visual = model["visual_config"]
    assert visual["visual_input_mode"] == "patches"
    assert visual["image_token_gears"] == [96, 196, 384, 704, 1536]
    assert visual["image_token_capacity"] == 1536
    assert visual["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
    assert "max_size_w" not in visual and "max_size_h" not in visual


def test_two_node_a16_workflow_binds_exact_fx_names_without_promoting_neighbor_linears():
    from pathlib import Path

    import yaml

    from xhquant.api import QuantScheme, create_quant_config, to_frontend_graph, to_quant_graph

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (
            root / "configs_merak/workflows/xh2a/llm_models/qwen3_vl_reranker/2b/"
            "qwen3_vl_reranker_2b_xh2a_w4a8_a16_nodes2.yaml"
        ).read_text()
    )
    selected = {
        "model.language_model.layers.26.mlp.up_proj",
        "model.language_model.layers.27.mlp.gate_proj",
    }
    scheme = workflow["export"]["model"]["quant_scheme"]
    assert set(scheme["nodes"]) == {name.replace(".", "_") for name in selected} | {"lm_head"}

    class ProjectionPath(nn.Module):
        """A small real Linear graph isolates exact FX-name override binding."""

        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.layers = nn.ModuleDict({str(i): nn.Module() for i in [26, 27]})
            for layer in self.model.language_model.layers.values():
                layer.mlp = nn.Module()
                for name in ["up_proj", "gate_proj", "down_proj"]:
                    setattr(layer.mlp, name, nn.Linear(64, 64, bias=False))
            self.lm_head = nn.Linear(64, 64, bias=False)

        def forward(self, hidden):
            for i in [26, 27]:
                mlp = self.model.language_model.layers[str(i)].mlp
                hidden = mlp.down_proj(mlp.up_proj(hidden) * mlp.gate_proj(hidden))
            return self.lm_head(hidden)

    # W4 carrier admission is covered separately; use native weights here to
    # exercise the SDK's real frontend and per-node activation-scheme resolver.
    frontend = to_frontend_graph(ProjectionPath(), "TorchFX", [torch.ones(1, 64)])
    quantized = to_quant_graph(frontend, "XH2a", create_quant_config(QuantScheme(target_device="XH2a", **scheme)))
    for layer in [26, 27]:
        for projection in ["up_proj", "gate_proj", "down_proj"]:
            name = f"model.language_model.layers.{layer}.mlp.{projection}"
            config = quantized.get_submodule(name)._quant_cfg
            assert config.act_schema.quantizer.qspec.man_bit == (16 if name in selected else 8)
    head = quantized.get_submodule("lm_head")._quant_cfg
    assert head.w_schema.quantizer.qspec.man_bit == 8
    assert head.act_schema.quantizer.qspec.man_bit == 16


@pytest.mark.parametrize("grid", [(1, 4, 4), (1, 2, 8), (1, 8, 2), (1, 4, 8)])
def test_dynamic_visual_matches_native_for_aspect_ratio_padding_and_deepstack(grid):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    from xhmodel_merak.xh_llm.models.qwen3_5.visual_token_gears import (
        build_visual_token_gear_inputs,
        pad_flattened_patches,
    )
    from xhmodel_merak.xh_llm.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel as ExportVisionModel
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.visual import token_vision_scope
    from xhmodel_merak.xh_llm.wrap_model import wrap_llm_model
    from xhquant.api import ConfigDict

    torch.manual_seed(19)
    config = Qwen3VLVisionConfig(
        depth=3,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        out_hidden_size=256,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        deepstack_visual_indexes=[0, 1, 2],
    )
    native = Qwen3VLVisionModel(config).eval()
    target = ExportVisionModel(config).eval()
    target.load_state_dict(native.state_dict())
    with token_vision_scope():
        target = wrap_llm_model(target, ConfigDict(image_token_capacity=8, visual_rope_cache_length=16, patch_size=16))
    grid = torch.tensor([grid])
    pixels = torch.randn(int(grid.prod()), 3 * 2 * 16 * 16)
    padded, valid = pad_flattened_patches(pixels, 32)
    geometry = build_visual_token_gear_inputs(
        grid,
        patch_capacity=32,
        num_position_embeddings=config.num_position_embeddings,
        dtype=torch.float32,
    )
    names = ("position_ids", "position_weights", "rotary_position_ids", "attention_mask")
    with torch.inference_mode():
        expected = native(pixels, grid_thw=grid)
        result = target(padded, *[geometry[name] for name in names])
        padded[:, valid:] = torch.randn_like(padded[:, valid:]) * 100
        changed_padding = target(padded, *[geometry[name] for name in names])
    for actual, reference, changed in zip(
        result, [expected.pooler_output, *expected.deepstack_features], changed_padding, strict=True
    ):
        torch.testing.assert_close(actual[0, : valid // 4], reference, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual[:, : valid // 4], changed[:, : valid // 4], atol=0, rtol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_token_vision_scope_restores_bindings_and_preserves_qwen35(fail):
    from xhmodel_merak.xh_llm.models.qwen3_5 import modeling_qwen3_5 as qwen35
    from xhmodel_merak.xh_llm.models.qwen3_vl import modeling_qwen3_vl as qwen3vl
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.visual import token_vision_scope
    from xhmodel_merak.xh_llm.register import XHLLM_TRACEABLE_MODULES

    registry = XHLLM_TRACEABLE_MODULES
    mappings = (registry._registry, registry._key_registry, registry._dynamic_classes)
    names = ("VisionAttention", "VisionBlock", "VisionModel", "VisionPatchEmbed", "VisionPatchMerger")
    classes = [
        getattr(module, prefix + name)
        for module, prefix in ((qwen35, "Qwen3_5"), (qwen3vl, "Qwen3VL"))
        for name in names
    ]
    before = [{cls: mapping.get(cls) for cls in classes} for mapping in mappings]

    def exercise():
        with token_vision_scope():
            # A nested conversion must restore the outer scope on exit too.
            outer = [{cls: mapping.get(cls) for cls in classes} for mapping in mappings]
            with token_vision_scope():
                assert registry._registry[qwen3vl.Qwen3VLVisionModel] is not before[0][qwen3vl.Qwen3VLVisionModel]
                for mapping, original in zip(mappings, before, strict=True):
                    for cls in classes[:5]:
                        assert mapping.get(cls) is original[cls]
            assert [{cls: mapping.get(cls) for cls in classes} for mapping in mappings] == outer
            if fail:
                raise RuntimeError("conversion failed")

    if fail:
        with pytest.raises(RuntimeError, match="conversion failed"):
            exercise()
    else:
        exercise()
    assert [{cls: mapping.get(cls) for cls in classes} for mapping in mappings] == before
