"""Latest MiniCPM canvas inputs and DSpark's distinct proposal contract."""

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
import os

from _minicpm_v47_fixture import minicpmv47_modeling  # noqa: F401

import pytest
import torch

from xhmodel_merak.xh_llm.models.minicpm_v_4_7.draft_checkpoint import (
    read_draft_checkpoint,
    validate_draft_target,
)
from xhmodel_merak.model_catalog.registry import get_recipe


@pytest.mark.parametrize("dspark", (False, True))
@pytest.mark.parametrize("explicit", (False, True))
@pytest.mark.parametrize("raises", (False, True))
def test_draft_export_environment_restores_defaults_and_explicit_overrides(monkeypatch, dspark, explicit, raises):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.text_model import _draft_export_environment

    flags = ("XHQUANT_STRICT_EXPORT",)
    for name in flags:
        monkeypatch.delenv(name, raising=False)
        if explicit:
            monkeypatch.setenv(name, "1")
    before = {name: os.environ.get(name) for name in flags}
    config = SimpleNamespace(dflash_config=SimpleNamespace(is_dspark=dspark))
    try:
        with _draft_export_environment(config):
            expected = "1" if explicit else "0" if dspark else None
            assert all(os.environ.get(name) == expected for name in flags)
            if raises:
                raise RuntimeError("export failure")
    except RuntimeError as error:
        assert raises and str(error) == "export failure"
    assert {name: os.environ.get(name) for name in flags} == before


def _checkpoint(tmp_path, *, moe=False):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    text = dict(
        hidden_size=2048 if moe else 1024, vocab_size=248144 if moe else 248094, num_hidden_layers=40 if moe else 24
    )
    ids = [3, 11, 19, 27, 35] if moe else [3, 7, 11, 15, 19]
    config = {
        **text,
        "architectures": ["Qwen3DSparkModel"],
        "num_hidden_layers": 5,
        "num_target_layers": text["num_hidden_layers"],
        "target_layer_ids": ids,
        "markov_head_type": "vanilla",
        "markov_rank": 256,
        "mask_token_id": 248076,
        "block_size": 15,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "draft_vocab_size": text["vocab_size"],
    }
    (target / "config.json").write_text(json.dumps({"text_config": text}))
    (draft / "config.json").write_text(json.dumps(config))
    return target, draft, config


@pytest.mark.parametrize("as_object", (False, True))
@pytest.mark.parametrize("draft_flash", (None, False, True))
def test_draft_attention_override_survives_target_binding_and_metadata(tmp_path, as_object, draft_flash):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.dspark_config import (
        MiniCPMV47DFlashConfig as XHQwen3_5_DFlashConfig,
        MiniCPMV47ModelConfig as XHQwen3_5ModelConfig,
    )

    target, draft, _ = _checkpoint(tmp_path, moe=True)
    settings = {"hf_model": str(draft)}
    if draft_flash is not None:
        settings["flash_attention"] = {"enable": draft_flash}
    config = XHQwen3_5ModelConfig(
        model_name="minicpm", hf_model=str(target), visual_config=None,
        spec_decode_mode="dflash", num_draft_tokens=3,
        flash_attention={"enable": True},
        dflash_config=XHQwen3_5_DFlashConfig(model_name="draft", **settings) if as_object else settings,
    )
    expected = True if draft_flash is None else draft_flash
    assert config.flash_attention["enable"] is True
    assert config.dflash_config.flash_attention["enable"] is expected
    restored = XHQwen3_5ModelConfig.from_dict(config.to_dict())
    assert restored.flash_attention["enable"] is True
    assert restored.dflash_config.flash_attention["enable"] is expected


@pytest.mark.parametrize("moe", (False, True))
def test_dspark_pairing_and_public_recipe(tmp_path, moe):
    from xhmodel_merak.model_integrations.minicpm_v_4_7.planning import resolve_config
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.dspark_config import MiniCPMV47ModelConfig as XHQwen3_5ModelConfig

    target, draft, raw = _checkpoint(tmp_path, moe=moe)
    normalized = validate_draft_target(draft, target)
    assert normalized["dflash_config"]["target_layer_ids"] == raw["target_layer_ids"]
    config = XHQwen3_5ModelConfig(
        model_name="minicpm",
        hf_model=str(target),
        visual_config=None,
        spec_decode_mode="dflash",
        num_draft_tokens=15,
        dflash_config={"hf_model": str(draft)},
    )
    assert config.dflash_config.num_target_layers == 5
    assert config.dflash_config.is_dspark
    assert config.dflash_config.noise_token_id == 248076
    assert config.dflash_config.block_size == 15
    with pytest.raises(ValueError, match="capacity"):
        XHQwen3_5ModelConfig(
            model_name="minicpm",
            hf_model=str(target),
            visual_config=None,
            spec_decode_mode="dflash",
            num_draft_tokens=16,
            dflash_config={"hf_model": str(draft)},
        )
    recipe = get_recipe("minicpm-v-4.7-35b-a3b" if moe else "minicpm-v-4.7-1b")
    variant = next(v for v in recipe.variants if v.name == "dflash")
    assert variant.required_sources == ("dflash",)
    assert variant.extra_components == ("dflash",)
    replacements = resolve_config(
        recipe=recipe,
        variant=variant,
        model_dir=target,
        sources={"dflash": SimpleNamespace(path=str(draft))},
        workflow={},
    )
    assert replacements["export.model.dflash_config.hf_model"] == str(draft)
    assert replacements["export.model.dflash_config.noise_token_id"] == 248076
    saved = config.to_dict()
    # Published metadata cannot depend on an assistant config remaining on disk.
    (draft / "config.json").unlink()
    restored = XHQwen3_5ModelConfig.from_dict(saved)
    assert restored.dflash_config.is_dspark
    assert restored.num_draft_tokens == 15


@pytest.mark.parametrize(
    "field,value",
    (
        ("vocab_size", 100),
        ("hidden_size", 100),
        ("num_target_layers", 12),
        ("target_layer_ids", [3, 24]),
        ("target_layer_ids", [7, 3]),
        ("markov_head_type", "gated"),
    ),
)
def test_incompatible_dspark_fails_before_export(tmp_path, field, value):
    target, draft, raw = _checkpoint(tmp_path)
    raw[field] = value
    (draft / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        validate_draft_target(draft, target)


def test_vanilla_markov_proposals_use_anchor_then_each_previous_proposal():
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.dspark_heads import (
        DSparkConfidenceHead,
        DSparkMarkovHead,
        greedy_dspark_proposals,
    )

    markov = DSparkMarkovHead(4, 4)
    confidence = DSparkConfidenceHead(6)
    with torch.no_grad():
        markov.markov_w1.weight.copy_(torch.eye(4))
        markov.markov_w2.weight.copy_(torch.roll(torch.eye(4), 1, dims=0) * 10)
        confidence.proj.weight.fill_(0.25)
        confidence.proj.bias.fill_(-0.5)
    hidden = torch.zeros(1, 3, 2)
    logits = torch.zeros(1, 3, 4)
    tokens, probabilities = greedy_dspark_proposals(logits, hidden, torch.tensor([0]), markov, confidence, True)
    assert tokens.tolist() == [[1, 2, 3]]  # includes output position zero
    torch.testing.assert_close(probabilities, torch.sigmoid(torch.full((1, 3), -0.25)))


@pytest.mark.parametrize("mode,grid,tokens", (("4x", 4, 4), ("16x", 8, 4)))
def test_canvas_positions_and_chunked_decode_match_native(mode, grid, tokens):
    from transformers.models.minicpmv4_7.configuration_minicpmv4_7 import MiniCPMV4_7Config
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.positions import canvas_positions, position_chunk

    config = MiniCPMV4_7Config(
        mrope_mode="canvas",
        image_token_id=31,
        video_token_id=32,
        image_start_id=33,
        image_end_id=34,
        slice_start_id=35,
        slice_end_id=36,
        newline_id=198,
    )
    ids = torch.tensor([[1, 33, *([31] * tokens), 34, 2, 3]])
    positions, delta = canvas_positions(
        config, {"input_ids": ids, "target_sizes": torch.tensor([[grid, grid]])}, downsample_mode=mode
    )
    assert positions.shape == (3, 1, ids.shape[-1])
    assert not torch.equal(positions[1], positions[2])
    reconstructed = torch.cat(
        [position_chunk(positions, delta, offset=i, length=4) for i in range(0, ids.shape[-1], 4)], dim=-1
    )
    torch.testing.assert_close(reconstructed[..., : ids.shape[-1]], positions)
    decode = position_chunk(positions, delta, offset=ids.shape[-1], length=3)
    expected = torch.arange(ids.shape[-1], ids.shape[-1] + 3).view(1, 1, -1).expand(3, 1, -1) + delta.unsqueeze(0)
    torch.testing.assert_close(decode, expected)


@pytest.mark.parametrize("rank", (1, 2))
def test_speculative_prefill_scatters_each_crop_once_and_verify_uses_canvas_delta(rank):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.spec_decode import MiniCPMV47SpecDecodeModel

    model = MiniCPMV47SpecDecodeModel.__new__(MiniCPMV47SpecDecodeModel)
    torch.nn.Module.__init__(model)
    model._device = torch.device("cpu")
    model._dtype = torch.float16
    model.token_embedding = torch.nn.Embedding(64, 4, dtype=torch.float16)
    model.pad_token_id = 0
    model.image_token_id = 31
    positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6).clone()
    positions[1] += 10
    positions[2] += 20
    model.prepared = SimpleNamespace(
        input_ids=torch.tensor([[1, 31, 31, 2, 31, 3]]),
        image_embeds=torch.arange(12).reshape(3, 4).half() + 100,
        position_ids=positions,
        rope_deltas=torch.tensor([[-2]]),
    )
    info = {
        "inputs_embeds": SimpleNamespace(shape=(1, 4, 4), dtype=torch.float16),
        "past_seq_length": SimpleNamespace(shape=(1,), dtype=torch.int32),
        "current_input_length": SimpleNamespace(shape=(1,), dtype=torch.int32),
        "linear_attn_mask": SimpleNamespace(shape=(1, 4), dtype=torch.float16),
    }
    for name in ("time_position_ids", "hight_position_ids", "width_position_ids"):
        info[name] = SimpleNamespace(shape=(4,) if rank == 1 else (1, 4), dtype=torch.int32)
    session = SimpleNamespace(get_input_names=lambda: list(info), get_input=lambda name: info[name])
    for phase in ("prefill", "decode"):
        setattr(model, f"{phase}_session", session)
        for field, name in (
            ("inputs", "inputs_embeds"),
            ("past_seq", "past_seq_length"),
            ("current_seq", "current_input_length"),
            ("mask", "linear_attn_mask"),
        ):
            setattr(model, f"_{phase}_{field}_name", name)
            setattr(model, f"_{phase}_{field}_info", info[name])
    first = model._build_prefill_feed(model.prepared.input_ids[:, :4], 4, 0, {})
    second = model._build_prefill_feed(model.prepared.input_ids[:, 4:], 2, 4, {})
    torch.testing.assert_close(first["inputs_embeds"][0, 1:3], model.prepared.image_embeds[:2])
    torch.testing.assert_close(second["inputs_embeds"][0, :1], model.prepared.image_embeds[2:])
    for axis, name in enumerate(("time_position_ids", "hight_position_ids", "width_position_ids")):
        torch.testing.assert_close(first[name].flatten(), positions[axis, 0, :4].int())
        torch.testing.assert_close(second[name].flatten(), torch.cat((positions[axis, 0, 4:], torch.zeros(2))).int())
    verify = model._build_decode_feed(torch.tensor([[4, 5, 6]]), 6, {}, 3)
    for name in ("time_position_ids", "hight_position_ids", "width_position_ids"):
        torch.testing.assert_close(verify[name].flatten(), torch.arange(4, 8, dtype=torch.int32))
    torch.testing.assert_close(verify["linear_attn_mask"], torch.tensor([[1, 1, 1, 0]]).half())


def _alignment_core(n=None, confidence=True, raw_scores=False, *, block_size=15):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.dspark_model import MiniCPMDSparkModel

    return MiniCPMDSparkModel(
        mode="decode", hidden_size=64, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, intermediate_size=128, num_hidden_layers=1, rms_norm_eps=1e-6,
        num_target_layers=1, target_layer_ids=[0], vocab_size=128,
        input_sequence_length=block_size, max_pe_length=64, max_sequence_length=32,
        rope_theta=10000000., architecture="Qwen3DSparkModel", block_size=block_size,
        dspark_config={"markov_rank": 8, "enable_confidence_head": True, "confidence_head_with_markov": True},
        num_output_draft_tokens=n, compute_confidence=confidence, confidence_logits=raw_scores,
    ).half().eval()


def _alignment_args(block_size=15):
    return [torch.randn(1, block_size, 64).half(), torch.tensor([2]), torch.tensor([block_size]),
            torch.tensor([4]), torch.zeros(1, 32).half(),
            *[torch.zeros(1, 1, 32, 32).half() for _ in range(2)]]


@pytest.mark.parametrize("n", [1, 3, 15])
def test_pruned_heads_preserve_full_query_block_proposals(n):
    from xhquant.core import CacheTensor
    torch.manual_seed(47)
    full = _alignment_core()
    trimmed = _alignment_core(n, raw_scores=True)
    trimmed.load_state_dict(full.state_dict(), strict=True)
    args = _alignment_args()
    seen = []
    handle = trimmed.lm_head.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].shape))
    with torch.no_grad():
        expected_ids, expected_confidence = full.forward_decode(
            args[0], args[1], args[2], args[4], args[3], *[CacheTensor(t.clone()) for t in args[-2:]])
        actual_ids, actual_scores = trimmed.forward_decode(
            args[0], args[1], args[2], args[4], args[3], *[CacheTensor(t.clone()) for t in args[-2:]])
    handle.remove()
    assert seen == [torch.Size([1, n, 64])]
    torch.testing.assert_close(actual_ids, expected_ids[:, :n], rtol=0, atol=0)
    torch.testing.assert_close(actual_scores.sigmoid(), expected_confidence[:, :n], rtol=0, atol=0)


@pytest.mark.parametrize(
    "block_size,proposals,confidence,raw_scores",
    [(1, None, True, False), (3, None, True, False), (15, None, True, False),
     (15, 3, False, True), (15, 3, True, True)],
)
def test_dspark_small_decode_exports_and_runs_hmonnx(
    tmp_path, block_size, proposals, confidence, raw_scores,
):
    from xhquant.api import (
        PrecisionMode, convert_quanted_model_to_hmonnx, ptq_quantize,
        to_frontend_graph, to_quant_graph,
    )
    from xhquant.core import CacheTensor
    from xhmodel_merak.exec_backends import HMONNXInferenceConfig, HMONNXInferenceV2, MultiHMONNXLoader
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.dspark_export import _build_dflash_export_adapter

    torch.set_num_threads(2)
    torch.manual_seed(47)
    core = _alignment_core(proposals, confidence, raw_scores, block_size=block_size)
    adapter = _build_dflash_export_adapter(core, mode="decode", num_hidden_layers=1, is_dspark=True)
    args = _alignment_args(block_size)
    graph = to_quant_graph(to_frontend_graph(adapter, "TorchFX", args), "XH2a", {"quant_type": "w8a8h1_sefp"})
    ptq_quantize(graph, [args], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    graph.fixed()
    with torch.no_grad():
        expected = graph(*args[:-2], *[CacheTensor(t.clone()) for t in args[-2:]])
    path = str(tmp_path / "dspark.onnx")
    convert_quanted_model_to_hmonnx(graph, args, path)
    loader = MultiHMONNXLoader({"draft": path})
    session = HMONNXInferenceV2.from_onnx_graph(
        path, loader.graphs["draft"], HMONNXInferenceConfig(exec_devices=["cpu"]),
    )
    values = [v.to(dtype=i.dtype) for v, i in zip(args, session.inputs, strict=True)]
    values[-2:] = [CacheTensor(t.clone()) for t in values[-2:]]
    actual = session(*values)
    output_length = block_size if proposals is None else proposals
    if confidence:
        assert actual[0].shape == actual[1].shape == (1, output_length)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], atol=.02, rtol=.02)
    else:
        actual = actual[0] if isinstance(actual, (tuple, list)) else actual
        assert actual.shape == (1, output_length)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _alignment_runtime_files(tmp_path, legacy_confidence=False):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def info(name, shape, dtype=TensorProto.FLOAT16):
        return helper.make_tensor_value_info(name, dtype, shape)

    def write(name, nodes, inputs, outputs, constants=()):
        path = tmp_path / f"{name}.onnx"
        for index, node in enumerate(nodes):
            node.name = f"{name}_{node.op_type}_{index}"
        graph = helper.make_graph(nodes, name, inputs, outputs, list(constants))
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        onnx.save(model, path)
        return {"onnx": str(path)}

    files = {}
    for length in range(1, 5):
        files[length] = write(f"target_{length}",
            [helper.make_node("Gather", ["inputs_embeds", "next_token"], ["logits"], axis=2),
             helper.make_node("Identity", ["inputs_embeds"], ["target_hidden"])],
            [info("inputs_embeds", [1, length, 4]), info("past_seq_length", [1], TensorProto.INT32),
             info("current_input_length", [1], TensorProto.INT32), info("linear_attn_mask", [1, length]),
             *[info(name, [length], TensorProto.INT32) for name in ("time_position_ids", "hight_position_ids", "width_position_ids")]],
            [info("logits", [1, length, 4]), info("target_hidden", [1, length, 4])],
            [numpy_helper.from_array(np.array([3, 0, 1, 2], dtype=np.int64), "next_token")])
    context = write("context", [helper.make_node("Identity", ["target_hidden"], ["context_out"])],
        [info("target_hidden", [1, 4, 4]), info("past_seq_length", [1], TensorProto.INT32),
         info("current_input_length", [1], TensorProto.INT32)], [info("context_out", [1, 4, 4])])
    probabilities = np.array([[.9, .8, .2]], dtype=np.float32)
    confidence_name = "confidence" if legacy_confidence else "confidence_logits"
    scores = probabilities if legacy_confidence else np.log(probabilities / (1 - probabilities))
    draft = write("draft", [helper.make_node("Identity", ["proposals"], ["draft_token_ids"]),
                            helper.make_node("Identity", ["scores"], [confidence_name])],
        [info("noise_embedding", [1, 15, 4]), info("anchor_token_id", [1], TensorProto.INT32),
         info("past_seq_length", [1], TensorProto.INT32), info("current_input_length", [1], TensorProto.INT32),
         info("attn_mask", [1, 32])],
        [info("draft_token_ids", [1, 3], TensorProto.INT64), info(confidence_name, [1, 3])],
        [numpy_helper.from_array(np.array([[2, 3, 0]], dtype=np.int64), "proposals"),
         numpy_helper.from_array(scores.astype(np.float16), "scores")])
    return files, context, draft


@pytest.mark.parametrize("threshold,length", [(0., 4), (.3, 3), (.95, 1)])
@pytest.mark.parametrize("legacy_confidence", [False, True])
def test_real_v2_confidence_routes_exact_verify_and_preserves_greedy(tmp_path, threshold, length, legacy_confidence):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.spec_decode import MiniCPMV47SpecDecodeModel
    files, context, draft = _alignment_runtime_files(tmp_path, legacy_confidence)
    runtime = MiniCPMV47SpecDecodeModel(
        prefill=files[4], decode=files[4], draft={"context": context, "context_decode": context, "decode": draft},
        session_device="cpu", auto_offload=False, spec_decode_mode="dflash", block_size=4,
        hidden_output_name="target_hidden", dflash_noise_token_id=0, max_context_tokens=32,
        enable_confidence_truncation=True, confidence_threshold=threshold, verify_graphs=files)
    runtime.set_input_embeddings(torch.nn.Embedding.from_pretrained(torch.eye(4)))
    runtime.to('cpu').set_exec_device('cpu')
    tokenizer = SimpleNamespace(eos_token_id=None, decode=lambda ids, **kw: ','.join(map(str, ids)))
    text, stats = runtime.generate(torch.tensor([[0]]), tokenizer=tokenizer, max_new_tokens=8, return_stats=True)
    assert text == '1,2,3,0,1,2,3,0'
    assert set(stats['verify_lengths_per_round']) == {length}
    assert not runtime._v2_graphs
    runtime.prepared = SimpleNamespace(position_ids=torch.arange(6).view(1, 1, 6).expand(3, 1, 6), rope_deltas=torch.tensor([[-2]]))
    state = runtime._verify_session_states[3]
    feed = runtime._build_verify_feed(state, torch.tensor([[1, 2, 3]]), 6, {})
    for name in ('time_position_ids', 'hight_position_ids', 'width_position_ids'):
        torch.testing.assert_close(feed[name], torch.tensor([4, 5, 6], dtype=feed[name].dtype))


def test_confidence_config_round_trip_retains_existing_interface(tmp_path):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.dspark_config import MiniCPMV47ModelConfig as XHQwen3_5ModelConfig
    target, draft, raw = _checkpoint(tmp_path)
    raw.update(enable_confidence_head=True, confidence_head_with_markov=True)
    (draft / 'config.json').write_text(json.dumps(raw))
    config = XHQwen3_5ModelConfig(model_name='mini', hf_model=str(target), visual_config=None,
        spec_decode_mode='dflash', num_draft_tokens=3,
        dflash_config={'hf_model': str(draft), 'enable_confidence_truncation': True, 'confidence_threshold': .3})
    saved = config.to_dict()
    (draft / 'config.json').unlink()
    restored = XHQwen3_5ModelConfig.from_dict(saved)
    assert restored.spec_decode_mode == 'dflash'
    assert restored.dflash_config.block_size == 15
    assert restored.dflash_config.enable_confidence_truncation
    assert restored.dflash_config.confidence_threshold == .3
    assert not hasattr(restored, 'dspark_config')


def test_real_hybrid_target_exports_and_runs_all_confidence_verify_lengths(tmp_path, monkeypatch):
    monkeypatch.setenv("XHQUANT_STRICT_EXPORT", "0")
    import json
    from pathlib import Path
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from xhmodel_merak.xh_llm.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from xhmodel_merak.xh_llm.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.dspark_config import MiniCPMV47ModelConfig as XHQwen3_5ModelConfig
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.dspark_model import MiniCPMDSparkModel as DFlashModel
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.text_model import MiniCPMV47TextModel
    root=tmp_path
    target=root/'target';draft=root/'draft';draft.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(47)
    hf_cfg=Qwen3_5Config(text_config=dict(vocab_size=128,hidden_size=128,intermediate_size=256,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=64,
        layer_types=['linear_attention','full_attention'],linear_num_key_heads=2,linear_num_value_heads=2,
        linear_key_head_dim=32,linear_value_head_dim=32,linear_conv_kernel_dim=4,
        rope_parameters=dict(rope_theta=10000000.,partial_rotary_factor=.25,mrope_interleaved=True,mrope_section=[3,3,2]),
        pad_token_id=0,eos_token_id=127,tie_word_embeddings=False),
        vision_config=dict(depth=1,hidden_size=64,intermediate_size=128,out_hidden_size=128,num_heads=2,
                           patch_size=14,temporal_patch_size=2,spatial_merge_size=2,num_position_embeddings=16))
    hf=Qwen3_5ForConditionalGeneration(hf_cfg).half().eval();hf.save_pretrained(target)
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[PAD]':0,'[UNK]':1,'test':2},unk_token='[UNK]')),pad_token='[PAD]',unk_token='[UNK]')
    tokenizer.save_pretrained(target)
    draft_cfg=dict(architectures=['Qwen3DSparkModel'],hidden_size=128,vocab_size=128,intermediate_size=256,
     num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=64,rms_norm_eps=1e-6,
     num_target_layers=2,target_layer_ids=[0,1],block_size=15,mask_token_id=126,
     markov_rank=8,markov_head_type='vanilla',attention_mode='gqa',enable_confidence_head=True,
     confidence_head_with_markov=True,rope_parameters={'rope_theta':10000000.})
    (draft/'config.json').write_text(json.dumps(draft_cfg))
    core=DFlashModel(mode='decode',hidden_size=128,num_attention_heads=2,num_key_value_heads=1,head_dim=64,
     intermediate_size=256,num_hidden_layers=1,rms_norm_eps=1e-6,num_target_layers=2,target_layer_ids=[0,1],
     vocab_size=128,input_sequence_length=15,max_pe_length=128,max_sequence_length=128,architecture='Qwen3DSparkModel',block_size=15,dspark_config=draft_cfg).half()
    save_file({k:v.contiguous() for k,v in core.state_dict().items() if k!='lm_head.weight'},draft/'model.safetensors')
    config=XHQwen3_5ModelConfig(model_name='tiny_minicpm',hf_model=str(target),visual_config=None,
     context_max_length=128,prefill_chunk_length=64,max_pe_length=128,spec_decode_mode='dflash',num_draft_tokens=3,
     flash_attention={'enable':False},fuse_gdr_ops=True,fuse_gdr_block_recurrent_ops=True,
     dflash_config={'hf_model':str(draft),'enable_confidence_truncation':True,'confidence_threshold':.3})
    embedding = torch.nn.Embedding.from_pretrained(hf.model.language_model.embed_tokens.weight.detach().clone())
    model=MiniCPMV47TextModel(config);model.to_wrap(hf);model.to('cpu')
    meta=model.export_hmonnx(str(root/'out'))
    data = meta.to_dict()
    contract = data['spec_decode']
    assert contract['mode'] == 'dflash'
    assert contract['draft']['block_size'] == 15
    assert contract['draft']['num_output_draft_tokens'] == 3
    assert contract['draft']['decode_outputs'] == ['draft_token_ids', 'confidence_logits']
    assert set(contract['verify_graphs']) == {'1','2','3','4'}
    artifact = next((root/'out').glob('hmquant_*'))
    assert json.loads((artifact / 'golden_meta_info.json').read_text())['spec_decode'] == contract
    import onnx
    for length, path in contract['verify_graphs'].items():
        graph = onnx.load(artifact/path, load_external_data=False)
        assert graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value == int(length)
        assert sum(n.op_type == 'Where' for n in graph.graph.node) == int(length)
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.spec_decode import MiniCPMV47SpecDecodeModel
    texts = []
    for threshold in (0., .99):
        runtime = MiniCPMV47SpecDecodeModel(
            prefill={'onnx':str(artifact/data['prefill_hmonnx'])},
            decode={'onnx':str(artifact/data['decode_hmonnx'])},
            draft={name:{'onnx':str(artifact/contract['draft'][name+'_hmonnx'])} for name in ('context','context_decode','decode')},
            session_device='cpu', enable_golden=(threshold == .99), auto_offload=False, spec_decode_mode='dflash', block_size=4,
            hidden_output_name='target_hidden', dflash_noise_token_id=126, max_context_tokens=128,
            enable_confidence_truncation=True, confidence_threshold=threshold,
            verify_graphs={length:{'onnx':str(artifact/path)} for length,path in contract['verify_graphs'].items()})
        runtime.set_input_embeddings(embedding)
        runtime.to('cpu').set_exec_device('cpu')
        from types import SimpleNamespace
        fake_tokenizer = SimpleNamespace(eos_token_id=None, decode=lambda ids, **kw: ','.join(map(str,ids)))
        text, stats = runtime.generate(torch.tensor([[2,3,4]]), fake_tokenizer, max_new_tokens=8, return_stats=True)
        texts.append(text)
        if threshold == .99:
            assert (artifact / contract['verify_graphs']['1']).parent.joinpath('step_0').is_dir()
        assert set(stats['verify_lengths_per_round']) == ({4} if threshold == 0. else {1})
    assert texts[0] == texts[1]
    for path in contract['verify_graphs'].values():
        assert (artifact / path).is_file()
    for name in ('context', 'context_decode', 'decode'):
        assert (artifact / contract['draft'][name + '_hmonnx']).is_file()
    draft_graph = onnx.load(artifact / contract['draft']['decode_hmonnx'], load_external_data=False)
    assert draft_graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 15


def test_dspark_export_preserves_qwen_classes_and_loader_on_success_and_failure():
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.dspark_model import (
        MiniCPMDSparkDraftExport, MiniCPMDSparkDraftModel,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5 import _dflash_model_impl, qwen3_5_dflash_model

    original = qwen3_5_dflash_model.XHQwen3_5DFlashDraftModel
    loader = _dflash_model_impl.DFlashModel.from_pretrained
    methods = {name: getattr(original, name) for name in ("__init__", "get_dummy_inputs", "get_export_cfg")}
    for fail in (False, True):
        with pytest.raises(RuntimeError, match="export failed") if fail else contextlib.nullcontext():
            with MiniCPMDSparkDraftExport(3):
                assert qwen3_5_dflash_model.XHQwen3_5DFlashDraftModel is MiniCPMDSparkDraftModel
                assert issubclass(MiniCPMDSparkDraftModel, original)
                assert _dflash_model_impl.DFlashModel.from_pretrained is loader
                assert all(getattr(original, name) is method for name, method in methods.items())
                if fail:
                    raise RuntimeError("export failed")
        assert qwen3_5_dflash_model.XHQwen3_5DFlashDraftModel is original


@pytest.mark.parametrize('options,match', [
    ({'enable_confidence_truncation': 1}, 'boolean'),
    ({'confidence_threshold': float('nan')}, 'confidence_threshold'),
    ({'confidence_threshold': -0.1}, 'confidence_threshold'),
    ({'confidence_threshold': 1.1}, 'confidence_threshold'),
    ({'enable_confidence_truncation': True}, 'checkpoint confidence head'),
])
def test_invalid_confidence_options_fail_before_export(tmp_path, options, match):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.dspark_config import MiniCPMV47ModelConfig as XHQwen3_5ModelConfig
    target, draft, _ = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match=match):
        XHQwen3_5ModelConfig(model_name='mini', hf_model=str(target), visual_config=None,
            spec_decode_mode='dflash', num_draft_tokens=3, dflash_config={'hf_model': str(draft), **options})


def test_verify_recovery_eos_is_emitted_and_only_anchor_state_is_committed():
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.spec_decode_loop import SpecDecodeVerifyResult, run_spec_decode_loop
    committed, emitted = [], []
    generated, stats = run_spec_decode_loop(
        max_new_tokens=5, initial_token_id=2, initial_past_seq_len=3,
        initial_last_hidden=None, initial_mtp_past_seq_len=0, eos_token_id=9, num_drafts=1,
        run_draft=lambda *args: [8],
        verify_round=lambda anchor, draft, past: SpecDecodeVerifyResult(past, [anchor]+draft, [9, 10], None, None),
        apply_verify_result=lambda result, steps: committed.append((result.initial_seq_len, steps)),
        post_verify=lambda *args: (None, 0), on_token=emitted.append)
    assert generated == emitted == [2, 9]
    assert committed == [(3, 1)]
    assert stats['accepted_drafts_total'] == 0
