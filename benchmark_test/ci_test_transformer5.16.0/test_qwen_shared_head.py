"""Target-owned heads across Qwen dense, MoE, Next and speculative graphs."""

import json
from pathlib import Path

import pytest
import torch

from xhmodel_merak.xh_llm.shared_head import (
    SharedHeadMixin,
    load_checkpoint_head,
    quantized_head,
    target_head_bits,
)


@pytest.mark.parametrize("family", ["dense", "moe", "next"])
@pytest.mark.parametrize("bits", [4, 5, 8])
def test_family_head_inheritance(family, bits):
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5ModelConfig
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig
    from xhmodel_merak.xh_llm.models.qwen3_next.xh_qwen3_next_config import XHQwen3NextModelConfig

    cls = {"dense": XHQwen3_5ModelConfig, "moe": XHQwen3_5MoeModelConfig, "next": XHQwen3NextModelConfig}[family]
    kwargs = dict(
        model_name="tiny",
        hf_model="unused",
        spec_decode_mode="mtp",
        mtp_config={},
        quant_scheme={"quant_type": "w8a8h1_sefp", "nodes": {"lm_head": {"quant_type": f"w{bits}a8h1_sefp"}}},
    )
    config = cls(**kwargs)
    assert config.mtp_config.draft_head_weight_bits == bits
    assert target_head_bits(config.mtp_config.quant_scheme) == bits
    from types import SimpleNamespace
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import build_qwen35_spec_decode_contract

    metadata = SimpleNamespace(
        mtp_prefill_config=SimpleNamespace(hmonnx="mtp_prefill.onnx"),
        mtp_decode_config=SimpleNamespace(hmonnx="mtp_decode.onnx"),
    )
    assert build_qwen35_spec_decode_contract(config, metadata)["shared_head"] == {
        "version": 1, "owner": "target", "logical_bits": bits
    }
    for override in (
        {"spec_draft_head_weight_bits": 5 if bits != 5 else 4},
        {"mtp_config": {"draft_head_weight_bits": 5 if bits != 5 else 4}},
        {
            "mtp_config": {
                "quant_scheme": {"nodes": {"lm_head": {"quant_type": "w5a8h1_sefp" if bits != 5 else "w4a8h1_sefp"}}}
            }
        },
    ):
        with pytest.raises(ValueError, match="conflicts"):
            cls(**{**kwargs, **override})




@pytest.mark.parametrize("bits", [4, 5, 8])
def test_fixed_head_export_and_independent_runtime(bits, tmp_path):
    from xhquant.api import (
        PrecisionMode,
        to_frontend_graph,
        to_quant_graph,
        ptq_quantize,
        convert_quanted_model_to_hmonnx,
    )
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceV2
    from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(128, 256, bias=False).half()
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.randn_like(self.lm_head.weight) * 0.02)

        def forward(self, x):
            return self.lm_head(x)

    torch.manual_seed(45)
    model = Head().eval()
    inputs = [torch.randn(1, 2, 128).half()]
    scheme = {"quant_type": f"w{bits}a8h1_sefp"}
    source = to_quant_graph(to_frontend_graph(model, "TorchFX", inputs), "XH2a", scheme)
    ptq_quantize(
        source, [inputs], PrecisionMode.ALIGNED, ["cpu"], auto_release_unused_parameters=True, infer_shape=False
    )
    source.fixed()
    head = quantized_head(source)
    # Integrated exports write the target before constructing draft graphs.
    convert_quanted_model_to_hmonnx(source, inputs, str(tmp_path / "target.onnx"))
    # Different float weights: re-quantizing a draft would produce different codes.
    draft = to_quant_graph(to_frontend_graph(Head().eval(), "TorchFX", inputs), "XH2a", scheme)
    from types import SimpleNamespace

    binding = SharedHeadMixin()
    binding.config = SimpleNamespace(draft_head_weight_bits=bits)
    binding.bind_target_head(head)
    assert binding._prepare_quantized_graph(draft) is draft
    binding.config.draft_head_weight_bits = 5 if bits != 5 else 4
    with pytest.raises(ValueError, match="precision conflicts"):
        binding.bind_target_head(head)
    ptq_quantize(
        draft, [inputs], PrecisionMode.ALIGNED, ["cpu"], auto_release_unused_parameters=True, infer_shape=False
    )
    assert draft.lm_head.qweight is head.qweight
    assert draft.lm_head.scale_or_exp is head.scale_or_exp
    torch.testing.assert_close(source(*inputs), draft(*inputs), rtol=0, atol=0)
    sessions = []
    for name, graph in (("target", source), ("draft", draft)):
        path = tmp_path / f"{name}.onnx"
        if name != "target":
            convert_quanted_model_to_hmonnx(graph, inputs, str(path))
        loader = MultiHMONNXLoader({name: str(path)})
        sessions.append(HMONNXInferenceV2.from_onnx_graph(str(path), loader.graphs[name]))
    # Each exported graph must execute correctly on its own, with no runtime
    # head discovery, binding or storage-address requirement.
    torch.testing.assert_close(sessions[0](*inputs), sessions[1](*inputs), rtol=0, atol=0)
    # Exercise the V2 entry with normal SDK session construction.
    import onnx
    from xhmodel_merak.xh_llm.models.qwen3_5.spec_decode_v2 import Qwen3_5SpecDecodeV2Model

    graph = onnx.load(tmp_path / "target.onnx")
    old_input, old_output = graph.graph.input[0].name, graph.graph.output[0].name
    graph.graph.input[0].name = "inputs_embeds"
    graph.graph.output[0].name = "logits"
    for node in graph.graph.node:
        for index, name in enumerate(node.input):
            if name == old_input:
                node.input[index] = "inputs_embeds"
        for index, name in enumerate(node.output):
            if name == old_output:
                node.output[index] = "logits"
    for name in ("past_seq_length", "current_input_length"):
        graph.graph.input.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.INT64, [1]))
    graph.graph.input.append(onnx.helper.make_tensor_value_info("linear_attn_mask", onnx.TensorProto.FLOAT16, [1, 2]))
    files = {}
    for name in ("prefill", "decode", "draft_prefill", "draft_decode"):
        path = tmp_path / f"runtime_{name}.onnx"
        onnx.save(graph, path)
        files[name] = {"onnx": str(path)}
    runtime = Qwen3_5SpecDecodeV2Model(
        prefill=files["prefill"],
        decode=files["decode"],
        draft={"prefill": files["draft_prefill"], "decode": files["draft_decode"]},
        spec_decode_mode="mtp",
        session_device="cpu",
        auto_offload=False,
    )
    feed = [inputs[0], torch.tensor([0]), torch.tensor([2]), torch.ones(1, 2).half()]
    outputs = [session(*feed) for session in (
        runtime.decode_session, runtime.prefill_session,
        runtime.draft_prefill_session, runtime.draft_decode_session,
    )]
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)


@pytest.mark.parametrize("tied", [True, False])
def test_embedding_fallback_requires_tie(tmp_path, tied):
    from safetensors.torch import save_file

    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 8, "vocab_size": 16, "tie_word_embeddings": tied}))
    weight = torch.randn(16, 8).half()
    save_file({"model.embed_tokens.weight": weight}, tmp_path / "model.safetensors")
    if tied:
        head, bits = load_checkpoint_head(tmp_path)
        assert bits is None and torch.equal(head.weight, weight)
    else:
        with pytest.raises(ValueError, match="Untied"):
            load_checkpoint_head(tmp_path)


@pytest.mark.parametrize("architecture", ["DFlashDraftModel", "DFlash2DraftModel"])
@pytest.mark.parametrize("bits", [4, 5, 8])
def test_dflash_head_inheritance(tmp_path, architecture, bits):
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5ModelConfig

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": [architecture],
                "hidden_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "num_hidden_layers": 1,
                "conv_kernel_size": 4,
                "conv_group_size": 32,
                "sliding_window": 128,
                "dflash_config": {
                    "mask_token_id": 0,
                    "target_layer_ids": [0],
                    "conv_kernel_size": 4,
                    "conv_group_size": 32,
                    "selector_rank": 16,
                    "selector_top_k": 32,
                },
            }
        )
    )
    config = XHQwen3_5ModelConfig(
        model_name="tiny",
        hf_model="unused",
        spec_decode_mode="dflash",
        quant_scheme={"nodes": {"lm_head": {"quant_type": f"w{bits}a8h1_sefp"}}},
        dflash_config={"hf_model": str(tmp_path)},
    )
    assert config.dflash_config.draft_head_weight_bits == bits
    assert target_head_bits(config.dflash_config.quant_scheme, "core_lm_head") == bits
    # Passing a previously materialized implicit draft must inherit as well.
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5_DFlashConfig

    standalone = XHQwen3_5_DFlashConfig(model_name="draft", hf_model=str(tmp_path))
    changed = XHQwen3_5ModelConfig(
        model_name="tiny",
        hf_model="unused",
        spec_decode_mode="dflash",
        quant_scheme={"nodes": {"lm_head": {"quant_type": "w5a8h1_sefp"}}},
        dflash_config=standalone,
    )
    assert changed.dflash_config.draft_head_weight_bits == 5
    assert target_head_bits(changed.dflash_config.quant_scheme, "core_lm_head") == 5


def test_standard_mtp_workflows():
    import yaml
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig
    from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import build_qwen35_autoround_kwargs
    from gptqmodel.quantization.qwen35_mtp import MTPQuantizationConfig

    root = Path(__file__).resolve().parents[2] / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b"
    for name, mixed in (
        ("qwen3_6_35b_a3b_full_head_w5_mtp.yaml", False),
        ("qwen3_6_35b_a3b_full_head_w5_auto_mixed_mtp.yaml", True),
    ):
        workflow = yaml.safe_load((root / name).read_text())
        config = XHQwen3_5MoeModelConfig(**workflow["export"]["model"])
        kwargs = build_qwen35_autoround_kwargs(
            hf_model_dir="unused",
            output_dir="unused_out",
            device="cpu",
            quant_cfg=workflow["quant"],
            export_model_cfg=workflow["export"]["model"],
            workflow_seed=42,
        )
        assert kwargs["head"]["bits"] == 5
        assert config.mtp_config.draft_head_weight_bits == workflow["quant"]["head"]["bits"] == 5
        assert ("auto_mixed_precision" in workflow["quant"]) is mixed
        if mixed:
            options = MTPQuantizationConfig(**workflow["quant"]["mtp"])
            assert options.expert_bits == 4 and options.algorithm == "inherit"


@pytest.mark.parametrize(
    "scheme,expected",
    [
        ({"nodes": {"lm_head": "w5a8h1_sefp"}}, 5),
        (
            {
                "quant_type": "w8a8h1_sefp",
                "nodes": {"lm_head": {"quant_type": "w4a8h1_sefp"}},
                "nodes_cfg": {"lm_head": {"quant_type": "w5a8h1_sefp"}},
            },
            5,
        ),
        ({"nodes_cfg": {"lm_head": {"w_schema": {"bits": 4}}}}, 4),
    ],
)
def test_head_resolution_matches_export(scheme, expected):
    from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import _head_recipe_kwargs

    assert target_head_bits(scheme) == expected
    assert _head_recipe_kwargs({"head": {"bits": expected}}, {"quant_scheme": scheme})["head"]["bits"] == expected




@pytest.mark.parametrize("family", ["dense", "moe", "next"])
def test_real_hybrid_target_quantizes_head_once(tmp_path, monkeypatch, family):
    """Keep real attention/GDN/experts and both prefill/verify conversion paths."""
    from collections import Counter
    from transformers import (
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeConfig,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3NextConfig,
        Qwen3NextForCausalLM,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5ModelConfig
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_model import XHQwen3_5MoeModel
    from xhmodel_merak.xh_llm.models.qwen3_next.xh_qwen3_next_config import XHQwen3NextModelConfig
    from xhmodel_merak.xh_llm.models.qwen3_next.qwen3_next_model import XHQwen3NextModel
    from xhquant.quantization.xh2a.qmodules._qlinear_base import QLinearBase

    torch.set_num_threads(2)
    config_cls, native_cls, merak_cfg_cls, merak_cls = {
        "dense": (Qwen3_5Config, Qwen3_5ForConditionalGeneration, XHQwen3_5ModelConfig, XHQwen3_5Model),
        "moe": (Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration, XHQwen3_5MoeModelConfig, XHQwen3_5MoeModel),
        "next": (Qwen3NextConfig, Qwen3NextForCausalLM, XHQwen3NextModelConfig, XHQwen3NextModel),
    }[family]
    text = dict(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    extra = {}
    if family == "next":
        config = config_cls(**text)
    else:
        text["rope_parameters"] = dict(
            rope_type="default", rope_theta=10000000.0, partial_rotary_factor=0.25, mrope_section=[3, 3, 2]
        )
        vision = dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=128,
            patch_size=16,
            spatial_merge_size=2,
        )
        config = config_cls(text_config=text, vision_config=vision)
        extra["visual_config"] = dict(
            model_name="disabled",
            enable=False,
            hf_model=str(tmp_path),
            visual_input_mode="patches",
            image_token_gears=[16],
            image_token_capacity=16,
            spatial_merge_size=2,
        )
    native = native_cls(config).half().eval()
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.02)
    native.save_pretrained(tmp_path)
    model_config = merak_cfg_cls(
        model_name="tiny_shared_head",
        hf_model=str(tmp_path),
        context_max_length=64,
        prefill_chunk_length=64,
        max_pe_length=128,
        spec_decode_mode="mtp",
        mtp_config={},
        output_post_norm_hidden=True,
        quant_scheme={"quant_type": "w8a8h1_sefp"},
        **extra,
    )
    calls = Counter()
    original = QLinearBase.weight_static_quant

    def counted(module, *args, **kwargs):
        calls[id(module)] += 1
        return original(module, *args, **kwargs)

    monkeypatch.setattr(QLinearBase, "weight_static_quant", counted)
    model = merak_cls(model_config)
    model.to_wrap(native)
    model.to_quanted_aligned()
    prefill = quantized_head(model._quanted_model.prefill)
    verify = quantized_head(model._quanted_model.decode)
    assert prefill is not verify
    assert calls[id(prefill)] == 1 and calls[id(verify)] == 0
    torch.testing.assert_close(prefill.qweight, verify.qweight, rtol=0, atol=0)
    torch.testing.assert_close(prefill.scale_or_exp, verify.scale_or_exp, rtol=0, atol=0)
    assert not hasattr(model, "_shared_target_head")


def test_next_cli_inherits_target_head():
    from examples_merak.llm.qwen3_next.qwen3_next_xh_export_hmonnx import build_config, build_parser

    parser = build_parser()
    base = ["--model", "unused", "--output-dir", "unused_out", "--mtp"]
    assert build_config(parser.parse_args(base)).mtp_config.draft_head_weight_bits == 8
    with pytest.raises(ValueError, match="conflicts"):
        build_config(parser.parse_args(base + ["--mtp-head-weight-bits", "4"]))
    assert (
        build_config(
            parser.parse_args(base + ["--quant-type", "w5a8h1_sefp", "--mtp-head-weight-bits", "5"])
        ).mtp_config.draft_head_weight_bits
        == 5
    )


@pytest.mark.parametrize("architecture", ["DFlashDraftModel", "DFlash2DraftModel"])
@pytest.mark.parametrize("flash_attention", [False, True])
def test_real_dflash_decode_reuses_w5_head(tmp_path, architecture, flash_attention):
    from xhmodel_merak.xh_llm.models.qwen3_5._dflash_model_impl import DFlashModel
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_dflash_model import _build_dflash_export_adapter
    from xhmodel_merak.xh_llm.shared_head import bind_quantized_head
    from xhquant.api import to_frontend_graph, to_quant_graph, ptq_quantize, PrecisionMode, convert_quanted_model_to_hmonnx
    torch.set_num_threads(2)
    core = DFlashModel(mode="decode", hidden_size=128, num_attention_heads=4, num_key_value_heads=2,
                       head_dim=32, intermediate_size=256, num_hidden_layers=1, rms_norm_eps=1e-6,
                       num_target_layers=1, target_layer_ids=[0], vocab_size=256, input_sequence_length=4,
                       max_pe_length=32, max_sequence_length=32, architecture=architecture, block_size=4,
                       conv_kernel_size=4, conv_group_size=32, selector_rank=16, selector_top_k=32,
                       sliding_window=32, flash_attention={"enable": flash_attention}).half().eval()
    with torch.no_grad():
        for name, value in core.named_parameters():
            value.copy_(torch.ones_like(value) if "norm" in name else torch.randn_like(value) * 0.02)
    core.prepare_for_export()
    is_v2 = architecture == "DFlash2DraftModel"
    adapter = _build_dflash_export_adapter(core, mode="decode", num_hidden_layers=1,
                                          use_flash_attention=flash_attention, is_dflash2=is_v2)
    assert core.layers[0].self_attn.use_flash_attention is (flash_attention and not is_v2)
    args = [torch.randn(1, 4, 128).half(), torch.tensor([0]), torch.tensor([4])]
    if is_v2:
        args.append(torch.zeros(1, dtype=torch.int64))  # compact cache position
        args.append(torch.zeros(1, dtype=torch.int64))
    if is_v2 or not flash_attention:
        args.append(torch.zeros((1, 4, 32) if is_v2 else (1, 32)).half())
    args.extend([torch.zeros(1, 2, 32, 32).half(), torch.zeros(1, 2, 32, 32).half()])
    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(128, 256, bias=False).half()
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.randn_like(self.lm_head.weight) * 0.02)
        def forward(self, hidden):
            return self.lm_head(hidden)
    head_inputs = [args[0]]
    target = to_quant_graph(to_frontend_graph(Target(), "TorchFX", head_inputs), "XH2a", {"quant_type": "w5a8h1_sefp"})
    ptq_quantize(target, [head_inputs], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    target.fixed()
    head = quantized_head(target)
    graph = to_quant_graph(to_frontend_graph(adapter, "TorchFX", args), "XH2a", {"quant_type": "w8a8h1_sefp"})
    bind_quantized_head(graph, head)
    ptq_quantize(graph, [args], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    draft_head = quantized_head(graph)
    assert draft_head.qweight is head.qweight
    assert draft_head.scale_or_exp is head.scale_or_exp
    assert draft_head.w_cfg.qspec.man_bit == 5
    graph.fixed()
    convert_quanted_model_to_hmonnx(graph, args, str(tmp_path / "draft.onnx"))
    assert (tmp_path / "draft.onnx").is_file()
    convert_quanted_model_to_hmonnx(target, head_inputs, str(tmp_path / "target.onnx"))
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceV2, HMONNXInferenceConfig
    from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader
    paths = {name: str(tmp_path / f"{name}.onnx") for name in ("target", "draft")}
    loader = MultiHMONNXLoader(paths)
    sessions = {name: HMONNXInferenceV2.from_onnx_graph(path, loader.graphs[name],
                HMONNXInferenceConfig(exec_devices=["cpu"])) for name, path in paths.items()}
    runtime = sessions["draft"]
    # Export normalizes integer control inputs to the serving INT32 ABI.
    runtime_args = [value.to(dtype=info.dtype) for value, info in zip(args, runtime.inputs, strict=True)]
    from xhquant.core.cache_tensor import CacheTensor
    def runtime_inputs():
        return [CacheTensor(value.clone()) if "cache" in info.name else value.clone()
                for value, info in zip(runtime_args, runtime.inputs, strict=True)]
    output = runtime(*runtime_inputs())
    for value in output if isinstance(output, (tuple, list)) else [output]:
        assert torch.isfinite(value).all()
