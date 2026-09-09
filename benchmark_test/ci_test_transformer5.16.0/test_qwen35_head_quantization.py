"""Qwen head recipe and real local W4/W5 packed-consumption contracts."""
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b"


@pytest.mark.parametrize("export_bits,quant_bits,accepted", [
    (8, None, True), (8, 8, True), (4, 4, True), (5, 5, True),
    (4, None, False), (5, 4, False), (8, 4, False), (4, 8, False),
])
def test_head_config_agreement(export_bits, quant_bits, accepted):
    from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import _head_recipe_kwargs

    quant = {} if quant_bits is None else {"head": {"bits": quant_bits}}
    export = {"quant_scheme": {"quant_type": "w8a8h1_sefp",
                               "nodes": {"lm_head": {"quant_type": f"w{export_bits}a8h1_sefp"}}}}
    if not accepted:
        with pytest.raises(ValueError):
            _head_recipe_kwargs(quant, export)
    else:
        assert bool(_head_recipe_kwargs(quant, export)) == (export_bits < 8)


def test_legacy_w8_skip_does_not_require_checkpoint_or_calibration(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow

    workflow = Qwen35Workflow(str(tmp_path / "unused"), str(CONFIG_DIR / "qwen3_6_35b_a3b_full.yaml"))
    result = workflow.quant(str(tmp_path / "unused-out"), "cpu", {"quant": None})
    assert result.skipped
    assert result.quanted_model_dir is None
    assert not (tmp_path / "unused-out").exists()


@pytest.mark.parametrize("bits", [4, 5])
def test_existing_skip_flag_validates_and_reuses_packed_head(bits, tmp_path):
    import json
    from safetensors.torch import save_file
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow

    source = tmp_path / "checkpoint"
    source.mkdir()
    packed = TorchQuantLinear(bits=bits, group_size=64, sym=True, desc_act=False,
                               in_features=128, out_features=64, bias=False)
    linear = torch.nn.Linear(128, 64, bias=False)
    linear.weight.data.copy_((torch.arange(8192).reshape(64, 128) % (2**bits) - 2**(bits-1)).float())
    packed.pack_original(linear, torch.ones(64, 2), torch.full((64, 2), float(2**(bits-1))),
                         torch.arange(128, dtype=torch.int32) // 64)
    save_file({"lm_head." + key: tensor for key, tensor in packed.state_dict().items()},
              source / "model.safetensors")
    (source / "config.json").write_text(json.dumps({"hidden_size": 128, "vocab_size": 64,
        "quantization_config": {"quant_method": "gptq", "provider": "auto-round",
            "checkpoint_format": "gptq_v2", "bits": 4, "group_size": 64, "sym": True,
            "desc_act": False, "lm_head": True, "dynamic": {"+:^lm_head$": {"bits": bits}}}}))
    config = yaml.safe_load((CONFIG_DIR / "qwen3_6_35b_a3b_full.yaml").read_text())
    config["quant"]["head"] = {"bits": bits}
    config["quant"]["calibration"]["jsonl"] = str(tmp_path / "absent.jsonl")
    config["export"]["model"]["quant_scheme"]["nodes"]["lm_head"]["quant_type"] = f"w{bits}a8h1_sefp"
    config_file = tmp_path / "workflow.yaml"
    config_file.write_text(yaml.safe_dump(config))
    workflow = Qwen35Workflow(str(source), str(config_file))
    # Real GPTQModel head inspection; no body load, calibration or checkpoint copy.
    result = workflow.quant(str(tmp_path / "unused-out"), "cpu", {"quant": None})
    assert workflow._resolve_export_model_dir(result) == str(source)
    assert not (tmp_path / "unused-out").exists()
    assert not source.with_name(source.name + f"_head_w{bits}").exists()
    # Export revalidates independently, including per-call overrides.
    other = 5 if bits == 4 else 4
    with pytest.raises(ValueError, match="conflicts"):
        workflow.export(result, str(tmp_path / "export"), "cpu", {
            "export.model.quant_scheme.nodes.lm_head.quant_type": f"w{other}a8h1_sefp"})
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("algorithm", ["autoround", "gptqmodel"])
def test_head_skip_result_keeps_the_output_checkpoint(algorithm, tmp_path, monkeypatch):
    from gptqmodel.recipes.qwen35_head import Qwen35HeadRecipeResult
    from xhmodel_merak.xh_llm.models.qwen3_5 import quant_adapter

    config = yaml.safe_load((CONFIG_DIR / "qwen3_6_35b_a3b_full.yaml").read_text())
    config["quant"]["head"] = {"bits": 5}
    config["export"]["model"]["quant_scheme"]["nodes"]["lm_head"]["quant_type"] = "w5a8h1_sefp"
    output = tmp_path / "copied-checkpoint"
    provenance = {"action": "skipped", "operation": {"actual": {"bits": 5, "algorithm": "autoround"}}}
    # Stub only the expensive backend. GPTQModel's real-model tests exercise
    # the skip decision and byte-preserving save; this covers the Merak ABI.
    monkeypatch.setattr(quant_adapter, "_load_callable", lambda *args: lambda **kwargs:
        Qwen35HeadRecipeResult(str(output), "skip", "gptqmodel_hf", provenance))
    run = getattr(quant_adapter, f"quantize_with_{algorithm}_api")
    result = run(model_dir=str(tmp_path / "source"), output_dir=str(output), device="cpu",
                 quant_cfg=config["quant"], export_model_cfg=config["export"]["model"], workflow_seed=42)
    assert result.quanted_model_dir == str(output)
    assert result.meta == provenance
    # QuantResult.skipped would make export select the original input path.
    assert result.skipped is False


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("algorithm", ["gptq", "autoround"])
def test_head_yaml_to_recipe(bits, algorithm, tmp_path):
    from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import (
        build_qwen35_gptqmodel_kwargs, build_qwen35_autoround_kwargs)
    from gptqmodel.recipes.qwen35 import quantize_qwen35
    from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround

    config = yaml.safe_load((CONFIG_DIR / f"qwen3_6_35b_a3b_full_head_w{bits}.yaml").read_text())
    config["quant"]["method"] = algorithm
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen3_5_moe"}')
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text('{"text":"calibration input"}\n')
    config["quant"]["calibration"]["jsonl"] = str(dataset)
    builder = build_qwen35_gptqmodel_kwargs if algorithm == "gptq" else build_qwen35_autoround_kwargs
    recipe = quantize_qwen35 if algorithm == "gptq" else quantize_qwen35_autoround
    kwargs = builder(hf_model_dir=str(source), output_dir=str(tmp_path / "out"),
        device="cpu", quant_cfg=config["quant"], export_model_cfg=config["export"]["model"], workflow_seed=42)
    result = recipe(**kwargs, dry_run=True)
    assert "scope" not in kwargs
    assert result.provenance["head_stage"]["head"]["algorithm"] == algorithm
    assert result.provenance["head_stage"]["head"]["bits"] == bits
    assert not (tmp_path / "out").exists()
    config["export"]["model"]["quant_scheme"]["nodes"]["lm_head"]["quant_type"] = "w8a8h1_sefp"
    with pytest.raises(ValueError, match="conflicts"):
        builder(hf_model_dir=str(source), output_dir=str(tmp_path / "out"),
            device="cpu", quant_cfg=config["quant"], export_model_cfg=config["export"]["model"], workflow_seed=42)


@pytest.mark.parametrize("bits", [4, 5, 8])
@pytest.mark.parametrize("weight_mode", ["packed", "dequantize"])
def test_packed_head_preserves_codes_and_logical_width(bits, weight_mode, tmp_path):
    import onnx
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from xhmodel_merak.xh_llm._dequant_converter import (
        gptqmodel_torch_qlinear_converter, gptqmodel_torch_qlinear_packed_converter)
    from xhquant.api import (PrecisionMode, convert_quanted_model_to_hmonnx,
                             ptq_quantize, to_frontend_graph, to_quant_graph)

    generator = torch.Generator().manual_seed(27)
    signed = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (64, 128), generator=generator)
    scales = torch.full((64, 2), 0.03125)
    linear = torch.nn.Linear(128, 64, bias=False)
    linear.weight.data.copy_(signed.float() * scales.repeat_interleave(64, dim=1))
    packed = TorchQuantLinear(bits=bits, group_size=64, sym=True, desc_act=False,
                               in_features=128, out_features=64, bias=False)
    packed.pack_original(linear, scales, torch.full_like(scales, 1 << (bits - 1)),
                         torch.arange(128, dtype=torch.int32) // 64)
    packed.qzero_format(format=2)
    if weight_mode == "packed":
        wrapped = gptqmodel_torch_qlinear_packed_converter(packed)
        assert wrapped.packed_weight.bits == (8 if bits == 5 else bits)
        assert wrapped.packed_weight.logical_bits == bits
        restored = wrapped.packed_weight.dequantize(dtype=torch.float32)
    else:
        # The normal Qwen exporter uses the existing dense + integer-code path.
        gptqmodel_torch_qlinear_converter(packed)
        wrapped = packed
        restored = wrapped.weight.float()
    torch.testing.assert_close(restored, linear.weight, rtol=0, atol=0)
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = wrapped

        def forward(self, hidden):
            return self.lm_head(hidden)

    inputs = (torch.ones(1, 2, 128, dtype=torch.float16),)
    frontend = to_frontend_graph(Head().eval(), "TorchFX", inputs)
    quantized_model = to_quant_graph(frontend, "XH2a", {"quant_type": f"w{bits}a8h1_sefp"})
    ptq_quantize(quantized_model, [list(inputs)], PrecisionMode.ALIGNED, ["cpu"],
                 auto_release_unused_parameters=True, infer_shape=False)
    quantized = quantized_model.lm_head
    assert quantized.w_cfg.qspec.man_bit == bits
    torch.testing.assert_close(quantized.qweight.reshape(128, 64).T, signed.to(torch.int8), rtol=0, atol=0)
    output = tmp_path / "head.onnx"
    convert_quanted_model_to_hmonnx(quantized_model, list(inputs), str(output))
    head_node = next(node for node in onnx.load(output).graph.node if node.op_type == "Linear")
    attributes = {a.name: onnx.helper.get_attribute_value(a) for a in head_node.attribute}
    assert attributes["hmfp_weight_man_bit"] == bits
