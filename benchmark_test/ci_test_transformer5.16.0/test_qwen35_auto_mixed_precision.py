"""YAML -> real recipes and producer plan -> real Merak precision schema."""

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_head_w4_auto_mixed.yaml"
)


@pytest.mark.parametrize("backend", ["autoround", "gptqmodel"])
@pytest.mark.parametrize("head_bits,expected_kl", [(4, .03), (5, .03)])
def test_yaml_to_recipe_search_flag(backend, head_bits, expected_kl, tmp_path):
    from gptqmodel.recipes.qwen35 import quantize_qwen35
    from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround

    from xhmodel_merak.xh_llm.models.qwen3_5 import quant_adapter

    config_path = CONFIG.with_name(CONFIG.name.replace("head_w4", f"head_w{head_bits}"))
    config = yaml.safe_load(config_path.read_text())
    quant = config["quant"]
    assert quant["head"]["bits"] == head_bits
    assert config["export"]["model"]["quant_scheme"]["nodes"]["lm_head"]["quant_type"] == f"w{head_bits}a8h1_sefp"
    quant["method"] = "gptq" if backend == "gptqmodel" else "autoround"
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5_moe"}')
    builder = getattr(quant_adapter, f"build_qwen35_{backend}_kwargs")
    kwargs = builder(
        hf_model_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        device="cuda:0",
        quant_cfg=quant,
        export_model_cfg=config["export"]["model"],
        workflow_seed=42,
    )
    assert kwargs["auto_mixed_precision"]["candidates"] == [4, 5, 8]
    image_dataset = ROOT / "data/calib_data/realworldqa_search_64.jsonl"
    assert kwargs["auto_mixed_precision"]["image_dataset"] == str(image_dataset.resolve())
    assert quant["auto_mixed_precision"]["image_dataset"].startswith("xh2modelzoo://")
    assert kwargs["expert_bits"] == kwargs["expert_down_bits"] == 4
    recipe = quantize_qwen35 if backend == "gptqmodel" else quantize_qwen35_autoround
    result = recipe(**kwargs, dry_run=True)
    assert result.output_dir == str(tmp_path / "out")
    head_stage = result.provenance["head_stage"]
    assert head_stage["head"]["bits"] == head_bits
    assert head_stage["head"]["algorithm"] == quant["method"]
    assert head_stage["calibration_jsonl"] == str((ROOT / "data/calib_data/NeelNanda-pile-10k.jsonl").resolve())
    command = result.provenance["command"]
    assert json.loads(command[command.index("--auto-mixed-precision") + 1])["max_kl"] == expected_kl
    search_config = json.loads(command[command.index("--auto-mixed-precision") + 1])
    assert search_config["image_dataset"] == str(image_dataset.resolve())
    del quant["auto_mixed_precision"]
    old = builder(
        hf_model_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        device="cuda:0",
        quant_cfg=quant,
        export_model_cfg=config["export"]["model"],
        workflow_seed=42,
    )
    assert "auto_mixed_precision" not in old


def _packed_checkpoint(root, attention_bits=5, qkv_output_size=64):
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from gptqmodel.quantization.auto_mixed_precision import dynamic_overrides, save_search_report
    from safetensors.torch import save_file

    groups = {
        "0:attention": (
            "model.language_model.layers.0.linear_attn", ["in_proj_qkv", "in_proj_z", "out_proj"], attention_bits,
        ),
        "0:shared_expert": (
            "model.language_model.layers.0.mlp.shared_expert",
            ["gate_proj", "up_proj", "down_proj"],
            4,
        ),
        "1:attention": ("model.language_model.layers.1.self_attn", ["q_proj", "k_proj", "v_proj", "o_proj"], 8),
        "1:shared_expert": (
            "model.language_model.layers.1.mlp.shared_expert",
            ["gate_proj", "up_proj", "down_proj"],
            5,
        ),
    }
    tensors, module_bits = {}, {}
    for prefix, projections, bits in groups.values():
        for projection in projections:
            name = f"{prefix}.{projection}"
            output_size = qkv_output_size if projection == "in_proj_qkv" else 64
            linear = torch.nn.Linear(64, output_size, bias=False)
            linear.weight.data.copy_(
                (torch.arange(output_size * 64).reshape(output_size, 64) % (2**bits) - 2 ** (bits - 1)).float()
            )
            packed = TorchQuantLinear(
                bits=bits, group_size=64, sym=True, desc_act=False, in_features=64, out_features=output_size, bias=False
            )
            packed.pack_original(
                linear,
                torch.ones(output_size, 1),
                torch.full((output_size, 1), float(2 ** (bits - 1))),
                torch.zeros(64, dtype=torch.int32),
            )
            packed.qzero_format(format=2)
            tensors.update({f"{name}.{k}": t for k, t in packed.state_dict().items()})
            module_bits[name] = bits
    root.mkdir()
    save_file(tensors, root / "model.safetensors")
    report = {
        "version": 1,
        "status": "accepted",
        "group_size": 64,
        "module_bits": module_bits,
        "unit_bits": {u: g[2] for u, g in groups.items()},
    }
    config = {
        "text_config": {"num_hidden_layers": 2},
        "quantization_config": {
            "bits": 4,
            "group_size": 64,
            "quant_method": "gptq",
            "checkpoint_format": "gptq_v2",
            "dynamic": dynamic_overrides(report),
        },
    }
    (root / "config.json").write_text(json.dumps(config))
    save_search_report(root, report)
    return report


def test_checkpoint_plan_sets_real_graph_weight_precision_and_preserves_head(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen3_5.mixed_precision import apply_checkpoint_precision
    from xhquant.quantization.xh2a._xh2a_quant_schema import XH2aQuantSchema

    root = tmp_path / "checkpoint"
    report = _packed_checkpoint(root)
    config = yaml.safe_load(CONFIG.read_text())["export"]
    original = copy.deepcopy(config)
    resolved = apply_checkpoint_precision(config, root, required=True)
    assert config == original
    scheme = resolved["model"]["quant_scheme"]
    schema = XH2aQuantSchema.from_schema(scheme)
    for name, bits in report["module_bits"].items():
        for node in (name, name.replace(".", "_")):
            q = schema.get_quant_config(node, torch.nn.Linear)
            assert q.w_cfg.quantizer.qspec.man_bit == bits
    assert schema.get_quant_config("lm_head", torch.nn.Linear).w_cfg.quantizer.qspec.man_bit == 4
    assert schema.get_quant_config("unrelated", torch.nn.Linear).w_cfg.quantizer.qspec.man_bit == 8
    key = next(iter(report["module_bits"]))
    config["model"]["quant_scheme"]["nodes"][key] = {"quant_type": "w4a8h1_sefp"}
    with pytest.raises(ValueError, match="conflicts"):
        apply_checkpoint_precision(config, root)
    # A logical W5 label on a W8-configured payload must fail, even with valid shapes.
    raw = json.loads((root / "config.json").read_text())
    raw["quantization_config"]["group_size"] = 128
    (root / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="group size"):
        apply_checkpoint_precision(original, root)
    raw["quantization_config"]["group_size"] = 64
    raw["quantization_config"]["dynamic"] = {}
    (root / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="conflicts"):
        apply_checkpoint_precision(original, root)


@pytest.mark.parametrize("context,flash", [(8192, False), (262144, True)])
def test_export_persists_generated_nodes_without_relaxing_user_overrides(tmp_path, monkeypatch, context, flash):
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    root = tmp_path / "checkpoint"
    report = _packed_checkpoint(root)
    config = yaml.safe_load(CONFIG.read_text())
    # This fixture exercises the export/config boundary; low-bit head payload
    # validation is covered separately by the head checkpoint tests.
    config["quant"]["head"]["bits"] = 8
    config["export"]["model"]["quant_scheme"]["nodes"]["lm_head"]["quant_type"] = "w8a8h1_sefp"
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(yaml.safe_dump(config))
    workflow = Qwen35Workflow(str(root), str(config_path))
    original = workflow.workflow_config
    before = copy.deepcopy(original.data)
    result = QuantResult(raw_model_dir=str(root), skipped=True)

    class BeforeModelLoadError(RuntimeError):
        pass

    def stop_before_model_load(self, quant_result, config_overrides, work_dir):
        resolved = self._build_export_config(
            quant_result, self.workflow_config.with_overrides(config_overrides))
        assert resolved["model"]["context_max_length"] == context
        assert resolved["model"]["flash_attention"]["enable"] is flash
        for name, bits in report["module_bits"].items():
            assert resolved["model"]["quant_scheme"]["nodes"][name]["quant_type"] == f"w{bits}a8h1_sefp"
        raise BeforeModelLoadError

    # Leave the real public export and base workflow config/dump path intact;
    # stop only where a full pretrained model would otherwise be loaded.
    monkeypatch.setattr(Qwen35Workflow, "get_model", stop_before_model_load)
    output = tmp_path / "export"
    with pytest.raises(BeforeModelLoadError):
        workflow.export(result, str(output), "cpu", {
            "export.model.context_max_length": context,
            "export.model.flash_attention.enable": flash,
        })
    assert workflow.workflow_config is original
    assert original.data == before
    dumped = yaml.safe_load((output / "workflow_override.yaml").read_text())
    model = dumped["export"]["model"]
    assert model["context_max_length"] == context
    assert model["flash_attention"]["enable"] is flash
    nodes = model["quant_scheme"]["nodes"]
    assert nodes["lm_head"]["quant_type"] == "w8a8h1_sefp"
    for name, bits in report["module_bits"].items():
        for alias in (name, name.replace(".", "_")):
            assert nodes[alias]["quant_type"] == f"w{bits}a8h1_sefp"
    with pytest.raises(KeyError, match="does not exist"):
        workflow.export(result, str(tmp_path / "invalid"), "cpu", {
            "export.model.quant_scheme.nodes": {"unknown_node": {"quant_type": "w4a8h1_sefp"}},
        })
    assert not (tmp_path / "invalid").exists()


def test_checkpoint_plan_reaches_hmonnx_projection_attributes(tmp_path):
    import onnx
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from safetensors.torch import load_file

    from xhmodel_merak.xh_llm._dequant_converter import gptqmodel_torch_qlinear_converter
    from xhmodel_merak.xh_llm.models.qwen3_5.mixed_precision import apply_checkpoint_precision
    from xhquant.api import (
        PrecisionMode,
        convert_quanted_model_to_hmonnx,
        ptq_quantize,
        to_frontend_graph,
        to_quant_graph,
    )

    root = tmp_path / "checkpoint"
    report = _packed_checkpoint(root)
    tensors = load_file(root / "model.safetensors")
    names = [next(n for n, b in report["module_bits"].items() if b == bits) for bits in (4, 5, 8)]

    class Projections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in names:
                module = TorchQuantLinear(
                    bits=report["module_bits"][name],
                    group_size=64,
                    sym=True,
                    desc_act=False,
                    in_features=64,
                    out_features=64,
                    bias=False,
                )
                module.load_state_dict({k[len(name) + 1 :]: v for k, v in tensors.items() if k.startswith(name + ".")})
                module.qzero_format(format=2)
                gptqmodel_torch_qlinear_converter(module)
                parent = self
                parts = name.split(".")
                for part in parts[:-1]:
                    if not hasattr(parent, part):
                        parent.add_module(part, torch.nn.Module())
                    parent = getattr(parent, part)
                parent.add_module(parts[-1], module)

        def forward(self, x):
            return tuple(self.get_submodule(name)(x) for name in names)

    export_cfg = apply_checkpoint_precision(yaml.safe_load(CONFIG.read_text())["export"], root)
    inputs = (torch.ones(1, 2, 64, dtype=torch.float16),)
    frontend = to_frontend_graph(Projections().eval(), "TorchFX", inputs)
    quantized = to_quant_graph(frontend, "XH2a", export_cfg["model"]["quant_scheme"])
    ptq_quantize(
        quantized,
        [list(inputs)],
        PrecisionMode.ALIGNED,
        ["cpu"],
        auto_release_unused_parameters=True,
        infer_shape=False,
    )
    for name in names:
        bits = report["module_bits"][name]
        module = quantized.get_submodule(name)
        signed = (torch.arange(4096).reshape(64, 64) % (2**bits) - 2 ** (bits - 1)).to(torch.int8)
        torch.testing.assert_close(module.qweight.reshape(64, 64).T, signed, rtol=0, atol=0)
    output = tmp_path / "mixed_projections.onnx"
    convert_quanted_model_to_hmonnx(quantized, list(inputs), str(output))
    nodes = [n for n in onnx.load(output).graph.node if n.op_type == "Linear"]
    assert len(nodes) == 3
    actual = {
        n.name: {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}["hmfp_weight_man_bit"] for n in nodes
    }
    assert sorted(actual.values()) == [4, 5, 8]


@pytest.mark.parametrize("bits", [4, 5, 8])
def test_split_qkv_preserves_calibrated_weights_and_hmonnx_precision(tmp_path, bits):
    import onnx
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from safetensors.torch import load_file
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedDeltaNet,
        Qwen3_5MoeTextConfig,
    )

    from xhmodel_merak.xh_llm._dequant_converter import gptqmodel_torch_qlinear_converter
    from xhmodel_merak.xh_llm.models.qwen3_5.mixed_precision import apply_checkpoint_precision
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model  # noqa: F401 -- register real wrappers
    from xhmodel_merak.xh_llm.wrap_model import wrap_llm_model
    from xhquant.api import (
        ConfigDict,
        PrecisionMode,
        convert_quanted_model_to_hmonnx,
        ptq_quantize,
        to_frontend_graph,
        to_quant_graph,
    )

    root = tmp_path / "checkpoint"
    _packed_checkpoint(root, attention_bits=bits, qkv_output_size=256)
    tensors = load_file(root / "model.safetensors")
    prefix = "model.language_model.layers.0.linear_attn"
    name = prefix + ".in_proj_qkv"
    packed = TorchQuantLinear(
        bits=bits, group_size=64, sym=True, desc_act=False, in_features=64, out_features=256, bias=False,
    )
    packed.load_state_dict({k[len(name) + 1:]: v for k, v in tensors.items() if k.startswith(name + ".")})
    packed.qzero_format(format=2)
    gptqmodel_torch_qlinear_converter(packed)
    signed = packed.quant_weight.detach().clone()
    weights = packed.weight.detach().clone()
    native = Qwen3_5MoeGatedDeltaNet(Qwen3_5MoeTextConfig(
        hidden_size=64, linear_num_key_heads=1, linear_key_head_dim=64,
        linear_num_value_heads=2, linear_value_head_dim=64,
    ), layer_idx=0).half()
    native.in_proj_qkv = packed
    wrapped = wrap_llm_model(native, ConfigDict(
        use_cache=False, split_conv_cache=True, input_sequence_length=2, linear_chunk_size=2, batch_size=1,
    ))
    names = [prefix + f".in_proj_{part}" for part in ("q", "k", "v")]
    signed_parts = signed.split([64, 64, 128], dim=0)
    for part, expected_weight, expected_signed in zip(
        ("q", "k", "v"), weights.split([64, 64, 128], dim=0), signed_parts, strict=True,
    ):
        projection = getattr(wrapped, f"in_proj_{part}")
        torch.testing.assert_close(projection.weight, expected_weight, rtol=0, atol=0)
        torch.testing.assert_close(projection.quant_weight, expected_signed, rtol=0, atol=0)

    class Projections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            parent = self
            parts = prefix.split(".")
            for part in parts[:-1]:
                parent.add_module(part, torch.nn.Module())
                parent = getattr(parent, part)
            parent.add_module(parts[-1], wrapped)

        def forward(self, x):
            return tuple(self.get_submodule(n)(x) for n in names)

    export = yaml.safe_load(CONFIG.read_text())["export"]
    resolved = apply_checkpoint_precision(export, root)
    nodes = resolved["model"]["quant_scheme"]["nodes"]
    for n in names:
        for alias in (n, n.replace(".", "_")):
            assert nodes[alias]["quant_type"] == f"w{bits}a8h1_sefp"
    unsplit = copy.deepcopy(export)
    unsplit["model"]["split_conv_cache"] = False
    assert names[0] not in apply_checkpoint_precision(unsplit, root)["model"]["quant_scheme"]["nodes"]
    conflict = copy.deepcopy(export)
    conflict["model"]["quant_scheme"]["nodes"][names[0]] = {"quant_type": f"w{4 if bits != 4 else 8}a8h1_sefp"}
    with pytest.raises(ValueError, match="conflicts"):
        apply_checkpoint_precision(conflict, root)

    inputs = (torch.ones(1, 2, 64, dtype=torch.float16),)
    frontend = to_frontend_graph(Projections().eval(), "TorchFX", inputs)
    quantized = to_quant_graph(frontend, "XH2a", resolved["model"]["quant_scheme"])
    ptq_quantize(
        quantized, [list(inputs)], PrecisionMode.ALIGNED, ["cpu"],
        auto_release_unused_parameters=True, infer_shape=False,
    )
    for n, expected in zip(names, signed_parts, strict=True):
        actual = quantized.get_submodule(n).qweight.reshape(64, expected.shape[0]).T
        torch.testing.assert_close(actual, expected.to(torch.int8), rtol=0, atol=0)
    output = tmp_path / "split_qkv.onnx"
    convert_quanted_model_to_hmonnx(quantized, list(inputs), str(output))
    linear_nodes = [n for n in onnx.load(output).graph.node if n.op_type == "Linear"]
    assert len(linear_nodes) == 3
    for node in linear_nodes:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        assert attrs["hmfp_weight_man_bit"] == bits


def test_skip_export_requires_plan_only_when_enabled(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text("{}")
    workflow = Qwen35Workflow(str(root), str(CONFIG))
    result = QuantResult(raw_model_dir=str(root), skipped=True)
    skipped = workflow.workflow_config.with_overrides({"quant": None})
    with pytest.raises(ValueError, match="no precision plan"):
        workflow._build_export_config(result, skipped)
    disabled = workflow.workflow_config.with_overrides({"quant.auto_mixed_precision": False})
    assert workflow._build_export_config(result, disabled)["model"]["hf_model"] == str(root)
