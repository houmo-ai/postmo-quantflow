"""Scaled real Wall checkpoint; all production build and runtime code executes."""

import json
from dataclasses import replace


def prepare(case, root, monkeypatch):
    import torch
    import yaml
    from PIL import Image
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

    from xhmodel_merak.model_catalog.registry import get_catalog
    from xhmodel_merak.xh_other_model.models.wall_oss.source import Source

    source = root / "source"
    source.mkdir()
    cfg = dict(
        model_type="qwen2_5_vl",
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        action_hidden_size=64,
        state_hidden_size=128,
        num_experts=2,
        dim_inputs=[128, 64],
        attention_moe=True,
        mlp_moe=True,
        norm_moe=True,
        causal_action_attention_mask=True,
        rope_theta=1000000.0,
        rope_scaling=dict(type="mrope", mrope_section=[16, 8, 8]),
        vocab_size=80,
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        noise_scheduler=dict(s=0.999, num_inference_timesteps=10),
        experts=[dict(hidden_size=128, intermediate_size=192), dict(hidden_size=64, intermediate_size=128)],
    )
    vc = Qwen2_5_VLVisionConfig(
        depth=2, hidden_size=160, intermediate_size=192, num_heads=2, out_hidden_size=128, fullatt_block_indexes=[1]
    )
    vc._attn_implementation = "sdpa"
    cfg["vision_config"] = vc.to_dict()
    (source / "config.json").write_text(json.dumps(cfg))
    train = dict(
        dof_config={"actions": 26},
        agent_pos_config={"state": 26},
        data=dict(
            action_horizon_flow=32,
            use_state_string_representation=True,
            state_bins=512,
            resolution=dict(face_view=448, right_wrist_view=448),
        ),
    )
    (source / "config.yml").write_text(yaml.safe_dump(train))
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|im_end|>": 2,
        "<|im_start|>": 3,
        "<|image_pad|>": 60,
        "<|video_pad|>": 61,
        "<|vision_start|>": 62,
        "<|vision_end|>": 63,
    }
    vocab.update({f"token_{i}": i for i in range(4, 60)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        additional_special_tokens=[
            "<|im_start|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        ],
    )
    processor = Qwen2_5_VLProcessor(
        tokenizer=tokenizer, image_processor=Qwen2VLImageProcessor(), video_processor=Qwen2VLVideoProcessor()
    )
    processor.save_pretrained(source)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        policy = Source(cfg)
        visual = Qwen2_5_VisionTransformerPretrainedModel(vc)
        weights = {"model.embed_tokens.weight": torch.randn(80, 128) * 0.02}
        for i, experts in enumerate(policy.layers):
            for j, e in enumerate(experts):
                state = e.state_dict()
                p = f"model.layers.{i}."
                for local, checkpoint in {
                    "norm1.weight": f"input_layernorms.{j}.weight",
                    "norm2.weight": f"post_attention_layernorms.{j}.weight",
                    "qkv.weight": f"self_attn.qkv_proj_experts.{j}.weight",
                    "qkv.bias": f"self_attn.qkv_proj_experts.{j}.bias",
                    "out.weight": f"self_attn.o_proj_experts.{j}.weight",
                    "gate_up.weight": f"moe.experts.{j}.gate_up_proj.weight",
                    "down.weight": f"moe.experts.{j}.down_proj.weight",
                }.items():
                    weights[p + checkpoint] = state[local]
        for name in ("w1", "w2", "w3"):
            weights[f"action_preprocessor.{name}.weight"] = getattr(policy, name).weight
        weights["action_preprocessor.action_proj_back.weight"] = policy.action_out.weight
        weights["model.norms.1.weight"] = policy.action_norm.weight
        state = visual.state_dict()
        for name, value in state.items():
            if ".mlp.gate_proj." in name:
                other = state[name.replace("gate_proj", "up_proj")]
                weights["visual." + name.replace("gate_proj", "gate_up_proj")] = torch.cat((value, other), 0)
            elif ".mlp.up_proj." not in name:
                weights["visual." + name] = value
        for kind in ("action", "propri"):
            weights[f"action_preprocessor.normalizer_{kind}.min.x2_normal"] = torch.zeros(26)
            weights[f"action_preprocessor.normalizer_{kind}.delta.x2_normal"] = torch.ones(26)
        save_file({k: v.detach().contiguous() for k, v in weights.items()}, str(source / "model.safetensors"))
    catalog = get_catalog()
    recipe = catalog._recipes[case.model_id]
    monkeypatch.setitem(
        catalog._recipes,
        case.model_id,
        replace(
            recipe,
            config_replacements=(("export.context_length", 1024),),
            artifact_name_template="hmquant_xh2_wall_oss_small_{deployment_precision}",
        ),
    )
    front, wrist = root / "front.png", root / "wrist.png"
    Image.new("RGB", (640, 480), (10, 100, 50)).save(front)
    Image.new("RGB", (480, 640), (100, 10, 50)).save(wrist)
    return dict(
        model_id=case.model_id,
        model_dir=str(source),
        output_dir=str(root / "artifact"),
        device="cuda:0",
        deployment_precision={"policy": "w8a8_sefp"},
    ), dict(instruction="pick up the cup", images=[str(front), str(wrist)], state=[0.0] * 26, steps=2, seed=9)


def check_inference(moved, inputs, record):
    import numpy as np

    from xhmodel_merak.model_inference import inference_model

    output = inference_model(artifact_dir=str(moved), inputs=inputs, device="cuda:0")
    assert output.model_id == "wall-oss-0.5" and (output.horizon, output.action_dim) == (32, 26)
    assert output.stats["resized_sizes"] == [(448, 336), (336, 448)]
    assert np.isfinite(np.asarray(output.actions)).all()
    from xhmodel_merak.xh_other_model.models.wall_oss.golden import dump_golden

    golden = dump_golden(moved, device="cuda:0", inputs=inputs, output_dir=moved.parent / "golden")
    assert golden["complete"] and len(golden["records"]) == 2
    assert all(item["graphs"] == ["denoise", "prefill", "visual"] for item in golden["records"])
    assert not list(moved.rglob("step_*"))
    import shutil

    import pytest

    from xhmodel_merak.xh_other_model.models.wall_oss.runtime import read_metadata

    tampered = moved.parent / "tampered"
    shutil.copytree(moved, tampered)
    meta_path = tampered / "runtime_meta_info.json"
    meta = json.loads(meta_path.read_text())
    legacy = json.loads(meta_path.read_text())
    legacy["prefill_length"] = legacy.pop("prefill_chunk_length")
    legacy["deployment_config"]["prefill_length"] = legacy["deployment_config"].pop("prefill_chunk_length")
    meta_path.write_text(json.dumps(legacy))
    assert read_metadata(tampered) == meta
    legacy["prefill_chunk_length"] = legacy["prefill_length"] + 1
    meta_path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="prefill_chunk_length.*disagree"):
        read_metadata(tampered)
    meta["context_length"] += 256
    meta["deployment_config"]["context_length"] += 256
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="physical graph ports"):
        read_metadata(tampered)
    shutil.rmtree(tampered)
    repeated = inference_model(artifact_dir=str(moved), inputs=inputs, device="cuda:0")
    np.testing.assert_array_equal(output.actions, repeated.actions)
    assert not repeated.stats["truncation"]["truncated"]
    long_inputs = {**inputs, "instruction": "token_4 " * 1500}
    long_output = inference_model(artifact_dir=str(moved), inputs=long_inputs, device="cuda:0")
    assert long_output.stats["truncation"]["truncated"]
    assert long_output.stats["prefix_tokens"] + 32 <= 1024
    assert long_output.stats["cache"]["prefix_unchanged"]
    assert np.isfinite(np.asarray(long_output.actions)).all()
    json.dumps(output.to_dict(), allow_nan=False)
    record.update(repeat_bit_exact=True, passed=True, stage="complete", inference=output.to_dict())
