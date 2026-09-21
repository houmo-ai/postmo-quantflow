"""Translate workflow parameters to GPTQModel's public GLM AutoRound recipe."""

from pathlib import Path


def quantize_glm53(*, model_dir, output_dir, device, config):
    from gptqmodel.recipes.glm5_3_flash_autoround import quantize_glm5_3_flash_autoround
    from ..qwen3_5.quant_adapter import _resolve_calibration_value

    allowed = {
        "algorithm",
        "method",
        "artifact_format",
        "bits",
        "group_size",
        "sym",
        "seed",
        "calibration",
        "runtime",
        "moe",
        "prepared_model_dir",
        "num_layers",
        "iters",
    }
    sections = {
        "calibration": {"jsonl", "dataset", "text_key", "nsamples", "seqlen"},
        "runtime": {
            "device",
            "batch_size",
            "preparation_devices",
            "device_map",
            "gradient_accumulate_steps",
        },
        "moe": {"expert_bits", "attention_bits", "dense_mlp_bits", "shared_expert_bits"},
    }
    unknown = set(config) - allowed
    for section, keys in sections.items():
        unknown.update(f"{section}.{k}" for k in set(config.get(section) or {}) - keys)
    if unknown:
        raise ValueError(f"GLM quant parameters are not consumed: {sorted(unknown)}")
    if config.get("algorithm", "gptqmodel") != "gptqmodel":
        raise ValueError("GLM mixed precision requires algorithm=gptqmodel")
    calibration, runtime, moe = (dict(config.get(key) or {}) for key in ("calibration", "runtime", "moe"))
    if any(moe.get(key, 8) != 8 for key in ("attention_bits", "dense_mlp_bits", "shared_expert_bits")):
        raise ValueError("GLM attention, dense MLP and shared experts must be W8")
    return quantize_glm5_3_flash_autoround(
        model_dir=model_dir,
        output_dir=str(Path(output_dir).resolve() / "checkpoint"),
        prepared_model_dir=config.get("prepared_model_dir"),
        num_layers=config.get("num_layers"),
        method=config.get("method", "autoround"),
        artifact_format=config.get("artifact_format", "gptqmodel_hf"),
        bits=config.get("bits", 8),
        expert_bits=moe.get("expert_bits", 4),
        group_size=config.get("group_size", 64),
        sym=config.get("sym", True),
        seed=config.get("seed", 42),
        iters=config.get("iters", 200),
        nsamples=calibration.get("nsamples", 128),
        seqlen=calibration.get("seqlen", 2048),
        calibration_jsonl=_resolve_calibration_value(calibration["jsonl"]) if calibration.get("jsonl") else None,
        calibration_dataset=calibration.get("dataset", "NeelNanda/pile-10k"),
        calibration_text_key=calibration.get("text_key", "text"),
        device=runtime.get("device", device),
        batch_size=runtime.get("batch_size", 1),
        preparation_devices=tuple(runtime.get("preparation_devices", ["cpu"])),
        device_map=runtime.get("device_map"),
        gradient_accumulate_steps=runtime.get("gradient_accumulate_steps", 2),
    )
