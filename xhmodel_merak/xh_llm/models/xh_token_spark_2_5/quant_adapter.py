"""Translate Spark workflow configuration into the GPTQModel worker contract.

GPTQModel and its bundled AutoRound own calibration, optimization and checkpoint writing.
The adapter only resolves resources, stages locally registered model metadata,
and hands the resulting checkpoint back to the export workflow.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ...workflows.result import QuantResult


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_calibration(value: str) -> Path:
    if value.startswith(("xh2modelzoo://", "repo://")):
        path = repo_root() / value.split("://", 1)[1]
    else:
        path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Spark calibration JSONL does not exist: {path}")
    return path.resolve()


def layer_config_arg(config: dict) -> str:
    """Serialize AutoRound's unquoted CLI syntax (JSON quotes are literal there)."""
    entries = []
    for pattern, values in config.items():
        re.compile(pattern)
        if any(character in pattern for character in "{}:, \t\n"):
            raise ValueError(f"Unsupported AutoRound CLI layer pattern: {pattern!r}")
        if set(values) - {"bits", "data_type"}:
            raise ValueError(f"Unsupported layer_config fields: {values}")
        if values.get("bits") not in (4, 8, 16):
            raise ValueError(f"Spark layer bits must be 4, 8 or 16: {values}")
        data_type = values.get("data_type", "int")
        if data_type not in {"int", "int_sym_gptq", "fp"}:
            raise ValueError(f"Unsupported layer data_type: {data_type}")
        entries.append(f"{pattern}:{{bits:{values['bits']},data_type:{data_type}}}")
    return "{" + ",".join(entries) + "}"


def build_autoround_command(*, model_dir, output_dir, device, quant_cfg) -> list[str]:
    allowed = {
        "algorithm",
        "method",
        "artifact_format",
        "output_format",
        "bits",
        "group_size",
        "sym",
        "data_type",
        "layer_config",
        "iters",
        "seed",
        "format",
        "calibration",
        "runtime",
    }
    if set(quant_cfg) - allowed:
        raise ValueError(f"Unknown Spark quant fields: {sorted(set(quant_cfg) - allowed)}")
    if quant_cfg.get("algorithm") not in {"gptqmodel", "autoround"} or quant_cfg.get("method") != "autoround":
        raise ValueError("Spark quant requires method=autoround and algorithm=gptqmodel or autoround")
    for key in ("artifact_format", "output_format"):
        if quant_cfg.get(key, "gptqmodel_hf") != "gptqmodel_hf":
            raise ValueError(f"Spark {key} must be gptqmodel_hf")
    if quant_cfg.get("format", "auto_gptq") != "auto_gptq":
        raise ValueError("Spark AutoRound format must be auto_gptq")
    if quant_cfg.get("bits", 4) not in (4, 8) or quant_cfg.get("group_size", 64) != 64:
        raise ValueError("Spark supports W4/W8 AutoRound with group_size=64")
    if quant_cfg.get("sym", True) is not True:
        raise ValueError("Spark export requires symmetric quantization")
    if quant_cfg.get("data_type", "int_sym_gptq") not in {"int_sym_gptq", "int"}:
        raise ValueError("Spark AutoRound data_type must be int_sym_gptq or int")
    calibration = quant_cfg["calibration"]
    runtime = quant_cfg.get("runtime", {})
    if set(calibration) - {"jsonl", "text_key", "nsamples", "seqlen", "concat"}:
        raise ValueError("Unknown Spark calibration fields")
    if set(runtime) - {"batch_size", "gradient_accumulate_steps", "trust_remote_code", "low_gpu_mem_usage"}:
        raise ValueError("Unknown Spark runtime fields")
    if calibration.get("text_key", "text") != "text":
        raise ValueError("AutoRound local JSONL requires text_key=text")
    dataset = str(resolve_calibration(calibration["jsonl"]))
    if calibration.get("concat", True):
        dataset += ":concat"
    device = str(device)
    if device == "cuda":
        device = "0"
    elif re.fullmatch(r"cuda:\d+", device):
        device = device.split(":")[1]
    command = [sys.executable, "-m", "xhmodel_merak.xh_llm.models.xh_token_spark_2_5.autoround_cli", str(model_dir)]
    options = {
        "output_dir": str(output_dir),
        "scheme": f"W{quant_cfg.get('bits', 4)}A16",
        "bits": quant_cfg.get("bits", 4),
        "group_size": 64,
        "data_type": quant_cfg.get("data_type", "int_sym_gptq"),
        "iters": quant_cfg.get("iters", 200),
        "seed": quant_cfg.get("seed", 42),
        "nsamples": calibration.get("nsamples", 128),
        "seqlen": calibration.get("seqlen", 2048),
        "batch_size": runtime.get("batch_size", 8),
        "gradient_accumulate_steps": runtime.get("gradient_accumulate_steps", 1),
        "dataset": dataset,
        "device_map": device,
        "format": "auto_gptq",
    }
    for key in ("iters", "nsamples", "seqlen", "batch_size", "gradient_accumulate_steps"):
        if not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f"Spark {key} must be a positive integer")
    for key, value in options.items():
        command.extend([f"--{key}", str(value)])
    # Explicit exclusions keep the reference profile even if AutoRound changes
    # its small-Linear/embedding defaults. Norms are not Linear targets.
    layers = dict(quant_cfg.get("layer_config", {}))
    for pattern in ("model.embedding", "lm_head", "model.layers.[0-9]+.self_attn.g_proj"):
        if pattern in layers and layers[pattern].get("bits") != 16:
            raise ValueError(f"Spark AutoRound must leave {pattern} in floating point")
        layers[pattern] = {"bits": 16, "data_type": "fp"}
    command.extend(["--layer_config", layer_config_arg(layers)])
    if runtime.get("low_gpu_mem_usage", False):
        command.append("--low_gpu_mem_usage")
    if not runtime.get("trust_remote_code", False):
        command.append("--disable_trust_remote_code")
    return command


@contextmanager
def staged_model(model_dir):
    source = Path(model_dir).resolve()
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "spark2_5":
        raise ValueError("Spark workflow requires model_type=spark2_5")
    if config.get("quantization_config"):
        raise ValueError("Already quantized checkpoint: use quant=None to export without requantization")
    # CLI bootstrap registers the repository classes. Removing auto_map selects
    # them even when the caller allows remote code; never edit the input model.
    config.pop("auto_map", None)
    with tempfile.TemporaryDirectory(prefix="spark2_5_autoround_") as temporary:
        staged = Path(temporary) / source.name
        staged.mkdir()
        for entry in source.iterdir():
            if entry.name != "config.json":
                (staged / entry.name).symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
        (staged / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        yield staged


def resolve_checkpoint(output_dir) -> Path:
    root = Path(output_dir)
    candidates = [root, *(p.parent for p in root.glob("*/config.json"))]
    valid = []
    for path in candidates:
        config_path = path / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        if config.get("quantization_config") and any(path.glob("*.safetensors")):
            valid.append(path)
    if len(valid) != 1:
        raise RuntimeError(f"Expected one AutoRound checkpoint under {root}, found {valid}")
    return valid[0].resolve()


def quantize_autoround(*, model_dir, output_dir, device, quant_cfg) -> QuantResult:
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Quantization output is not empty: {output}")
    # Validate before staging or starting a GPU worker.
    build_autoround_command(model_dir=model_dir, output_dir=output, device=device, quant_cfg=quant_cfg)
    with staged_model(model_dir) as staged:
        command = build_autoround_command(model_dir=staged, output_dir=output, device=device, quant_cfg=quant_cfg)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root()) + os.pathsep + env.get("PYTHONPATH", "")
        print("[Spark AutoRound]", command, flush=True)
        subprocess.run(command, check=True, env=env)
    checkpoint = resolve_checkpoint(output)
    # Staging selects our local implementation without executing checkpoint
    # code. Restore the original dispatch metadata for source identity checks
    # and readers that explicitly opt into the checkpoint's implementation.
    source = Path(model_dir).resolve()
    source_config = json.loads((source / "config.json").read_text())
    if "auto_map" in source_config:
        config_path = checkpoint / "config.json"
        config = json.loads(config_path.read_text())
        config["auto_map"] = source_config["auto_map"]
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        for name in ("configuration_spark.py", "modeling_spark.py", "__init__.py"):
            if (source / name).is_file():
                shutil.copy2(source / name, checkpoint / name)
    return QuantResult(raw_model_dir=str(Path(model_dir).resolve()), quanted_model_dir=str(checkpoint))
