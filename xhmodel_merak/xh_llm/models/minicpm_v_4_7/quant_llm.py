"""GPTQ / AutoRound quantization of MiniCPM-V-4.7 Dense and MoE LLM backbones."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_REPO_RESOURCE_PREFIXES = ("xh2modelzoo://", "repo://")
_DEFAULT_AUTOROUND_DATASET = "xh2modelzoo://data/calib_data/NeelNanda-pile-10k.jsonl"
_LEGACY_AUTOROUND_DATASETS = {"NeelNanda/pile-10k", "pile-10k"}
_AUTOROUND_LINEAR_ATTRS = (
    "bits",
    "group_size",
    "sym",
    "data_type",
    "act_bits",
    "act_group_size",
    "act_sym",
    "act_data_type",
    "act_dynamic",
    "super_bits",
    "super_group_size",
    "transform_config",
    "scale",
    "zp",
    "weight_global_scale",
    "scale_dtype",
    "q_scale_thresh",
    "tuning_device",
    "fixed_by_user",
)
_ATTN_LINEAR_SUFFIXES = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".linear_attn.in_proj_qkv",
    ".linear_attn.in_proj_z",
    ".linear_attn.in_proj_b",
    ".linear_attn.in_proj_a",
    ".linear_attn.out_proj",
)
_SHARED_EXPERT_SUFFIXES = (
    ".mlp.shared_expert.gate_proj",
    ".mlp.shared_expert.up_proj",
    ".mlp.shared_expert.down_proj",
)
_EXPERT_UP_GATE_SUFFIXES = (
    ".gate_proj",
    ".up_proj",
)
# CausalLM packed-expert fallback (before AutoRound unfuse).
_CAUSAL_EXPERT_GATE_UP_PATTERN = "model.layers.*.mlp.experts.*.(gate_proj|up_proj)"
_CAUSAL_EXPERT_DOWN_PATTERN = "model.layers.*.mlp.experts.*.down_proj"


def _load_calibration_samples(
    calibration_jsonl: str,
    *,
    nsamples: int,
    sampling_seed: int | None,
) -> list[str]:
    records: list[str] = []
    with open(calibration_jsonl, encoding="utf-8") as handle:
        for line in handle:
            text = json.loads(line).get("text", "")
            if text:
                records.append(text)

    if not records:
        raise ValueError(f"calibration jsonl has no usable 'text' samples: {calibration_jsonl}")
    if sampling_seed is None or len(records) <= nsamples:
        return records[:nsamples]

    import numpy as np

    indices = np.random.default_rng(sampling_seed).choice(
        len(records),
        size=nsamples,
        replace=False,
    )
    return [records[int(index)] for index in indices]


def _resolve_calibration_path(calibration_jsonl: str) -> str:
    """Resolve calibration path with the same repo-resource conventions as 4.5."""
    if os.path.isabs(calibration_jsonl) and os.path.isfile(calibration_jsonl):
        return calibration_jsonl
    relative_path = calibration_jsonl
    for prefix in _REPO_RESOURCE_PREFIXES:
        if relative_path.startswith(prefix):
            relative_path = relative_path.removeprefix(prefix)
            break
    relative_path = relative_path.lstrip("/")
    repo_root = Path(__file__).resolve().parents[4]

    candidates: list[Path] = []
    data_root = os.environ.get("XH2MODELZOO_DATA_ROOT")
    if data_root:
        root = Path(data_root).expanduser()
        candidates.append(root / relative_path)
        if relative_path.startswith("data/"):
            candidates.append(root / relative_path.removeprefix("data/"))
    env_root = os.environ.get("XH2MODELZOO_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser() / relative_path)
    candidates.append(Path.cwd() / relative_path)
    candidates.append(repo_root / relative_path)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return str(candidate.resolve())
    raise ValueError(
        f"calibration jsonl not found: {calibration_jsonl} "
        "(checked XH2MODELZOO_DATA_ROOT, XH2MODELZOO_ROOT, cwd and repo root)"
    )


def quantize_minicpm_v47_llm_gptq(
    *,
    model_dir: str,
    output_dir: str,
    quant_cfg: Mapping[str, Any],
    device: str,
) -> str:
    """GPTQ-quantize the MiniCPM-V-4.7 Dense text backbone.

    GPTQModel cannot load the MiniCPM remote-code host, so the embedded
    Qwen3.5 text tower is extracted as a standalone ``Qwen3_5ForCausalLM``
    checkpoint, quantized with a text-only corpus, then dequantized back to
    float ``nn.Linear`` weights for HMONNX export via ``quant_weight``.
    """
    del device  # GPTQModel selects CUDA devices from the process environment.

    calibration_jsonl = str(quant_cfg.get("calibration_jsonl") or "")
    if not calibration_jsonl:
        raise ValueError("MiniCPM-V-4.7 GPTQ requires quant.calibration_jsonl (text-only corpus)")
    calibration_jsonl = _resolve_calibration_path(calibration_jsonl)

    gptqmodel_source = quant_cfg.get("gptqmodel_source") or os.environ.get("GPTQMODEL_SOURCE")
    if gptqmodel_source and gptqmodel_source not in sys.path:
        sys.path.insert(0, str(gptqmodel_source))

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeConfig

    nsamples = int(quant_cfg.get("nsamples", 128))
    seqlen = int(quant_cfg.get("seqlen", 1024))
    sampling_seed = quant_cfg.get("sampling_seed")
    samples = _load_calibration_samples(
        calibration_jsonl,
        nsamples=nsamples,
        sampling_seed=None if sampling_seed is None else int(sampling_seed),
    )
    if len(samples) < 16:
        print(
            f"[GPTQ] warning: calibration set has only {len(samples)} sample(s); "
            "more samples generally improve quantization quality"
        )

    group_size = int(quant_cfg.get("group_size", 64))
    if group_size != 64:
        raise ValueError(f"MiniCPM-V-4.7 GPTQ requires group_size=64 for XH2A, got {group_size}")

    quantize_config = QuantizeConfig(
        bits=int(quant_cfg.get("bits", 4)),
        group_size=group_size,
        desc_act=False,
        damp_percent=float(quant_cfg.get("damp_percent", 0.01)),
        offload_to_disk_path=str(Path(output_dir) / "gptqmodel_offload"),
    )

    backbone_dir = Path(output_dir) / "llm_backbone"
    prebuilt = quant_cfg.get("llm_backbone_dir")
    if prebuilt and Path(prebuilt).is_dir() and (Path(prebuilt) / "pytorch_model.bin").is_file():
        shutil.copytree(Path(prebuilt), backbone_dir, dirs_exist_ok=True)
    else:
        _export_dense_llm_backbone(model_dir, backbone_dir)

    model = GPTQModel.from_pretrained(
        str(backbone_dir),
        quantize_config=quantize_config,
        trust_remote_code=True,
    )
    model.quantize(samples, batch_size=1, calibration_concat_size=seqlen)

    quanted_model_dir = str(Path(output_dir) / "gptq_llm")
    model.save_quantized(quanted_model_dir)

    dequant_dir = Path(output_dir) / "gptq_llm_dequant"
    _dequantize_gptq_llm(quanted_model_dir, quantize_config, dequant_dir)
    return str(dequant_dir)


def _dequantize_gptq_llm(quanted_model_dir: str, quantize_config: Any, dequant_dir: Path) -> None:
    """Load the GPTQ artifact with the Torch backend and dequantize to nn.Linear."""
    import torch
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch import dequantize_model

    if dequant_dir.exists():
        shutil.rmtree(dequant_dir)
    dequant_dir.mkdir(parents=True, exist_ok=True)

    model = GPTQModel.from_quantized(
        quanted_model_dir,
        quantize_config=quantize_config,
        trust_remote_code=True,
        backend=BACKEND.TORCH,
    )
    model = dequantize_model(model)

    # GPTQModel wraps the causal LM; its state dict uses Qwen3_5ForCausalLM keys
    # (model.* + lm_head) which text_model remaps onto MiniCPM language_model.
    inner = model.model
    state = {key: value.detach().cpu().clone() for key, value in inner.state_dict().items()}
    torch.save(state, dequant_dir / "pytorch_model.bin")
    shutil.copy2(Path(quanted_model_dir) / "config.json", dequant_dir / "config.json")


def _export_dense_llm_backbone(model_dir: str, backbone_dir: Path) -> None:
    """Extract ``Qwen3_5ForCausalLM`` weights from the MiniCPM-V-4.7 host."""
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    if backbone_dir.exists():
        shutil.rmtree(backbone_dir)
    backbone_dir.mkdir(parents=True, exist_ok=True)

    host_cfg = json.loads(Path(model_dir, "config.json").read_text(encoding="utf-8"))
    text_cfg_dict = host_cfg.get("text_config")
    if not isinstance(text_cfg_dict, Mapping):
        raise ValueError(f"MiniCPM-V-4.7 config missing text_config: {model_dir}")

    text_cfg = Qwen3_5TextConfig(**dict(text_cfg_dict))
    text_cfg.architectures = ["Qwen3_5ForCausalLM"]
    text_cfg.save_pretrained(str(backbone_dir))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.save_pretrained(str(backbone_dir))

    state = _load_dense_causal_state(model_dir)
    torch.save(state, backbone_dir / "pytorch_model.bin")


def _load_dense_causal_state(model_dir: str) -> dict[str, Any]:
    """Map MiniCPM ``model.language_model.*`` (+ ``lm_head``) to CausalLM keys."""
    import torch
    from safetensors import safe_open

    model_path = Path(model_dir)
    single = model_path / "model.safetensors"
    index = model_path / "model.safetensors.index.json"
    shard_map: dict[str, str]
    if single.is_file():
        shard_map = {key: str(single) for key in _iter_safetensor_keys(single)}
    elif index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, Mapping):
            raise ValueError(f"invalid safetensors index: {index}")
        shard_map = {
            str(key): str(model_path / shard)
            for key, shard in weight_map.items()
            if str(key).startswith("model.language_model.") or str(key) == "lm_head.weight"
        }
    else:
        raise FileNotFoundError(f"MiniCPM-V-4.7 weights not found under {model_dir}")

    state: dict[str, Any] = {}
    by_shard: dict[str, list[str]] = {}
    for key, shard in shard_map.items():
        if not (key.startswith("model.language_model.") or key == "lm_head.weight"):
            continue
        by_shard.setdefault(shard, []).append(key)

    for shard, keys in by_shard.items():
        with safe_open(shard, framework="pt") as handle:
            for key in keys:
                tensor = handle.get_tensor(key).detach().cpu().clone()
                if key == "lm_head.weight":
                    state["lm_head.weight"] = tensor
                else:
                    state["model." + key[len("model.language_model.") :]] = tensor

    if "lm_head.weight" not in state:
        raise KeyError(f"lm_head.weight missing from MiniCPM-V-4.7 checkpoint: {model_dir}")
    if not any(key.startswith("model.") for key in state):
        raise KeyError(f"model.language_model.* weights missing from MiniCPM-V-4.7 checkpoint: {model_dir}")
    return state


def _iter_safetensor_keys(path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        return list(handle.keys())


def apply_causal_dequant_to_minicpm_host(full_model: Any, quant_weight: str) -> None:
    """Load a CausalLM dequant ``pytorch_model.bin`` onto MiniCPM text modules."""
    import torch

    if not os.path.isfile(quant_weight):
        raise FileNotFoundError(f"quant_weight backbone not found: {quant_weight}")
    backbone_state = torch.load(quant_weight, weights_only=True, map_location="cpu")
    backbone_state = {
        key: value for key, value in backbone_state.items() if not str(key).endswith(".bias")
    }

    language_state: dict[str, Any] = {}
    lm_head_weight = None
    for key, value in backbone_state.items():
        if key == "lm_head.weight":
            lm_head_weight = value
            continue
        if key.startswith("model."):
            language_state[key[len("model.") :]] = value
            continue
        raise KeyError(f"unexpected CausalLM dequant key for MiniCPM-V-4.7: {key}")

    missing, unexpected = full_model.model.language_model.load_state_dict(language_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to apply MiniCPM-V-4.7 CausalLM dequant to language_model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if lm_head_weight is None:
        raise KeyError("CausalLM dequant is missing lm_head.weight")
    with torch.no_grad():
        full_model.lm_head.weight.copy_(lm_head_weight)


def quantize_minicpm_v47_llm_autoround(
    *,
    model_dir: str,
    output_dir: str,
    quant_cfg: Mapping[str, Any],
    device: str,
) -> str:
    """AutoRound-quantize the MiniCPM-V-4.7 MoE text backbone in-process.

    Loads the MiniCPM host, assembles an in-memory ``Qwen3_5MoeForCausalLM``,
    runs AutoRound with mixed bits (attn/shared 8, experts 4 by default), then
    restores packed MoE experts and emits a float CausalLM ``pytorch_model.bin``
    for ``quant_weight`` / ``apply_causal_dequant_to_minicpm_host``.
    """
    import gc

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    autoround_source = (
        quant_cfg.get("autoround_source")
        or os.environ.get("AUTOROUND_SOURCE")
        or (Path(os.environ.get("GPTQMODEL_SOURCE", "")) / "third_party" / "auto-round")
    )
    if autoround_source and Path(autoround_source).is_dir():
        autoround_source = str(Path(autoround_source).resolve())
        if autoround_source not in sys.path:
            sys.path.insert(0, autoround_source)

    from auto_round import AutoRound
    from xhmodel_merak.xh_llm._dequant_converter import restore_autoround_qwen3_5_moe_sparse_block

    group_size = int(quant_cfg.get("group_size", 64))
    if group_size != 64:
        raise ValueError(f"MiniCPM-V-4.7 AutoRound requires group_size=64 for XH2A, got {group_size}")

    bits = int(quant_cfg.get("bits", 4))
    sym = _as_bool(quant_cfg.get("sym", True))
    data_type = str(quant_cfg.get("data_type", "int_sym_xh"))
    if data_type in {"int_sym_gptq", "int_sym_xh"} and not sym:
        raise ValueError(f"quant.data_type={data_type} requires quant.sym=true")

    calibration_cfg = _section(quant_cfg, "calibration")
    runtime_cfg = _section(quant_cfg, "runtime")
    moe_cfg = _section(quant_cfg, "moe")

    dataset = _resolve_autoround_dataset(
        calibration_cfg.get(
            "dataset",
            calibration_cfg.get(
                "jsonl",
                quant_cfg.get("dataset", quant_cfg.get("calibration_jsonl", _DEFAULT_AUTOROUND_DATASET)),
            ),
        )
    )
    nsamples = int(calibration_cfg.get("nsamples", quant_cfg.get("nsamples", 256)))
    seqlen = int(calibration_cfg.get("seqlen", quant_cfg.get("seqlen", 2048)))
    iters = int(quant_cfg.get("iters", 400))
    batch_size = int(runtime_cfg.get("batch_size", quant_cfg.get("batch_size", 8)))
    gradient_accumulate_steps = int(
        runtime_cfg.get(
            "gradient_accumulate_steps",
            quant_cfg.get("gradient_accumulate_steps", 1),
        )
    )
    seed = int(runtime_cfg.get("seed", quant_cfg.get("seed", 42)))
    device_map = runtime_cfg.get("device_map", quant_cfg.get("device_map", "0"))
    low_gpu_mem_usage = _as_bool(
        runtime_cfg.get("low_gpu_mem_usage", quant_cfg.get("low_gpu_mem_usage", True))
    )
    save_quantized = _as_bool(quant_cfg.get("save_quantized", False))
    export_format = str(quant_cfg.get("format", quant_cfg.get("autoround_format", "auto_round:gptqmodel")))

    attn_bits = _optional_int(moe_cfg.get("attn_bits", moe_cfg.get("self_attn_bits", 8)))
    shared_expert_bits = _optional_int(moe_cfg.get("shared_expert_bits", 8))
    expert_bits = _optional_int(moe_cfg.get("expert_bits", bits))
    expert_up_gate_bits = _optional_int(moe_cfg.get("expert_up_gate_bits"))
    expert_down_bits = _optional_int(moe_cfg.get("expert_down_bits"))

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    dequant_dir = out_root / "autoround_llm_dequant"
    if dequant_dir.exists():
        shutil.rmtree(dequant_dir)
    dequant_dir.mkdir(parents=True, exist_ok=True)

    print(f"[AutoRound] loading MiniCPM-V-4.7 MoE host from {model_dir}")
    host = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    causal = _build_moe_causal_from_host(host)
    del host
    gc.collect()

    layer_config = _build_moe_layer_config(
        causal,
        llm_group_size=group_size,
        attn_bits=attn_bits,
        shared_expert_bits=shared_expert_bits,
        expert_bits=expert_bits,
        expert_up_gate_bits=expert_up_gate_bits,
        expert_down_bits=expert_down_bits,
    )
    _print_moe_override_summary(layer_config)

    ar_device = _normalize_autoround_device(device)
    print(
        f"[AutoRound] quantize CausalLM bits={bits} group_size={group_size} "
        f"data_type={data_type} iters={iters} nsamples={nsamples} dataset={dataset} "
        f"device={ar_device} device_map={device_map}"
    )
    ar = AutoRound(
        model=causal,
        tokenizer=tokenizer,
        bits=bits,
        group_size=group_size,
        sym=sym,
        data_type=data_type,
        iters=iters,
        seqlen=seqlen,
        nsamples=nsamples,
        batch_size=batch_size,
        gradient_accumulate_steps=gradient_accumulate_steps,
        dataset=dataset,
        seed=seed,
        layer_config=layer_config or None,
        device=ar_device,
        device_map=device_map,
        low_gpu_mem_usage=low_gpu_mem_usage,
    )
    quantized_model, _ = ar.quantize()
    if quantized_model is not None:
        causal = quantized_model

    if save_quantized:
        quanted_dir = out_root / "autoround_llm"
        quanted_dir.mkdir(parents=True, exist_ok=True)
        print(f"[AutoRound] optional save_quantized -> {quanted_dir} format={export_format}")
        ar.save_quantized(str(quanted_dir), format=export_format)

    _strip_autoround_linear_attrs(causal)
    _restore_autoround_moe_blocks(causal, restore_autoround_qwen3_5_moe_sparse_block)
    _emit_causal_float_bin(causal, dequant_dir)

    del ar, causal, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(dequant_dir)


def _build_moe_causal_from_host(host: Any) -> Any:
    """Assemble ``Qwen3_5MoeForCausalLM`` from MiniCPM host text modules."""
    import torch
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    text_cfg = host.config.text_config
    if isinstance(text_cfg, Qwen3_5MoeTextConfig):
        causal_cfg = text_cfg
    else:
        text_dict = text_cfg.to_dict() if hasattr(text_cfg, "to_dict") else dict(text_cfg)
        text_dict.pop("architectures", None)
        causal_cfg = Qwen3_5MoeTextConfig(**text_dict)
    causal_cfg.architectures = ["Qwen3_5MoeForCausalLM"]

    with torch.device("meta"):
        causal = Qwen3_5MoeForCausalLM(causal_cfg)
    # Host language_model is already Qwen3_5MoeTextModel via AutoModel.from_config.
    causal.model = host.model.language_model
    causal.lm_head = host.lm_head
    if getattr(host, "generation_config", None) is not None:
        causal.generation_config = host.generation_config
    causal.eval()
    return causal


def _build_moe_layer_config(
    model: Any,
    *,
    llm_group_size: int,
    attn_bits: int | None,
    shared_expert_bits: int | None,
    expert_bits: int | None,
    expert_up_gate_bits: int | None,
    expert_down_bits: int | None,
) -> dict[str, dict[str, int]]:
    """Build mixed-bit overrides for CausalLM ``model.layers.*`` modules."""
    import torch

    if all(
        value is None
        for value in (attn_bits, shared_expert_bits, expert_bits, expert_up_gate_bits, expert_down_bits)
    ):
        return {}

    layer_config: dict[str, dict[str, int]] = {}
    saw_expert_linear = False
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        if attn_bits is not None and name.endswith(_ATTN_LINEAR_SUFFIXES):
            layer_config[name] = {"bits": attn_bits, "group_size": llm_group_size}
            continue

        if shared_expert_bits is not None and name.endswith(_SHARED_EXPERT_SUFFIXES):
            layer_config[name] = {"bits": shared_expert_bits, "group_size": llm_group_size}
            continue

        if ".mlp.experts." not in name:
            continue
        saw_expert_linear = True

        if expert_down_bits is not None and name.endswith(".down_proj"):
            layer_config[name] = {"bits": expert_down_bits, "group_size": llm_group_size}
            continue

        if expert_up_gate_bits is not None and name.endswith(_EXPERT_UP_GATE_SUFFIXES):
            layer_config[name] = {"bits": expert_up_gate_bits, "group_size": llm_group_size}
            continue

        if expert_bits is not None and name.endswith((".gate_proj", ".up_proj", ".down_proj")):
            layer_config[name] = {"bits": expert_bits, "group_size": llm_group_size}

    if not saw_expert_linear and any(
        value is not None for value in (expert_bits, expert_up_gate_bits, expert_down_bits)
    ):
        gate_up_bits = expert_up_gate_bits if expert_up_gate_bits is not None else expert_bits
        down_bits = expert_down_bits if expert_down_bits is not None else expert_bits
        if gate_up_bits is not None:
            layer_config[_CAUSAL_EXPERT_GATE_UP_PATTERN] = {
                "bits": gate_up_bits,
                "group_size": llm_group_size,
            }
        if down_bits is not None:
            layer_config[_CAUSAL_EXPERT_DOWN_PATTERN] = {
                "bits": down_bits,
                "group_size": llm_group_size,
            }
    return layer_config


def _print_moe_override_summary(layer_config: Mapping[str, Any]) -> None:
    if not layer_config:
        print("[AutoRound] LLM mixed-bit overrides: none")
        return
    counts = {"attention": 0, "shared_expert": 0, "experts": 0, "patterns": 0}
    for name in layer_config:
        if any(token in name for token in ("*", "|", "(", ")")):
            counts["patterns"] += 1
        if name.endswith(_ATTN_LINEAR_SUFFIXES):
            counts["attention"] += 1
        elif any(token in name for token in _SHARED_EXPERT_SUFFIXES):
            counts["shared_expert"] += 1
        elif ".mlp.experts." in name:
            counts["experts"] += 1
    print(
        "[AutoRound] LLM mixed-bit overrides: "
        f"attention={counts['attention']}, shared_expert={counts['shared_expert']}, "
        f"experts={counts['experts']}, patterns={counts['patterns']}"
    )


def _strip_autoround_linear_attrs(model: Any) -> None:
    import torch

    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        for attr in _AUTOROUND_LINEAR_ATTRS:
            if hasattr(module, attr):
                try:
                    delattr(module, attr)
                except Exception:
                    pass
        for attr_name in list(vars(module)):
            if attr_name.startswith("w_") or attr_name.startswith("act_"):
                try:
                    delattr(module, attr_name)
                except Exception:
                    pass


def _restore_autoround_moe_blocks(model: Any, restore_fn: Any) -> None:
    block_names = [
        name
        for name, module in model.named_modules()
        if type(module).__name__ == "LinearQwen3_5MoeSparseMoeBlock"
    ]
    print(f"[AutoRound] restoring {len(block_names)} LinearQwen3_5MoeSparseMoeBlock modules")
    for name in block_names:
        module = model.get_submodule(name) if name else model
        restored = restore_fn(module)
        if not name:
            raise RuntimeError("unexpected root-level LinearQwen3_5MoeSparseMoeBlock")
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, restored)


def _emit_causal_float_bin(model: Any, dequant_dir: Path) -> None:
    import torch

    dequant_dir.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    # Keep CausalLM keys (model.* + lm_head) expected by apply_causal_dequant_to_minicpm_host.
    torch.save(state, dequant_dir / "pytorch_model.bin")
    if hasattr(model, "config") and hasattr(model.config, "to_json_file"):
        model.config.to_json_file(str(dequant_dir / "config.json"))
    print(f"[AutoRound] wrote float CausalLM dequant to {dequant_dir / 'pytorch_model.bin'}")


def _resolve_autoround_dataset(dataset: Any) -> str:
    value = str(dataset or _DEFAULT_AUTOROUND_DATASET).strip()
    if value in _LEGACY_AUTOROUND_DATASETS:
        value = _DEFAULT_AUTOROUND_DATASET
    if value.startswith(_REPO_RESOURCE_PREFIXES) or value.endswith(".jsonl") or os.path.isfile(value):
        return _resolve_calibration_path(value)
    return value


def _normalize_autoround_device(device: str) -> str:
    token = str(device).strip()
    if token.isdigit():
        return f"cuda:{token}"
    if token.startswith("cuda"):
        return token
    return token


def _section(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = cfg.get(key) or {}
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"quant.{key} must be a mapping when provided")
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "apply_causal_dequant_to_minicpm_host",
    "quantize_minicpm_v47_llm_autoround",
    "quantize_minicpm_v47_llm_gptq",
]
