"""Serving package metadata helpers for Qwen4-Exp HMONNX exports."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch

from .cache_contract import FULL_ATTENTION, QWEN_SPARSE_ATTENTION, Qwen4ExpCacheContract
from .mtp_contract import mtp_verify_phase, resolve_mtp_verify_length


def _md5(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _copy_hf_assets(hf_model_dir: str | Path, output_dir: Path) -> str:
    source = Path(hf_model_dir)
    target = output_dir / "hf_config"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
        "merges.txt",
    ):
        src = source / name
        if src.is_file():
            shutil.copy2(src, target / name)
    return _relative(output_dir, target)


def _quant_embedding_path(output_dir: Path, embedding_state: Mapping[str, torch.Tensor] | torch.Tensor | None) -> str:
    target = output_dir / "quant_embedding.pt"
    if embedding_state is None:
        torch.save({"weight": torch.empty(0, 0, dtype=torch.float16)}, target)
    elif torch.is_tensor(embedding_state):
        torch.save({"weight": embedding_state.detach().cpu()}, target)
    else:
        torch.save({name: value.detach().cpu() for name, value in embedding_state.items()}, target)
    return _relative(output_dir, target)


def build_qwen4_exp_serving_metadata(
    *,
    output_dir: str | Path,
    model_name: str,
    hf_model_dir: str | Path,
    text_config: Any,
    model_config: Mapping[str, Any],
    prefill_hmonnx: str | Path,
    decode_hmonnx: str | Path,
    embedding_state: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
    pad_token_id: int | None = None,
    enable_page_attention: bool = False,
    visual_metadata: Mapping[str, Any] | None = None,
    mtp_metadata: Mapping[str, Any] | None = None,
    verify_hmonnx: str | Path | None = None,
) -> Path:
    """Write standard Merak metadata for Qwen4-Exp serving runtimes."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    from .cache_contract import QWEN4_EXP_LLM_CACHE_ABI, QWEN4_EXP_PAGE_CACHE_ABI

    cache_abi = QWEN4_EXP_PAGE_CACHE_ABI if enable_page_attention else QWEN4_EXP_LLM_CACHE_ABI
    context_length = int(model_config.get("context_max_length", 262144))
    contract = Qwen4ExpCacheContract.from_text_config(text_config)
    attention_layers = [
        layer.layer_idx for layer in contract.layers if layer.layer_type in {FULL_ATTENTION, QWEN_SPARSE_ATTENTION}
    ]
    prefill_length = int(model_config.get("prefill_chunk_length", 256))
    verify_length = resolve_mtp_verify_length(model_config) if verify_hmonnx is not None else 2
    verify_phase = mtp_verify_phase(verify_length)
    if verify_hmonnx is not None and context_length < verify_length:
        raise ValueError("Qwen4 MTP context length is smaller than the target verify length")
    quant_embedding = _quant_embedding_path(output_path, embedding_state)
    hf_config = _copy_hf_assets(hf_model_dir, output_path)
    prefill_path = Path(prefill_hmonnx)
    decode_path = Path(decode_hmonnx)
    if pad_token_id is None:
        pad_token_id = int(getattr(text_config, "eos_token_id", 0) or 0)

    resolved_model_config = {
        "model_name": model_name,
        "model_type": "Qwen4ExpForConditionalGeneration",
        "batch_size": int(model_config.get("batch_size", 1)),
        "context_max_length": context_length,
        "prefill_chunk_length": prefill_length,
        "num_logits_to_keep": int(model_config.get("num_logits_to_keep", 1)),
        "use_cache": True,
        "max_pe_length": int(model_config.get("max_pe_length", max(context_length, 32768))),
        "quant_scheme": dict(model_config.get("quant_scheme") or {}),
        "ple_quant_type": model_config.get("ple_quant_type", "sefp4"),
        "return_hidden_states": bool(model_config.get("return_hidden_states", False)),
        "qwen4_exp_internal_state": True,
        "qwen4_exp_attention_layers": attention_layers,
    }
    kv_cache_shape = [
        int(model_config.get("batch_size", 1)),
        contract.num_key_value_heads,
        context_length,
        contract.head_dim,
    ]
    meta = {
        "create_time": "",
        "model_config": resolved_model_config,
        "hf_config": hf_config,
        "quant_embedding": quant_embedding,
        "quant_embedding_md5": _md5(output_path / quant_embedding),
        "kv_cache": {
            "num_layers": len(attention_layers),
            "kv_cache_shape": [kv_cache_shape, list(kv_cache_shape)],
            "cache_axis": 2,
            "batch_size": int(model_config.get("batch_size", 1)),
            "cache_dtype": str(model_config.get("cache_dtype", "float16")),
            "use_cache": True,
        },
        "prefill_hmonnx": _relative(output_path, prefill_path),
        "decode_hmonnx": _relative(output_path, decode_path),
        "prefill_hmonnx_md5": _md5(prefill_path),
        "decode_hmonnx_md5": _md5(decode_path),
        "pad_token_id": int(pad_token_id),
        "qwen4_exp_cache_contract": contract.to_dict(),
        "qwen4_exp_cache_abi": cache_abi,
        "qwen4_exp_serving": {
            "enable_page_attention": bool(enable_page_attention),
            "qsa_attention": "selected_kv_matmul_attention",
            "max_num_seqs": 1,
        },
    }
    from .ple_export import read_qwen4_exp_ple_specs

    ple_assets = {}
    phase_paths = [("prefill", prefill_path), ("decode", decode_path)]
    if verify_hmonnx is not None:
        verify_path = Path(verify_hmonnx)
        if verify_path.resolve() in {prefill_path.resolve(), decode_path.resolve()}:
            raise ValueError("Qwen4 target verify must be an independent target verify graph")
        phase_paths.append((verify_phase, verify_path))
        meta[f"{verify_phase}_hmonnx"] = _relative(output_path, verify_path)
        meta[f"{verify_phase}_hmonnx_md5"] = _md5(verify_path)
    for phase, graph_path in phase_paths:
        specs = read_qwen4_exp_ple_specs(graph_path)
        for spec in specs:
            for key in ("packed", "exponent"):
                asset = (graph_path.parent / spec[key]["path"]).resolve()
                if not asset.is_relative_to(output_path.resolve()):
                    raise ValueError("PLE serving assets must be inside the artifact root")
                spec[key]["path"] = str(asset.relative_to(output_path.resolve()))
        if specs:
            ple_assets[phase] = {"version": 1, "layers": specs}
    if ple_assets:
        meta["qwen4_exp_ple"] = ple_assets
    if visual_metadata is not None:
        meta["visual_config"] = dict(visual_metadata)
    if mtp_metadata is not None:
        separate_graphs = "prefill" in mtp_metadata and "decode" in mtp_metadata
        if separate_graphs:
            if verify_hmonnx is None:
                raise ValueError("Separate Qwen4 MTP graphs require a dedicated target verify graph")
            draft_prefill, draft_decode = mtp_metadata["prefill"], mtp_metadata["decode"]
            if (draft_prefill["input_sequence_length"] != prefill_length
                    or draft_decode["input_sequence_length"] != 1):
                raise ValueError("Qwen4 MTP graph sequence lengths do not match prefill/decode")
            if any(draft_prefill[key] != draft_decode[key] for key in ("cache_contract", "hidden_size", "hc_count")):
                raise ValueError("Qwen4 MTP graphs must share the same HC and cache contract")
            if (draft_decode["hidden_size"] != int(text_config.hidden_size)
                    or draft_decode["hc_count"] != int(text_config.hc_count)):
                raise ValueError("Qwen4 MTP hidden size and HC streams must match the target")
            if any(part.get("cache_abi", cache_abi) != cache_abi for part in (draft_prefill, draft_decode)):
                raise ValueError("Qwen4 target and MTP graphs must share the same cache ABI")
            prefill_graph = _relative(output_path, draft_prefill["onnx_path"])
            graph = _relative(output_path, draft_decode["onnx_path"])
            if prefill_graph == graph:
                raise ValueError("Qwen4 MTP prefill and decode must be independent graphs")
        else:
            draft_decode = mtp_metadata
            graph = prefill_graph = _relative(output_path, mtp_metadata["onnx_path"])
        meta["spec_decode"] = {
            "mode": "mtp",
            "runtime_contract_version": 3 if separate_graphs else 2,
            "num_draft_tokens": 1,
            "verify_length": verify_length,
            "target_verify_graph": verify_phase if separate_graphs else "prefill",
            "target_verify_padding": verify_length if separate_graphs else prefill_length,
            "draft_prefill_onnx": prefill_graph,
            "draft_decode_onnx": graph,
            "draft_onnx": graph,
            "draft": {
                "prefill_hmonnx": prefill_graph,
                "decode_hmonnx": graph,
                "cache_contract": draft_decode["cache_contract"],
                "cache_abi": draft_decode.get("cache_abi", cache_abi),
                "hidden_size": draft_decode["hidden_size"],
                "hc_count": draft_decode["hc_count"],
            },
        }
        if separate_graphs:
            meta["spec_decode"]["draft"].update(prefill_chunk_length=prefill_length, decode_chunk_length=1)
            meta["spec_decode"]["target"] = {
                "verify_hmonnx": meta[f"{verify_phase}_hmonnx"], "verify_chunk_length": verify_length,
            }
    meta_path = output_path / "golden_meta_info.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    merak_config = {
        "architectures": ["MerakForCausalLM"],
        "config_format": "merak_llm",
        "load_format": "merak_llm",
        "xh_model": {
            "model_type": "hmonnx",
            "meta_info": "golden_meta_info.json",
        },
        "enable_page_attention": bool(enable_page_attention),
        "model_type": "merak_llm",
        "max_num_seqs": 1,
        "disable_prefix_caching": True,
    }
    (output_path / "merak_config.json").write_text(
        json.dumps(merak_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta_path


__all__ = ["build_qwen4_exp_serving_metadata"]
