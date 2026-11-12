"""Reusable Qwen4-Exp serving export orchestration."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_artifact.files import file_sha256 as _sha256_file
from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import (
    extract_qwen4_exp_text_config,
    extract_qwen4_exp_text_model,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.serving_export import (
    build_qwen4_exp_serving_metadata,
)

from .cache_contract import QWEN4_EXP_LLM_CACHE_ABI, QWEN4_EXP_PAGE_CACHE_ABI, resolve_qwen4_exp_cache_abi
from .checkpoint import load_qwen4_exp_model_for_export
from .mtp_contract import mtp_verify_phase, resolve_mtp_num_draft_tokens, resolve_mtp_verify_length
from .precision import DEFAULT_PLE_QUANT_TYPE, resolve_ple_quant_type
from .serving_graph import convert_qwen4_exp_kv_to_llm_cache, convert_qwen4_exp_kv_to_page_cache
from .static_export import build_qwen4_exp_serving_sample_inputs, export_qwen4_exp_hmonnx


def _graph_summary(path: str | Path) -> dict[str, int]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _validate_page_graph(
    path: str | Path,
    *,
    expected_attention_layers: int,
    expected_sequence_length: int | None = None,
    expected_hc_width: int | None = None,
    expected_recurrent_layers: int | None = None,
    require_fused_recurrent: bool = True,
    cache_abi: str = QWEN4_EXP_PAGE_CACHE_ABI,
) -> dict[str, int]:
    import onnx

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    onnx.checker.check_model(str(path))
    graph = onnx.load(str(path), load_external_data=False).graph
    for name, values, width in (
        ("inputs_embeds", graph.input, None),
        ("target_hidden_states", graph.output, expected_hc_width),
    ):
        if name == "target_hidden_states" and expected_hc_width is None:
            continue
        if expected_sequence_length is None:
            continue
        info = next((value for value in values if value.name == name), None)
        shape = [] if info is None else [dim.dim_value for dim in info.type.tensor_type.shape.dim]
        if len(shape) != 3 or shape[:2] != [1, expected_sequence_length] or (width is not None and shape[2] != width):
            raise ValueError(
                f"{path} has incompatible {name} shape {shape}; expected sequence={expected_sequence_length}"
            )
    counts = _graph_summary(path)
    cache_abi = resolve_qwen4_exp_cache_abi(cache_abi)
    cache_count = sum(
        node.op_type == "KVcache"
        and len(node.input) >= 4
        and node.input[3].startswith(("past_key_cache_", "past_value_cache_"))
        for node in graph.node
    )
    if cache_abi == QWEN4_EXP_LLM_CACHE_ABI:
        if any(
            value.name.startswith(
                ("present_past_key_cache_", "present_past_value_cache_", "present_qsa_indexer_key_cache_")
            )
            for value in graph.output
        ):
            raise RuntimeError(f"{path} exposes in-place K/V cache outputs; re-export the serving graph")
        if counts.get("LLMCache", 0):
            raise RuntimeError(f"{path} uses legacy LLMCache node names; re-export with KVcache nodes")
        if counts.get("PageKVCache", 0) or cache_count != 2 * expected_attention_layers:
            raise RuntimeError(f"{path} explicit KVcache writer count mismatch: counts={counts}")
    required_ops = ("PageKVCache",) if cache_abi == QWEN4_EXP_PAGE_CACHE_ABI else ()
    required_ops += ("SparseAttentionSelectKV",)
    for op_type in required_ops:
        if counts.get(op_type, 0) != expected_attention_layers:
            raise RuntimeError(
                f"{path} {op_type} count mismatch: expected={expected_attention_layers}, counts={counts}"
            )
    if counts.get("FlashAttention", 0):
        raise RuntimeError(f"{path} selected-KV attention still contains FlashAttention: counts={counts}")
    if cache_abi == QWEN4_EXP_PAGE_CACHE_ABI and counts.get("LLMCache", 0):
        raise RuntimeError(f"{path} still contains LLMCache nodes: counts={counts}")
    if expected_recurrent_layers is not None:
        expected_scans = expected_recurrent_layers * int(expected_sequence_length or 1)
        if (
            (require_fused_recurrent and counts.get("GDRRecurrentScan", 0) != expected_scans)
            or counts.get("GDRChunkScan", 0)
            or counts.get("GDRBlockTriInverse", 0)
        ):
            raise RuntimeError(f"{path} verify must use {expected_scans} recurrent token updates: counts={counts}")
        state_outputs = [
            value.name for value in graph.output if value.name.startswith("present_delta_recurrent_state_")
        ]
        if len(state_outputs) != expected_recurrent_layers:
            raise RuntimeError(f"{path} verify must return all {expected_recurrent_layers} recurrent states")
    return counts


def _write_manifest(path: Path, data: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _source_tensor_fingerprints(source, prefix):
    """Hash selected SafeTensor byte ranges, never unrelated target/PLE data."""
    source = Path(source).resolve()
    index = source / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    files = sorted(set(mapping.values())) if mapping is not None else [p.name for p in source.glob("*.safetensors")]
    headers = {}
    for filename in files:
        # Checkpoints (unlike published artifacts) may use HF snapshot symlinks
        # to the user's blob cache, as the native checkpoint loader does.
        path = source / filename
        with path.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 0 < size <= path.stat().st_size - 8:
                raise ValueError(f"Invalid SafeTensor header: {path}")
            header = json.loads(stream.read(size))
        headers[filename] = (header, size + 8)
    names = {
        name: filename
        for filename, (header, _) in headers.items()
        for name in header
        if name != "__metadata__" and (mapping is None or mapping.get(name) == filename)
    }
    selected = [name for name in names if name.startswith(prefix)]
    if not selected and prefix == "lm_head.":
        config = json.loads((source / "config.json").read_text())
        text = config.get("text_config", config)
        if text.get("tie_word_embeddings", config.get("tie_word_embeddings", False)):
            selected = [
                name
                for name in names
                if name
                in {
                    "model.embed_tokens.weight",
                    "model.language_model.embed_tokens.weight",
                }
            ]
    if not selected:
        raise ValueError(f"No source tensors for resumable {prefix} export in {source}")
    result = {}
    for name in sorted(selected):
        filename = names[name]
        header, start = headers[filename]
        item = header[name]
        offset, end = item["data_offsets"]
        path = source / filename
        if not 0 <= offset <= end <= path.stat().st_size - start:
            raise ValueError(f"Invalid source tensor byte range: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            stream.seek(start + offset)
            remaining = end - offset
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"Truncated source tensor: {name}")
                digest.update(chunk)
                remaining -= len(chunk)
        result[name] = {"shard": filename, "dtype": item["dtype"], "shape": item["shape"], "sha256": digest.hexdigest()}
    return result


def _draft_resource_fingerprints(root, graph_path):
    import onnx

    root, graph_path = Path(root).resolve(), Path(graph_path)
    graph = confined_file(root, graph_path.relative_to(root).as_posix(), "resume.draft_graph")
    model = onnx.load(graph, load_external_data=False)
    paths = {graph.relative_to(root).as_posix()}
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        for entry in tensor.external_data:
            if entry.key == "location":
                paths.add((graph.parent.relative_to(root) / entry.value).as_posix())
    return {
        relative: _sha256_file(confined_file(root, relative, "resume.draft_resource")) for relative in sorted(paths)
    }


def _fixed_head_bindings(model_dir, model_config, mtp_source_dir=None):
    source = Path(model_dir).resolve()
    assistant_source = Path(mtp_source_dir or model_dir).resolve()
    return {
        "source_model": str(source),
        "source_config_sha256": _sha256_file(source / "config.json"),
        "source_head_tensors": _source_tensor_fingerprints(source, "lm_head."),
        "mtp_source_model": str(assistant_source),
        "mtp_source_config_sha256": _sha256_file(assistant_source / "config.json"),
        "mtp_source_tensors": _source_tensor_fingerprints(assistant_source, "mtp."),
        "export_config_sha256": hashlib.sha256(
            json.dumps(model_config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _save_fixed_head_checkpoint(root, head, bindings):
    # SDK dynamic module classes are not pickle globals. Their nn.Module state
    # is serializable and retains the stable original class used by the registry.
    if not getattr(head, "is_quant_fixed", False) or head.qweight is None:
        raise ValueError("A fixed target head is required for resumable MTP export")
    path = Path(root) / ".qwen4_fixed_head.pt"
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(head.__getstate__(), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": path.name, "sha256": _sha256_file(path), "bindings": bindings}


def _load_fixed_head_checkpoint(root, record, bindings):
    from xhquant.quantization.xh2a.builder import XH2A_QMODULES

    if not isinstance(record, dict) or record.get("bindings") != bindings:
        raise ValueError("resume requires a fixed head checkpoint bound to the same source and export configuration")
    path = Path(root) / ".qwen4_fixed_head.pt"
    if (
        record.get("path") != path.name
        or path.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != record.get("sha256")
    ):
        raise ValueError("resume fixed head checkpoint is missing or its SHA256 differs")
    state = torch.load(path, map_location="cpu", weights_only=False)
    cls = XH2A_QMODULES[state["_orig_class"]]
    head = cls.__new__(cls)
    head.__setstate__(state)
    if not getattr(head, "is_quant_fixed", False) or head.qweight is None:
        raise ValueError("resume checkpoint does not contain a fixed target head")
    return head


def export_qwen4_exp_serving_package(
    *,
    model_dir: str | Path,
    output_dir: str | Path,
    model_config: dict,
    device: str,
    device_map=None,
    model_name: str | None = None,
    mtp_source_dir: str | Path | None = None,
    phase: str = "both",
    resume: bool = False,
    low_memory: bool | None = None,
) -> dict:
    """Export a serving package without running inference or generating Golden data."""
    from types import SimpleNamespace

    from ...utils import HUGE_MODEL_EXPORT_ENABLED_ENV, is_huge_model_export_enabled

    if low_memory is not None and not isinstance(low_memory, bool):
        raise ValueError("low_memory must be a boolean or None")
    if low_memory is None:
        low_memory = is_huge_model_export_enabled(os.environ.get(HUGE_MODEL_EXPORT_ENABLED_ENV, "1"))
    export_mode = "low_memory" if low_memory else "standard"

    ple_quant_type = resolve_ple_quant_type(model_config.get("ple_quant_type", DEFAULT_PLE_QUANT_TYPE))
    if low_memory and ple_quant_type != "sefp4":
        raise ValueError("Qwen4 low-memory export requires Host PLE SEFP4; use standard export for FP16 PLE")
    cache_abi = resolve_qwen4_exp_cache_abi(model_config.get("cache_abi", QWEN4_EXP_LLM_CACHE_ABI))
    enable_mtp = model_config.get("spec_decode_mode") == "mtp"
    num_draft_tokens = resolve_mtp_num_draft_tokens(model_config) if enable_mtp else None
    verify_length = resolve_mtp_verify_length(model_config) if enable_mtp else None
    verify_phase = mtp_verify_phase(verify_length) if enable_mtp else None
    allowed_phases = {"prefill", "decode", "both"}
    if enable_mtp:
        allowed_phases.add(verify_phase)
    if phase not in allowed_phases:
        raise ValueError(f"Unknown serving export phase: {phase}")
    if int(model_config.get("batch_size", 1)) != 1:
        raise ValueError("Qwen4-Exp serving export supports batch size 1")
    context_length = int(model_config.get("context_max_length", 262144))
    prefill_length = int(model_config.get("prefill_chunk_length", 256))
    if context_length < 4 or not 1 <= prefill_length <= context_length:
        raise ValueError("Require 4 <= context_max_length and 1 <= prefill_chunk_length <= context_max_length")
    if enable_mtp and context_length < verify_length:
        raise ValueError(f"Qwen4 MTP requires context_max_length >= verify length {verify_length}")
    if not enable_mtp and phase == "both" and not resume and cache_abi == QWEN4_EXP_LLM_CACHE_ABI:
        from xhmodel_merak._environment import scoped_environment

        from .configuration_qwen4_exp import XHQwen4ExpModelConfig
        from .qwen4_exp_model import XHQwen4ExpModel

        root = Path(output_dir).expanduser().resolve()
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise FileExistsError(f"refusing to overwrite existing output: {root}")
        config = dict(model_config)
        config.update(hf_model=str(model_dir), model_name=model_name or config.get("model_name", "qwen3_8_flash_next"))
        config.pop("export_mode", None)
        if device_map is not None:
            config["device_map"] = device_map
        model_entry = XHQwen4ExpModel(XHQwen4ExpModelConfig(**config))
        model_entry.to(device=device)
        with scoped_environment({HUGE_MODEL_EXPORT_ENABLED_ENV: "1" if low_memory else "0"}):
            model_entry.export_hmonnx(str(root))
        meta = json.loads((root / "golden_meta_info.json").read_text())
        manifest = {
            "status": "complete",
            "export_mode": export_mode,
            "source_model": str(Path(model_dir).resolve()),
            "output_dir": str(root),
            "meta": str(root / "golden_meta_info.json"),
            "context_length": context_length,
            "prefill_chunk_length": prefill_length,
            "ple_quant_type": ple_quant_type,
            "cache_abi": cache_abi,
            "graphs": {name: {"page_hmonnx": str(root / meta[name + "_hmonnx"])} for name in ("prefill", "decode")},
        }
        _write_manifest(root / "export_manifest.json", manifest)
        return manifest

    args = SimpleNamespace(
        model_dir=model_dir,
        output_dir=output_dir,
        device=device,
        device_map=device_map,
        model_name=model_name or model_config.get("model_name", "qwen3_8_flash_next"),
        context_length=context_length,
        prefill_chunk_length=prefill_length,
        phase=phase,
        resume=resume,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not args.resume and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest_path = output_dir / "export_manifest.json"
    previous_manifest: dict = {}
    if args.resume and manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if enable_mtp and previous_manifest.get("verify_length", 2) != verify_length:
            raise ValueError("resume verify length does not match the existing export")
        if previous_manifest.get("export_mode", "standard") != export_mode:
            raise ValueError("resume export mode does not match the existing export")
        if previous_manifest.get("ple_quant_type", "fp16") != ple_quant_type:
            raise ValueError("resume PLE precision does not match the existing export")
        if resolve_qwen4_exp_cache_abi(previous_manifest.get("cache_abi")) != cache_abi:
            raise ValueError("resume cache ABI does not match the existing export")
        if int(previous_manifest.get("context_length", -1)) != int(args.context_length):
            raise ValueError("resume context length does not match the existing export")
        if int(previous_manifest.get("prefill_chunk_length", -1)) != int(args.prefill_chunk_length):
            raise ValueError("resume prefill chunk length does not match the existing export")
        if Path(previous_manifest.get("source_model", "")).resolve() != Path(args.model_dir).resolve():
            raise ValueError("resume model directory does not match the existing export")

    export_config = dict(model_config)
    export_config.update(
        {
            "batch_size": 1,
            "context_max_length": int(args.context_length),
            "prefill_chunk_length": int(args.prefill_chunk_length),
            **({"num_draft_tokens": num_draft_tokens, "verify_length": verify_length} if enable_mtp else {}),
            "max_pe_length": max(int(export_config.get("max_pe_length", 0)), int(args.context_length)),
            "use_cache": True,
            "ple_quant_type": ple_quant_type,
            "cache_abi": cache_abi,
        }
    )
    if enable_mtp:
        # The assistant consumes every target HC stream before the final mixer.
        export_config["return_hidden_states"] = True
    quant_config = export_config.get("quant_scheme")
    mtp_quant_config = None
    if enable_mtp:
        from ...shared_head import inherit_head_config, resolve_shared_head_bits
        from .mtp_export import default_mtp_quant_scheme

        mtp_config = export_config.get("mtp_config") or {}
        head_bits = resolve_shared_head_bits(quant_config, None, mtp_config)
        mtp_config = inherit_head_config(mtp_config, head_bits)
        mtp_quant_config = mtp_config.get("quant_scheme") or default_mtp_quant_scheme(head_bits)
    head_bindings = _fixed_head_bindings(args.model_dir, export_config, mtp_source_dir) if enable_mtp else None
    fixed_head_record = previous_manifest.get("fixed_quantized_head")
    fixed_quantized_head = None
    if enable_mtp and args.resume and previous_manifest.get("graphs"):
        fixed_quantized_head = _load_fixed_head_checkpoint(output_dir, fixed_head_record, head_bindings)
    draft_records = dict(previous_manifest.get("mtp_graphs", {}))
    if enable_mtp and args.resume:
        for name in ("prefill", "decode"):
            path = output_dir / f"mtp_{name}" / "model.onnx"
            record = draft_records.get(name)
            if path.exists() or record is not None:
                if (
                    not isinstance(record, dict)
                    or record.get("bindings") != head_bindings
                    or record.get("resources") != _draft_resource_fingerprints(output_dir, path)
                ):
                    raise ValueError(f"resume MTP {name} requires a matching completed graph/resource record")
    loader = None
    if low_memory:
        from .streaming_loader import Qwen4ExpStreamingLoader

        loader = Qwen4ExpStreamingLoader(args.model_dir)
        model = loader.model
    else:
        model = load_qwen4_exp_model_for_export(
            args.model_dir,
            device=args.device,
            device_map=args.device_map,
        )
    try:
        # Loading must be sharded because the PLE table is larger than one GPU,
        # while xhquant's frontend restores a traced graph to its first parameter's
        # device. Keep the export graph CPU-resident and use ``args.device`` only
        # for module-at-a-time PTQ to avoid an accidental whole-model GPU move.
        if not low_memory:
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(model, recurse=True)
            model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        text_model = extract_qwen4_exp_text_model(model)
        text_config = extract_qwen4_exp_text_config(model)
        graph_manifest: dict[str, dict] = dict(previous_manifest.get("graphs", {}))
        expected_page_layers = sum(
            str(layer_type) in {"qwen_sparse_attention", "full_attention"} for layer_type in text_config.layer_types
        )
        requested_phases = ("prefill", "decode") if args.phase == "both" else (args.phase,)
        if enable_mtp and args.phase in {"decode", "both"}:
            requested_phases += (verify_phase,)
        phase_lengths = {"prefill": int(args.prefill_chunk_length), "decode": 1}
        if enable_mtp:
            phase_lengths[verify_phase] = verify_length

        for phase in requested_phases:
            sequence_length = phase_lengths[phase]
            previous_phase = graph_manifest.get(phase)
            if args.resume and previous_phase is not None:
                counts = _validate_page_graph(
                    previous_phase["page_hmonnx"],
                    expected_attention_layers=expected_page_layers,
                    cache_abi=cache_abi,
                    expected_sequence_length=sequence_length,
                    expected_hc_width=int(text_config.hidden_size * text_config.hc_count) if enable_mtp else None,
                    expected_recurrent_layers=(
                        sum(kind == "linear_attention" for kind in text_config.layer_types)
                        if phase == verify_phase
                        else None
                    ),
                    require_fused_recurrent=bool(export_config.get("use_fused_gdr", False)),
                )
                previous_phase["op_counts"] = counts
                previous_phase["resume_validation"] = "passed"
                print(f"reusing validated {phase} artifact: {previous_phase['page_hmonnx']}", flush=True)
                continue

            phase_started = time.time()
            raw_dir = output_dir / f"{phase}_raw"
            phase_options = dict(
                quant_config=quant_config,
                ple_quant_type=ple_quant_type,
                input_sequence_length=sequence_length,
                cache_length=int(args.context_length),
                quant_device=args.device,
                use_fused_gdr=bool(export_config.get("use_fused_gdr", False)),
                linear_attention_mode="recurrent" if phase == verify_phase else None,
                return_hidden_states=bool(export_config.get("return_hidden_states", False)),
            )
            if enable_mtp:
                phase_options.update(
                    shared_quantized_head=fixed_quantized_head,
                    capture_quantized_head=True,
                )
            if low_memory:
                from .streaming_export import export_qwen4_exp_streamed_phase

                result = export_qwen4_exp_streamed_phase(loader, raw_dir, **phase_options)
            else:
                _, _, sample_inputs = build_qwen4_exp_serving_sample_inputs(
                    model,
                    input_sequence_length=sequence_length,
                    context_length=int(args.context_length),
                    text_config=text_config,
                )
                result = export_qwen4_exp_hmonnx(
                    model,
                    raw_dir,
                    text_config=text_config,
                    sample_inputs=sample_inputs,
                    batch_size=1,
                    **phase_options,
                )
            if enable_mtp and fixed_quantized_head is None:
                if _fixed_head_bindings(args.model_dir, export_config, mtp_source_dir) != head_bindings:
                    raise ValueError("MTP source tensors changed during target head export")
                fixed_quantized_head = result.fixed_quantized_head
                fixed_head_record = _save_fixed_head_checkpoint(output_dir, fixed_quantized_head, head_bindings)
            page_dir = output_dir / phase
            suffix = "llm_cache" if cache_abi == QWEN4_EXP_LLM_CACHE_ABI else "page_kv"
            page_path = page_dir / f"qwen4_exp_{phase}_{suffix}.onnx"
            converter = (
                convert_qwen4_exp_kv_to_llm_cache
                if cache_abi == QWEN4_EXP_LLM_CACHE_ABI
                else convert_qwen4_exp_kv_to_page_cache
            )
            converter(
                result.hmonnx_path,
                page_path,
                context_length=int(args.context_length),
                num_kv_heads=int(text_config.num_key_value_heads),
                head_dim=int(text_config.head_dim),
            )
            counts = _validate_page_graph(
                page_path,
                expected_attention_layers=expected_page_layers,
                cache_abi=cache_abi,
                expected_sequence_length=sequence_length,
                expected_hc_width=int(text_config.hidden_size * text_config.hc_count) if enable_mtp else None,
                expected_recurrent_layers=(
                    sum(kind == "linear_attention" for kind in text_config.layer_types)
                    if phase == verify_phase
                    else None
                ),
                require_fused_recurrent=bool(export_config.get("use_fused_gdr", False)),
            )
            graph_manifest[phase] = {
                "raw_hmonnx": result.hmonnx_path,
                "page_hmonnx": str(page_path),
                "op_counts": counts,
                "elapsed_seconds": time.time() - phase_started,
                "input_sequence_length": sequence_length,
            }
            partial_manifest = {
                "status": "partial",
                "export_mode": export_mode,
                "source_model": str(Path(args.model_dir).resolve()),
                "output_dir": str(output_dir),
                "context_length": int(args.context_length),
                "prefill_chunk_length": int(args.prefill_chunk_length),
                **(
                    {
                        "num_draft_tokens": num_draft_tokens,
                        "verify_length": verify_length,
                        "fixed_quantized_head": fixed_head_record,
                        "mtp_graphs": draft_records,
                    }
                    if enable_mtp
                    else {}
                ),
                "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
                "graphs": graph_manifest,
                "ple_quant_type": ple_quant_type,
                "cache_abi": cache_abi,
                "elapsed_seconds": time.time() - started,
            }
            _write_manifest(manifest_path, partial_manifest)
            if not low_memory:
                del sample_inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        required_phases = {"prefill", "decode", verify_phase} if enable_mtp else {"prefill", "decode"}
        if not required_phases.issubset(graph_manifest):
            partial_manifest = {
                "status": "partial",
                "export_mode": export_mode,
                "source_model": str(Path(args.model_dir).resolve()),
                "output_dir": str(output_dir),
                "context_length": int(args.context_length),
                "prefill_chunk_length": int(args.prefill_chunk_length),
                **(
                    {
                        "num_draft_tokens": num_draft_tokens,
                        "verify_length": verify_length,
                        "fixed_quantized_head": fixed_head_record,
                    }
                    if enable_mtp
                    else {}
                ),
                "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
                "graphs": graph_manifest,
                "ple_quant_type": ple_quant_type,
                "cache_abi": cache_abi,
                "elapsed_seconds": time.time() - started,
            }
            _write_manifest(manifest_path, partial_manifest)
            return partial_manifest

        for phase in sorted(required_phases):
            graph_manifest[phase]["op_counts"] = _validate_page_graph(
                graph_manifest[phase]["page_hmonnx"],
                expected_attention_layers=expected_page_layers,
                cache_abi=cache_abi,
                expected_sequence_length=phase_lengths[phase],
                expected_hc_width=int(text_config.hidden_size * text_config.hc_count) if enable_mtp else None,
                expected_recurrent_layers=(
                    sum(kind == "linear_attention" for kind in text_config.layer_types)
                    if phase == verify_phase
                    else None
                ),
                require_fused_recurrent=bool(export_config.get("use_fused_gdr", False)),
            )

        visual_metadata = None
        visual_config = export_config.get("visual_config")
        if isinstance(visual_config, dict) and visual_config.get("enable", True):
            from .vision import export_qwen4_exp_visual

            visual_metadata = export_qwen4_exp_visual(
                args.model_dir,
                output_dir,
                image_token_gears=visual_config["image_token_gears"],
                quant_scheme=visual_config["quant_scheme"],
                device=args.device,
            )
        mtp_metadata = None
        if enable_mtp:
            from .mtp_export import _mtp_export_metadata, export_qwen4_exp_mtp

            mtp_metadata = {}
            target_head = getattr(model, "lm_head", None)
            head_owner = None
            try:
                if low_memory and any(
                    not (args.resume and (output_dir / f"mtp_{name}" / "model.onnx").is_file())
                    for name in ("prefill", "decode")
                ):
                    from .checkpoint import _adapt_gptq_linears

                    head_owner = torch.nn.Module()
                    head_owner.lm_head = loader.load_module("lm_head")
                    _adapt_gptq_linears(head_owner)
                    target_head = head_owner.lm_head
                for draft_phase, sequence_length in (("prefill", int(args.prefill_chunk_length)), ("decode", 1)):
                    published = output_dir / f"mtp_{draft_phase}" / "model.onnx"
                    if args.resume and published.is_file():
                        import onnx

                        from .cache_contract import Qwen4ExpCacheContract
                        from .mtp_model import mtp_text_config

                        source = Path(mtp_source_dir or args.model_dir)
                        source_config = json.loads((source / "config.json").read_text())
                        config = mtp_text_config(source_config.get("text_config", source_config))
                        _validate_page_graph(published, expected_attention_layers=1, cache_abi=cache_abi)
                        graph = onnx.load(published, load_external_data=False).graph
                        ports = {
                            value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
                            for value in (*graph.input, *graph.output)
                        }
                        expected = {
                            "next_token_embedding": [1, sequence_length, config.hidden_size],
                            "target_hidden": [1, sequence_length, config.hidden_size * config.hc_count],
                            "logits": [1, sequence_length, config.vocab_size],
                            "mtp_hidden_states": [1, sequence_length, config.hidden_size * config.hc_count],
                        }
                        if any(ports.get(name) != shape for name, shape in expected.items()):
                            raise ValueError(f"resume MTP {draft_phase} graph has incompatible tensor shapes")
                        mtp_metadata[draft_phase] = _mtp_export_metadata(
                            published,
                            cache_contract=Qwen4ExpCacheContract.from_text_config(config).to_dict(),
                            cache_abi=cache_abi,
                            hidden_size=config.hidden_size,
                            hc_count=config.hc_count,
                            input_sequence_length=sequence_length,
                        )
                        continue
                    mtp_metadata[draft_phase] = export_qwen4_exp_mtp(
                        mtp_source_dir or args.model_dir,
                        output_dir / f"mtp_{draft_phase}",
                        input_sequence_length=sequence_length,
                        cache_length=int(args.context_length),
                        cache_abi=cache_abi,
                        quant_config=mtp_quant_config,
                        device=args.device,
                        dtype=torch.float16,
                        lm_head=target_head,
                        shared_quantized_head=fixed_quantized_head,
                    )
                    import onnx

                    onnx.checker.check_model(str(published))
                    if _fixed_head_bindings(args.model_dir, export_config, mtp_source_dir) != head_bindings:
                        raise ValueError("MTP source tensors changed during draft export")
                    draft_records[draft_phase] = {
                        "bindings": head_bindings,
                        "resources": _draft_resource_fingerprints(output_dir, published),
                    }
                    progress_manifest = json.loads(manifest_path.read_text())
                    progress_manifest.update(status="partial", mtp_graphs=draft_records)
                    _write_manifest(manifest_path, progress_manifest)
            finally:
                if head_owner is not None:
                    loader.release(head_owner)
        if low_memory:
            prefix = next(name for name, module in model.named_modules() if module is text_model)
            embedding = loader.read_tensor(f"{prefix}.embed_tokens.weight").to(loader.dtype)
        else:
            embedding = text_model.embed_tokens.weight.detach().cpu()
        if embedding.ndim != 2 or not embedding.numel():
            raise ValueError("Serving export requires a nonempty 2-D token embedding")
        meta_path = build_qwen4_exp_serving_metadata(
            output_dir=output_dir,
            model_name=args.model_name,
            hf_model_dir=args.model_dir,
            text_config=text_config,
            model_config=export_config,
            prefill_hmonnx=graph_manifest["prefill"]["page_hmonnx"],
            decode_hmonnx=graph_manifest["decode"]["page_hmonnx"],
            embedding_state=embedding,
            pad_token_id=getattr(text_config, "pad_token_id", None),
            enable_page_attention=cache_abi == QWEN4_EXP_PAGE_CACHE_ABI,
            visual_metadata=visual_metadata,
            mtp_metadata=mtp_metadata,
            verify_hmonnx=graph_manifest[verify_phase]["page_hmonnx"] if enable_mtp else None,
        )
        manifest = {
            "status": "complete",
            "export_mode": export_mode,
            "source_model": str(Path(args.model_dir).resolve()),
            "output_dir": str(output_dir),
            "meta": str(meta_path),
            "context_length": int(args.context_length),
            "prefill_chunk_length": int(args.prefill_chunk_length),
            **(
                {
                    "num_draft_tokens": num_draft_tokens,
                    "verify_length": verify_length,
                    "fixed_quantized_head": fixed_head_record,
                }
                if enable_mtp
                else {}
            ),
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
            "graphs": graph_manifest,
            **({"mtp_graphs": draft_records} if enable_mtp else {}),
            "ple_quant_type": ple_quant_type,
            "cache_abi": cache_abi,
            "elapsed_seconds": time.time() - started,
        }
        _write_manifest(manifest_path, manifest)
        return manifest

    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
