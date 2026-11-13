"""Package four-layer Qwen4-Exp HMONNX exports in Merak Golden layout."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import torch


_PHASES = ("prefill", "decode")
_HF_CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
)
_SELECTED_KV_TE_ATTRS = (
    "mode",
    "hmfp_act_exp_bit",
    "hmfp_act_man_bit",
    "hmfp_act_nshare",
    "hmfp_act_rounding",
    "hmfp_act_hidden_bit",
    "hmfp_weight_exp_bit",
    "hmfp_weight_man_bit",
    "hmfp_weight_nshare",
    "hmfp_weight_rounding",
    "hmfp_weight_hidden_bit",
    "hmfp_psum_dtype",
    "hmfp_psum_round_mode",
    "output_dtype",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _assert_same_or_missing(source: Path, target: Path) -> None:
    if not target.exists():
        return
    if source.resolve() == target.resolve():
        return
    try:
        if os.path.samefile(source, target):
            return
    except FileNotFoundError:
        pass
    if source.stat().st_size == target.stat().st_size and _md5(source) == _md5(target):
        return
    raise FileExistsError(f"refusing to overwrite different artifact: {target}")


def _link_or_keep(source: Path, target: Path) -> None:
    _assert_same_or_missing(source, target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def _replace_file_with_same_guard(source: Path, target: Path) -> None:
    _assert_same_or_missing(source, target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object: {path}")
    return data


def _release_prefix(manifest: Mapping[str, Any]) -> str:
    if manifest.get("release_prefix"):
        prefix = str(manifest["release_prefix"])
        if Path(prefix).name != prefix or prefix != prefix.lower():
            raise ValueError(f"invalid release_prefix: {prefix!r}")
        return prefix
    context = int(manifest.get("context_length", 262144))
    prefill = int(manifest.get("prefill_chunk_length", 256))
    return f"hmquant_xh2_qwen3_8_flash_next_4layer_selectedkv_gdr_w4a8_{prefill}_{context}_20261010"


def _require_complete_manifest(source_dir: Path) -> dict[str, Any]:
    manifest_path = source_dir / "export_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing export_manifest.json: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"source export_manifest status must be 'complete', got {manifest.get('status')!r}")
    graphs = manifest.get("graphs")
    if not isinstance(graphs, dict):
        raise ValueError("source export_manifest must contain graphs")
    for phase in _PHASES:
        graph = graphs.get(phase)
        if not isinstance(graph, dict):
            raise ValueError(f"source export_manifest missing graphs.{phase}")
        hmonnx = graph.get("hmonnx")
        if not isinstance(hmonnx, str) or not hmonnx:
            raise ValueError(f"source export_manifest missing graphs.{phase}.hmonnx")
        op_counts = graph.get("op_counts")
        if not isinstance(op_counts, dict) or not op_counts:
            raise ValueError(f"source export_manifest missing graphs.{phase}.op_counts")
        if int(op_counts.get("SparseAttentionSelectKV", 0)) <= 0:
            raise ValueError(f"graphs.{phase}.op_counts must include SparseAttentionSelectKV")
    return manifest


def _onnx_attr_value(attribute: Any) -> Any:
    from onnx import helper

    value = helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [item.decode("utf-8") if isinstance(item, bytes) else item for item in value]
    return value


def _selected_kv_proof(onnx_path: Path, *, expected_score_mode: str | None) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    nodes: list[dict[str, Any]] = []
    for node in model.graph.node:
        if node.op_type != "SparseAttentionSelectKV":
            continue
        attrs = {attribute.name: _onnx_attr_value(attribute) for attribute in node.attribute}
        if any(name in attrs for name in ("q_bit", "k_bit", "compression_ratio")):
            required = ("q_bit", "k_bit", "compression_ratio")
            missing = [name for name in required if name not in attrs]
            if missing or any(attrs.get(name) not in (8, 16) for name in ("q_bit", "k_bit")):
                raise ValueError(f"Invalid SelectedKV precision attributes in {onnx_path}: {attrs}")
            if attrs["compression_ratio"] != 4 or attrs.get("pool_size", 1) != 4:
                raise ValueError(f"Qwen SelectedKV requires compression_ratio=4 and pool_size=4: {onnx_path}")
            if ("score_mode" in attrs or len(node.input) not in (6, 7)
                    or any(not node.input[index] for index in (0, 1, 2, 3, 5))):
                raise ValueError(f"SelectedKV requires raw history and explicit current_length: {onnx_path}")
            separate_kv = len(node.input) == 7
            if (len(node.output) != (2 if separate_kv else 1) or not all(node.output)
                    or (separate_kv and not node.input[6])):
                raise ValueError(f"SelectedKV requires paired cache inputs and outputs: {onnx_path}")
            cache_ports = (
                {"key_cache_input": node.input[2], "value_cache_input": node.input[6],
                 "selected_key_output": node.output[0], "selected_value_output": node.output[1]}
                if separate_kv else
                {"payload_cache_input": node.input[2], "selected_kv_output": node.output[0]}
            )
            nodes.append(
                {
                    "name": node.name,
                    "domain": node.domain,
                    "abi": "raw_history_current_length_gqa" if separate_kv else "raw_history_current_length",
                    **cache_ports,
                    "score_mode": "te",
                    "execution_contract": "fixed TE MatMul in current xhquant backend",
                    "q_bit": attrs["q_bit"],
                    "k_bit": attrs["k_bit"],
                    "compression_ratio": attrs["compression_ratio"],
                    "candidate_topk": attrs.get("candidate_topk"),
                    "pool_size": attrs.get("pool_size"),
                    "score_scale": attrs.get("score_scale"),
                    "valid_length_input": node.input[3],
                    "current_length_input": node.input[5],
                    "te_attrs_complete": True,
                    "missing_te_attrs": [],
                }
            )
            continue
        score_mode = str(attrs.get("score_mode", "vp")).lower()
        missing_te_attrs = [name for name in _SELECTED_KV_TE_ATTRS if name not in attrs]
        node_proof = {
            "name": node.name,
            "domain": node.domain,
            "score_mode": score_mode,
            "candidate_topk": attrs.get("candidate_topk"),
            "pool_size": attrs.get("pool_size"),
            "score_scale": attrs.get("score_scale"),
            "te_attrs_complete": not missing_te_attrs if score_mode == "te" else None,
            "missing_te_attrs": missing_te_attrs if score_mode == "te" else [],
            "q_bits": {
                "exp": attrs.get("hmfp_act_exp_bit"),
                "man": attrs.get("hmfp_act_man_bit"),
                "hidden": attrs.get("hmfp_act_hidden_bit"),
            },
            "k_bits": {
                "exp": attrs.get("hmfp_weight_exp_bit"),
                "man": attrs.get("hmfp_weight_man_bit"),
                "hidden": attrs.get("hmfp_weight_hidden_bit"),
            },
            "nshare": {
                "query": attrs.get("hmfp_act_nshare"),
                "key": attrs.get("hmfp_weight_nshare"),
            },
            "rounding": {
                "query": attrs.get("hmfp_act_rounding"),
                "key": attrs.get("hmfp_weight_rounding"),
            },
            "psum_rule": {
                "dtype": attrs.get("hmfp_psum_dtype"),
                "round_mode": attrs.get("hmfp_psum_round_mode"),
            },
            "output_dtype": attrs.get("output_dtype"),
            "te_attrs": {name: attrs[name] for name in _SELECTED_KV_TE_ATTRS if name in attrs},
        }
        nodes.append(node_proof)

    proof = {
        "requested_score_mode": expected_score_mode,
        "node_count": len(nodes),
        "nodes": nodes,
    }
    if expected_score_mode == "te":
        if not nodes:
            raise ValueError(f"TE SelectedKV package requires SparseAttentionSelectKV nodes: {onnx_path}")
        bad_mode = [node for node in nodes if node["score_mode"] != "te"]
        if bad_mode:
            raise ValueError(f"TE SelectedKV package found non-TE SparseAttentionSelectKV nodes: {onnx_path}")
        incomplete = [node for node in nodes if not node["te_attrs_complete"]]
        if incomplete:
            missing = {node["name"] or f"node_{idx}": node["missing_te_attrs"] for idx, node in enumerate(incomplete)}
            raise ValueError(f"TE SparseAttentionSelectKV missing HMFP attrs in {onnx_path}: {missing}")
    return proof


def _find_external_locations(onnx_path: Path) -> list[str]:
    import onnx
    from onnx import TensorProto, external_data_helper

    model = onnx.load(str(onnx_path), load_external_data=False)
    locations: list[str] = []
    for tensor in external_data_helper._get_all_tensors(model):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value and entry.value not in locations:
                locations.append(entry.value)
    return locations


def _package_phase(
    source_dir: Path,
    output_dir: Path,
    release_prefix: str,
    manifest: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    graph = manifest["graphs"][phase]
    source_onnx = Path(graph["hmonnx"])
    if not source_onnx.is_absolute():
        source_onnx = source_dir / source_onnx
    source_onnx = source_onnx.resolve(strict=True)
    phase_dir = output_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    target_onnx = phase_dir / f"{release_prefix}_{phase}_with_act.onnx"

    locations = _find_external_locations(source_onnx)
    if len(locations) != 1:
        raise ValueError(f"expected exactly one external data file for {phase}, got {locations}")
    source_external = source_onnx.parent / locations[0]
    if not source_external.is_file():
        raise FileNotFoundError(f"missing source external data for {phase}: {source_external}")

    target_external = phase_dir / source_external.name
    _link_or_keep(source_external, target_external)
    _replace_file_with_same_guard(source_onnx, target_onnx)
    from .ple_export import ple_resource_manifest, read_qwen4_exp_ple_specs

    resource_root, _ = ple_resource_manifest(source_onnx)
    shared_manifest = resource_root / "ple" / "manifest.json"
    resource_target = output_dir if shared_manifest.is_file() else phase_dir
    for spec in read_qwen4_exp_ple_specs(source_onnx):
        for kind in ("packed", "exponent"):
            relative = spec[kind]["path"]
            target = resource_target / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _link_or_keep(resource_root / relative, target)
    if shared_manifest.is_file():
        _replace_file_with_same_guard(shared_manifest, output_dir / "ple" / "manifest.json")
    selected_kv_proof = _selected_kv_proof(
        target_onnx,
        expected_score_mode=str(manifest.get("selected_kv_score_mode", "")).lower() or None,
    )

    for name in ("ple.json", "qwen4_exp_hmonnx_io.json", "qwen4_exp_hmonnx_preflight.json"):
        source_sidecar = source_onnx.parent / name
        if source_sidecar.is_file():
            _replace_file_with_same_guard(source_sidecar, phase_dir / name)

    return {
        "hmonnx": _relative(output_dir, target_onnx),
        "external_data": _relative(output_dir, target_external),
        "source_hmonnx": str(source_onnx),
        "source_external_data": str(source_external),
        "op_counts": graph["op_counts"],
        "selected_kv_proof": selected_kv_proof,
        "io_manifest": _relative(output_dir, phase_dir / "qwen4_exp_hmonnx_io.json")
        if (phase_dir / "qwen4_exp_hmonnx_io.json").is_file()
        else None,
        "preflight": _relative(output_dir, phase_dir / "qwen4_exp_hmonnx_preflight.json")
        if (phase_dir / "qwen4_exp_hmonnx_preflight.json").is_file()
        else None,
    }


def _normalize_embedding_state(embedding_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "weight" not in embedding_state or not torch.is_tensor(embedding_state["weight"]):
        raise ValueError("embedding_state must contain a Tensor under key 'weight'")
    normalized: dict[str, torch.Tensor] = {}
    for name, value in embedding_state.items():
        if not torch.is_tensor(value):
            raise ValueError(f"embedding_state value must be a Tensor: {name}")
        normalized[str(name)] = value.detach().cpu()
    return normalized


def _embedding_states_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for name, left_tensor in left.items():
        right_value = right[name]
        if not torch.is_tensor(right_value):
            return False
        right_tensor = right_value.detach().cpu()
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            return False
        if not torch.equal(left_tensor, right_tensor):
            return False
    return True


def _load_existing_embedding_state(path: Path) -> Mapping[str, Any]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older torch does not expose weights_only.
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, Mapping):
        raise TypeError(f"existing quant embedding must be a mapping: {path}")
    return loaded


def _save_quant_embedding(output_dir: Path, embedding_state: Mapping[str, torch.Tensor]) -> str:
    normalized = _normalize_embedding_state(embedding_state)
    target = output_dir / "quant_embedding.pt"
    if target.exists():
        existing = _load_existing_embedding_state(target)
        if not _embedding_states_equal(normalized, existing):
            raise FileExistsError(f"refusing to overwrite different quant embedding: {target}")
        return _relative(output_dir, target)
    temporary = output_dir / ".quant_embedding.pt.tmp"
    torch.save(normalized, temporary)
    temporary.replace(target)
    return _relative(output_dir, target)


def _pad_token_id_from_hf_config(hf_model_dir: str | Path) -> int:
    source = Path(hf_model_dir).expanduser().resolve(strict=True)
    candidates: list[Any] = []
    config_path = source / "config.json"
    if config_path.is_file():
        config = _load_json(config_path)
        text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
        candidates.extend(
            (
                config.get("pad_token_id"),
                text_config.get("pad_token_id"),
                config.get("eos_token_id"),
                text_config.get("eos_token_id"),
            )
        )
    tokenizer_config_path = source / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        tokenizer_config = _load_json(tokenizer_config_path)
        candidates.extend((tokenizer_config.get("pad_token_id"), tokenizer_config.get("eos_token_id")))
    for value in candidates:
        if isinstance(value, int):
            return int(value)
        if isinstance(value, list) and value and isinstance(value[0], int):
            return int(value[0])
    return 248044


def _existing_package_meta_is_same_source(
    existing: Mapping[str, Any],
    phase_artifacts: Mapping[str, Mapping[str, Any]],
) -> bool:
    four_layer = existing.get("qwen4_exp_four_layer")
    if not isinstance(four_layer, Mapping):
        return False
    existing_phases = four_layer.get("phase_artifacts")
    if not isinstance(existing_phases, Mapping):
        return False
    for phase in _PHASES:
        existing_phase = existing_phases.get(phase)
        if not isinstance(existing_phase, Mapping):
            return False
        for key in ("hmonnx", "external_data", "source_hmonnx"):
            if existing_phase.get(key) != phase_artifacts[phase].get(key):
                return False
    return True


def _copy_hf_config(output_dir: Path, hf_model_dir: str | Path) -> str:
    source = Path(hf_model_dir).expanduser().resolve(strict=True)
    target_dir = output_dir / "hf_config"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in _HF_CONFIG_FILES:
        source_file = source / name
        if not source_file.is_file():
            continue
        _replace_file_with_same_guard(source_file, target_dir / name)
        copied += 1
    if copied == 0:
        raise FileNotFoundError(f"no HF config/tokenizer files found under {source}")
    return _relative(output_dir, target_dir)


def _copy_source_config(source_dir: Path, output_dir: Path) -> str | None:
    for name in ("export_source_config.yaml", "export_source_config.py", "export_source_config.json"):
        source = source_dir / name
        if source.is_file():
            _replace_file_with_same_guard(source, output_dir / name)
            return name
    return None


def _write_merak_config(output_dir: Path) -> str:
    config = {
        "architectures": ["MerakForCausalLM"],
        "config_format": "merak_llm",
        "load_format": "merak_llm",
        "xh_model": {"model_type": "hmonnx", "meta_info": "golden_meta_info.json"},
        "enable_page_attention": False,
        "model_type": "merak_llm",
        "max_num_seqs": 1,
        "disable_prefix_caching": True,
    }
    path = output_dir / "merak_config.json"
    new_text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != new_text:
        raise FileExistsError(f"refusing to overwrite different merak_config.json: {path}")
    path.write_text(new_text, encoding="utf-8")
    return _relative(output_dir, path)


def _cache_contract_from_phase_io(output_dir: Path, phase_artifacts: Mapping[str, Mapping[str, Any]]) -> Any | None:
    for phase in _PHASES:
        io_manifest = phase_artifacts[phase].get("io_manifest")
        if not isinstance(io_manifest, str):
            continue
        io_path = output_dir / io_manifest
        if not io_path.is_file():
            continue
        data = _load_json(io_path)
        cache_contract = data.get("cache_contract")
        if cache_contract is not None:
            return cache_contract
    return None


def _build_meta(
    *,
    output_dir: Path,
    release_prefix: str,
    manifest: Mapping[str, Any],
    phase_artifacts: Mapping[str, Mapping[str, Any]],
    quant_embedding: str,
    hf_config: str,
    source_config: str | None,
    pad_token_id: int,
) -> dict[str, Any]:
    partial = manifest.get("partial_checkpoint") if isinstance(manifest.get("partial_checkpoint"), dict) else {}
    layer_types = list(manifest.get("layer_types") or partial.get("layer_types") or [])
    ple_layer_ids = list(manifest.get("ple_layer_ids") or partial.get("ple_layer_ids") or [])
    context_length = int(manifest.get("context_length", 262144))
    prefill_length = int(manifest.get("prefill_chunk_length", 256))
    model_config = {
        "model_name": release_prefix.removeprefix("hmquant_"),
        "chip_arch": "XH2a",
        "model_type": "qwen4_exp",
        "dtype": "float16",
        "batch_size": 1,
        "context_max_length": context_length,
        "prefill_chunk_length": prefill_length,
        "num_hidden_layers": int(manifest.get("num_hidden_layers", partial.get("num_hidden_layers", 4))),
        "num_logits_to_keep": 0,
        "use_cache": True,
        "use_fused_gdr": bool(manifest.get("use_fused_gdr", True)),
        "enable_page_attention": False,
        "final_hmonnx_only": True,
        "source_config": source_config,
    }
    cache_contract = _cache_contract_from_phase_io(output_dir, phase_artifacts)
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "prepared_incomplete_golden",
        "model_config": model_config,
        "hf_config": hf_config,
        "quant_embedding": quant_embedding,
        "quant_embedding_md5": _md5(output_dir / quant_embedding),
        "prefill_hmonnx": phase_artifacts["prefill"]["hmonnx"],
        "decode_hmonnx": phase_artifacts["decode"]["hmonnx"],
        "prefill_onnx": phase_artifacts["prefill"]["hmonnx"],
        "decode_onnx": phase_artifacts["decode"]["hmonnx"],
        "token_embedding_file": quant_embedding,
        "prefill_hmonnx_md5": _md5(output_dir / phase_artifacts["prefill"]["hmonnx"]),
        "decode_hmonnx_md5": _md5(output_dir / phase_artifacts["decode"]["hmonnx"]),
        "max_context_tokens": context_length,
        "pad_token_id": int(pad_token_id),
        "qwen4_exp_cache_contract": cache_contract,
        "qwen4_exp_four_layer": {
            "source_model": manifest.get("source_model"),
            "checkpoint_dir": manifest.get("checkpoint_dir"),
            "checkpoint_note": manifest.get("checkpoint_note"),
            "partial_checkpoint": partial,
            "layer_types": layer_types,
            "ple_layer_ids": ple_layer_ids,
            "num_hidden_layers": model_config["num_hidden_layers"],
            "expected_attention_layers": manifest.get("expected_attention_layers"),
            "expected_sparse_attention_select_kv_layers": manifest.get("expected_sparse_attention_select_kv_layers"),
            "expected_linear_attention_layers": manifest.get("expected_linear_attention_layers"),
            "expected_kv_cache_nodes": manifest.get("expected_kv_cache_nodes"),
            "selected_kv_score_mode": manifest.get("selected_kv_score_mode"),
            "selected_kv_abi": manifest.get("selected_kv_abi"),
            "selected_kv_q_bit": manifest.get("selected_kv_q_bit"),
            "selected_kv_k_bit": manifest.get("selected_kv_k_bit"),
            "selected_kv_compression_ratio": manifest.get("selected_kv_compression_ratio"),
            "selected_kv_phase_proofs": {
                phase: artifacts.get("selected_kv_proof") for phase, artifacts in phase_artifacts.items()
            },
            "final_hmonnx_only": True,
            "page_cache_modified_graphs": False,
            "golden_status": "not_generated",
            "phase_artifacts": phase_artifacts,
        },
        "qwen4_exp_serving": {
            "enable_page_attention": False,
            "qsa_attention": "SparseAttentionSelectKV",
            "max_num_seqs": 1,
        },
        "meta": {"class_name": "LLMModelMeta"},
    }
    return meta


def prepare_qwen4_exp_golden_package(
    source_dir: str | Path,
    output_dir: str | Path,
    embedding_state: Mapping[str, torch.Tensor],
    hf_model_dir: str | Path,
) -> dict[str, Any]:
    """Create a standard Merak Golden package shell from completed raw four-layer exports.

    The helper only relinks/renames artifacts and rewrites ONNX external-data
    locations. It does not execute HMONNX Golden; the metadata is intentionally
    marked incomplete until the caller runs and verifies per-phase Golden dumps.
    """

    source_path = Path(source_dir).expanduser().resolve(strict=True)
    output_path = Path(output_dir).expanduser().resolve()
    if output_path == source_path or output_path in source_path.parents or source_path in output_path.parents:
        raise ValueError("package output must not replace, contain, or be contained by the source export")
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = _require_complete_manifest(source_path)
    release_prefix = _release_prefix(manifest)
    phase_artifacts = {
        phase: _package_phase(source_path, output_path, release_prefix, manifest, phase) for phase in _PHASES
    }
    quant_embedding = _save_quant_embedding(output_path, embedding_state)
    hf_config = _copy_hf_config(output_path, hf_model_dir)
    source_config = _copy_source_config(source_path, output_path)
    _replace_file_with_same_guard(source_path / "export_manifest.json", output_path / "export_manifest.json")
    merak_config = _write_merak_config(output_path)
    pad_token_id = _pad_token_id_from_hf_config(hf_model_dir)

    meta = _build_meta(
        output_dir=output_path,
        release_prefix=release_prefix,
        manifest=manifest,
        phase_artifacts=phase_artifacts,
        quant_embedding=quant_embedding,
        hf_config=hf_config,
        source_config=source_config,
        pad_token_id=pad_token_id,
    )
    meta_path = output_path / "golden_meta_info.json"
    if meta_path.exists():
        existing_meta = _load_json(meta_path)
        if not _existing_package_meta_is_same_source(existing_meta, phase_artifacts):
            raise FileExistsError(f"refusing to overwrite golden_meta_info.json for a different package: {meta_path}")
        if existing_meta.get("status") != "prepared_incomplete_golden":
            meta = existing_meta
        else:
            meta["create_time"] = existing_meta.get("create_time", meta["create_time"])
            existing_four_layer = existing_meta.get("qwen4_exp_four_layer")
            if isinstance(existing_four_layer, Mapping):
                existing_golden_status = existing_four_layer.get("golden_status")
                if existing_golden_status and existing_golden_status != "not_generated":
                    meta["qwen4_exp_four_layer"]["golden_status"] = existing_golden_status
    new_text = json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    if meta_path.exists() and meta_path.read_text(encoding="utf-8") == new_text:
        pass
    else:
        meta_path.write_text(new_text, encoding="utf-8")

    return {
        "output_dir": str(output_path),
        "release_prefix": release_prefix,
        "meta_path": str(meta_path),
        "merak_config": merak_config,
        "prefill": phase_artifacts["prefill"],
        "decode": phase_artifacts["decode"],
        "status": meta["status"],
    }


__all__ = ["prepare_qwen4_exp_golden_package"]
