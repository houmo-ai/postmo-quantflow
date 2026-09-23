"""Describe actual autoregressive runtime graphs, including gears and LoRA views."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import confined_file


def describe_export(root: Path, meta: Mapping[str, Any], *, base: Path = Path()) -> dict[str, Any]:
    result: dict[str, Any] = {}
    seen: set[str] = set()

    def add(name: str, value: str, kind: str) -> None:
        path = (base / value).as_posix()
        if path in seen:
            return
        if name in result:
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, f"Duplicate export component: {name}")
        result[name] = {"path": path, "type": kind}
        seen.add(path)

    for name in ("prefill", "decode"):
        add(name, meta[f"{name}_hmonnx"], name)
    add("tokenizer", meta["hf_config"], "tokenizer")
    add("quant_embedding", meta["quant_embedding"], "embedding")
    if meta.get("per_layer_input_embedding"):
        add("per_layer_input_embedding", meta["per_layer_input_embedding"], "embedding")

    def graphs(value: Any, label: str) -> None:
        if isinstance(value, list):
            for item in value:
                graphs(item, label)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(item, str) and (key in {"hmonnx", "onnx"} or key.endswith(("_hmonnx", "_onnx"))):
                    if not item:
                        continue
                    # All graphs in this family's runtime contract are HMONNX,
                    # including legacy fields named *_onnx.
                    role = Path(item).parent.name or label
                    kind = "prefill" if "prefill" in key else "decode" if "decode" in key else "hmonnx"
                    add(role, item, kind)
                else:
                    graphs(item, label)

    for component in ("visual", "video_visual", "audio"):
        graphs(meta.get(f"{component}_config"), component)
    spec = meta.get("spec_decode")
    if isinstance(spec, Mapping):
        mode = spec.get("mode")
        if mode == "mtp":
            draft = spec.get("draft")
            phases = (
                (("prefill", draft.get("prefill_hmonnx"), "prefill"),
                 ("decode", draft.get("decode_hmonnx"), "decode"))
                if isinstance(draft, Mapping) else
                (("prefill", spec.get("draft_prefill_onnx") or spec.get("draft_onnx"), "prefill"),
                 ("decode", spec.get("draft_decode_onnx") or spec.get("draft_onnx"), "decode"))
            )
            standalone = spec.get("standalone_draft_decode_onnx")
            if standalone and standalone != dict((phase, value) for phase, value, _ in phases).get("decode"):
                phases += (("standalone_decode", standalone, "decode"),)
        elif mode == "dflash":
            phases = (
                ("context", spec.get("draft_context_onnx"), "prefill"),
                ("context_decode", spec.get("draft_context_decode_onnx"), "decode"),
                ("decode", spec.get("draft_decode_onnx") or spec.get("draft_onnx"), "decode"),
            )
        else:
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, f"Unknown speculative mode: {mode!r}")
        for phase, value, kind in phases:
            if value:
                # One physical graph can implement both stages. Preserve both
                # semantic entries, but do not list its legacy aliases again.
                result[f"{mode}_draft_{phase}"] = {"path": (base / value).as_posix(), "type": kind}

    adapters = meta.get("lora_adapters", [])
    if adapters:
        if base != Path():
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Nested LoRA inventories are not supported")
        children = {}
        for adapter in adapters:
            name = adapter["name"]
            if name in children:
                raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, f"Duplicate LoRA checkpoint: {name}")
            path = confined_file(root.resolve(), adapter["meta_file"], "lora.meta_file")
            child_meta = json.loads(path.read_text(encoding="utf-8"))
            # Child metadata can inherit the root adapter list; it describes a
            # single view, so only the root list defines checkpoint grouping.
            child_meta.pop("lora_adapters", None)
            children[name] = describe_export(root, child_meta, base=path.parent.relative_to(root.resolve()))
        result["lora"] = children
    return result
