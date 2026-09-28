"""Write the Merak loader entry point for an exported autoregressive graph."""

import json
from pathlib import Path

from .base import BaseLLMWorkflow
from .result import ExportResult


def finalize_merak_runtime_config(export_result: ExportResult) -> Path:
    meta_path = Path(BaseLLMWorkflow._find_golden_meta_file(export_result))
    config_path = _write_merak_runtime_config(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    root_dir = meta_path.parent.resolve()
    for adapter in meta.get("lora_adapters") or []:
        adapter_meta = (root_dir / adapter["meta_file"]).resolve()
        if not adapter_meta.is_relative_to(root_dir) or adapter_meta == meta_path.resolve():
            raise ValueError("LoRA runtime metadata must refer to a child of the exported model")
        _write_merak_runtime_config(adapter_meta)
    return config_path


def _write_merak_runtime_config(meta_path: Path) -> Path:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise TypeError(f"Golden metadata must be an object: {meta_path}")
    model_config = meta.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError(f"Golden metadata must contain an object field 'model_config': {meta_path}")

    explicit_page_attention = model_config.get("enable_page_attention")
    if isinstance(explicit_page_attention, bool):
        enable_page_attention = explicit_page_attention
    else:
        flash_attention = model_config.get("flash_attention")
        enable_page_attention = bool(isinstance(flash_attention, dict) and flash_attention.get("enable") is True)

    config = {
        "architectures": ["MerakForCausalLM"],
        "config_format": "merak_llm",
        "load_format": "merak_llm",
        "xh_model": {"model_type": "hmonnx", "meta_info": meta_path.name},
        "enable_page_attention": enable_page_attention,
        "model_type": "merak_llm",
    }
    config_path = meta_path.parent / "merak_config.json"
    temporary_path = config_path.with_name(f".{config_path.name}.tmp")
    temporary_path.write_text(json.dumps(config, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    temporary_path.replace(config_path)
    return config_path
