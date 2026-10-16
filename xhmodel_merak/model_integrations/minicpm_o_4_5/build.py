from __future__ import annotations

import inspect
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact import EFFECTIVE_WORKFLOW_NAME
from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_build.contracts import BuildPlan
from xhmodel_merak.model_build.runtime_assets import quant_runtime_asset_paths

from .artifact import describe_export, load_metadata, validate_runtime_contract


def resolve_config(*, variant, **_kwargs):
    return {"export.mode": variant.name}


def project_checkpoint_config(config: Mapping[str, Any], *, source: bool) -> dict[str, Any]:
    """GPTQ stores the official nested Qwen3 LLM, not its multimodal Host."""
    from transformers import Qwen3Config

    expected = ("minicpmo", ["MiniCPMO"]) if source else ("qwen3", ["Qwen3ForCausalLM"])
    if (config["model_type"], config["architectures"]) != expected:
        raise ModelError(ErrorCode.QUANT_ARTIFACT_MISMATCH, "MiniCPM-o-4.5 requires its Qwen3 LLM checkpoint")
    text = Qwen3Config.from_dict(dict(config)).to_dict()
    if source:
        # Keep every native Qwen3 constructor field; Host modalities are not
        # part of the LLM-only checkpoint. Native defaults also expand layer_types.
        fields = inspect.signature(Qwen3Config.__init__).parameters
        text = {key: value for key, value in text.items() if key in fields}
        text["architectures"] = ["Qwen3ForCausalLM"]
    # Generation IDs come from the packaged Host, not the LLM-only checkpoint.
    text["bos_token_id"] = text["eos_token_id"] = None
    return Qwen3Config.from_dict(text).to_dict()


def normalize_export(stage_path: Path, export_result: Any, plan: BuildPlan) -> None:
    del stage_path, export_result, plan


def package_runtime(stage_path: Path, plan: BuildPlan) -> None:
    source_root = Path(plan.model_dir)
    for source in quant_runtime_asset_paths(source_root):
        destination = stage_path / "hf_config" / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    # The exporter already copied these JSON files for its decoder metadata.
    # Unified runtime resources have one self-contained HF root instead.
    decoder_config = stage_path / "metadata" / "hf_config"
    if decoder_config.is_dir():
        shutil.rmtree(decoder_config)


def sanitize_artifact(stage_path: Path, plan: BuildPlan) -> None:
    del plan
    path = stage_path / "export_meta_info.json"
    meta = load_metadata(path)
    meta["hf_model"] = meta["hf_config"] = "hf_config"
    meta["config"] = EFFECTIVE_WORKFLOW_NAME
    meta["components"]["llm"]["hf_config"] = "hf_config"
    for name in ("llm", "tts"):
        child_path = confined_file(stage_path, meta["components"][name]["metadata"], f"{name}.metadata")
        child = load_metadata(child_path)
        child["hf_config"] = "../hf_config"
        child["model_config"]["hf_model"] = "../hf_config"
        child_path.write_text(json.dumps(child, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_artifact(stage_path: Path, plan: BuildPlan) -> dict[str, Any]:
    root = stage_path.resolve()
    try:
        meta = load_metadata(confined_file(root, "export_meta_info.json", "metadata_file"))
        validate_runtime_contract(
            meta,
            resolve_file=lambda value, field: confined_file(root, value, field),
            deployment_precision=plan.deployment_precision,
            variant=plan.variant,
        )
    except ModelError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, str(error), model=plan.model_id) from error
    return {
        "metadata_file": "export_meta_info.json", "export_entries": describe_export(meta),
        "runtime_kind": "minicpm_o_4_5_hmonnx", "components": list(plan.components),
        "requires_source": False, "loadable": True, "quantized": True,
    }
