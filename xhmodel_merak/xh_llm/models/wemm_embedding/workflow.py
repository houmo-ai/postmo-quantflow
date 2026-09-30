"""Workflow specialization for the prefill-only WeMM embedding runtime."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow
from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult


QWEN35_AUTOROUND_PROVENANCE = "qwen35_autoround_recipe_provenance.json"
WEMM_AUTOROUND_PROVENANCE = "wemm_embedding_autoround_recipe_provenance.json"
QWEN35_GPTQ_PROVENANCE = "qwen35_quant_recipe_provenance.json"
WEMM_GPTQ_PROVENANCE = "wemm_embedding_gptq_recipe_provenance.json"

PROVENANCE_ALIASES = {
    QWEN35_AUTOROUND_PROVENANCE: WEMM_AUTOROUND_PROVENANCE,
    QWEN35_GPTQ_PROVENANCE: WEMM_GPTQ_PROVENANCE,
}


def finalize_wemm_merak_runtime_config(export_result: ExportResult) -> Path:
    """Turn the shared Qwen runtime entry into a WeMM pooling entry.

    Qwen3.5 owns the common HMONNX release layout.  WeMM keeps that layout but
    has a different public runtime contract: prefill hidden states are pooled
    into an embedding and decode is never entered.  Record that distinction in
    the release config instead of requiring every deployment command to supply
    ``--runner pooling`` manually.
    """

    meta_path = Path(XHWeMMEmbeddingWorkflow._find_golden_meta_file(export_result))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError(f"WeMM golden metadata must be an object: {meta_path}")

    expected = {
        "embedding_output": "hidden_states",
        "embedding_pooling": "last_token",
        "embedding_normalize": True,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"WeMM Merak runtime requires {field}={value!r}, got {metadata.get(field)!r} in {meta_path}"
            )
    embedding_dimension = metadata.get("embedding_dimension")
    if type(embedding_dimension) is not int or embedding_dimension <= 0:
        raise ValueError(
            "WeMM Merak runtime requires a positive integer embedding_dimension, "
            f"got {embedding_dimension!r} in {meta_path}"
        )

    config_path = meta_path.parent / "merak_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"WeMM Merak config must be an object: {config_path}")
    config.update(
        {
            "architectures": ["WeMMEmbeddingMerakForEmbedding"],
            "runner_type": "pooling",
        }
    )
    temporary_path = config_path.with_name(f".{config_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(config_path)
    return config_path


class XHWeMMEmbeddingWorkflow(Qwen35Workflow):  # noqa: N801
    """Reuse Qwen3.5 quant/export while generating golden via ``embed``."""

    expected_model_cls_names = {"XHWeMMEmbeddingModel"}
    expected_model_config_cls_names = {"XHWeMMEmbeddingModelConfig"}
    family_label = "WeMM-Embedding"

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        quant_result = super().quant(output_dir, device, config_overrides)
        if quant_result.quanted_model_dir:
            self.ensure_quant_provenance_aliases(quant_result.quanted_model_dir)
        return quant_result

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        export_result = super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )
        finalize_wemm_merak_runtime_config(export_result)
        return export_result

    @staticmethod
    def ensure_quant_provenance_aliases(quant_model_dir: str | Path) -> list[Path]:
        quant_model_dir = Path(quant_model_dir)
        destinations = []
        for source_name, destination_name in PROVENANCE_ALIASES.items():
            source = quant_model_dir / source_name
            if not source.is_file():
                continue
            destination = quant_model_dir / destination_name
            shutil.copy2(source, destination)
            destinations.append(destination)
        return destinations

    @staticmethod
    def ensure_autoround_provenance_alias(quant_model_dir: str | Path) -> Path | None:
        destinations = XHWeMMEmbeddingWorkflow.ensure_quant_provenance_aliases(quant_model_dir)
        for destination in destinations:
            if destination.name == WEMM_AUTOROUND_PROVENANCE:
                return destination
        return None

    @staticmethod
    def ensure_gptq_provenance_alias(quant_model_dir: str | Path) -> Path | None:
        destinations = XHWeMMEmbeddingWorkflow.ensure_quant_provenance_aliases(quant_model_dir)
        for destination in destinations:
            if destination.name == WEMM_GPTQ_PROVENANCE:
                return destination
        return None

    def _build_export_config(
        self,
        quant_result: QuantResult,
        workflow_config: Any,
    ) -> dict[str, Any]:
        """Replace YAML ``hf_model: null`` with an existing model directory.

        The original recipe value must still be the null placeholder. After the
        base workflow overwrite, leftover null/empty values, files, and missing
        paths are rejected.
        """

        from .export_checks import (
            require_hf_model_directory,
            require_overwritten_hf_model,
            require_placeholder_hf_model,
        )

        require_placeholder_hf_model(workflow_config.export)
        # Qwen3.5 may inspect checkpoint metadata while building the export
        # config, so reject a failed placeholder overwrite before delegation.
        require_hf_model_directory(self._resolve_export_model_dir(quant_result))
        export_cfg = super()._build_export_config(quant_result, workflow_config)
        export_cfg["model"]["hf_model"] = require_overwritten_hf_model(export_cfg)
        return export_cfg

    def dump_golden(
        self,
        export_result,
        device: str,
        input_messages: Any = None,
    ) -> str:
        from xhmodel_merak.xh_llm import AutoLLMHONNXModel

        root_meta = self._find_golden_meta_file(export_result)
        if input_messages is None:
            item: dict[str, Any] = {"text": "Represent this input."}
        elif isinstance(input_messages, str):
            item = {"text": input_messages}
        elif isinstance(input_messages, Mapping):
            item = dict(input_messages)
        else:
            raise TypeError("WeMM golden input must be text or a mapping")

        model = AutoLLMHONNXModel.from_pretrained(
            root_meta,
            enable_golden=True,
            device_map=[device],
        )
        model.embed(item)
        return root_meta


__all__ = [
    "QWEN35_AUTOROUND_PROVENANCE",
    "QWEN35_GPTQ_PROVENANCE",
    "WEMM_AUTOROUND_PROVENANCE",
    "WEMM_GPTQ_PROVENANCE",
    "XHWeMMEmbeddingWorkflow",
    "finalize_wemm_merak_runtime_config",
]
