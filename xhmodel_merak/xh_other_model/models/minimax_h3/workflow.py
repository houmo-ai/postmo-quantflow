from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import onnx
import torch

from xhmodel_merak.exec_backends import HMONNXInferenceConfig, HMONNXInferenceV2
from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult

from .minimax_h3_converter import COMPONENTS, MiniMaxH3Converter


class MiniMaxH3Workflow(BaseOtherModelWorkflow):
    """Export MiniMax-H3 components independently to W8A8SSFP HMONNX."""

    def quant(self, output_dir: str, device: str, config_overrides: Mapping[str, Any] | None = None) -> QuantResult:
        config = self.workflow_config.with_overrides(config_overrides)
        if config.quant is not None:
            raise NotImplementedError("MiniMax-H3 quantization is integrated in export; set quant: null")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True, meta={"reason": "integrated export"})

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        config = self.workflow_config.with_overrides(config_overrides)
        export_cfg = config.build_export_dict()
        h3_cfg = export_cfg.get("minimax_h3") or {}
        components = tuple(h3_cfg.get("components", COMPONENTS))
        unknown = sorted(set(components) - set(COMPONENTS))
        if unknown:
            raise ValueError(f"Unsupported MiniMax-H3 components: {unknown}; expected {COMPONENTS}")
        output = Path(output_dir).expanduser().resolve()
        # if output.exists():
        #     raise FileExistsError(f"Export directory already exists: {output}")
        output.mkdir(parents=True, exist_ok=True)
        config.dump(str(output / f"{config.name}.yaml"))
        converter = MiniMaxH3Converter(
            self.model_dir,
            export_cfg,
            device=device,
            seed=self.seed,
        )
        component_meta = converter.export(output, components)
        meta_path = output / "export_meta_info.json"
        existing_components: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_meta = {}
            if isinstance(existing_meta.get("components"), dict):
                existing_components = existing_meta["components"]
        merged_components = dict(existing_components)
        merged_components.update(component_meta)
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": f"{config.name}.yaml",
            "model_type": export_cfg["model"]["type"],
            "target_device": export_cfg["target_device"],
            "components": merged_components,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return ExportResult(work_dir=str(output), config_file=str(output / f"{config.name}.yaml"), meta=meta)

    def dump_golden(self, export_result: ExportResult, device: str, input_messages: Any = None) -> str:
        del input_messages
        work_dir = Path(export_result.work_dir).expanduser().resolve()
        meta_file = work_dir / "export_meta_info.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"MiniMax-H3 export metadata not found: {meta_file}")
        export_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        execution_device = torch.device(device)
        component_results = {}
        for component, component_meta in export_meta.get("components", {}).items():
            hmonnx_file = work_dir / str(component_meta["hmonnx_file"])
            inputs, input_specs = _build_golden_inputs(hmonnx_file, component_meta, execution_device, self.seed)
            golden_dir = work_dir / component / "golden" / hmonnx_file.stem
            golden_dir.mkdir(parents=True, exist_ok=True)

            session_config = HMONNXInferenceConfig(
                enable_golden=True,
                enable_auto_offload=False,
                exec_devices=[execution_device],
            )
            session = HMONNXInferenceV2(str(hmonnx_file), session_config)
            session.save_golden_dir = str(golden_dir)
            session.golden_dir = str(golden_dir)
            with torch.no_grad():
                session.forward(*inputs)
            component_results[component] = {
                "hmonnx_file": str(hmonnx_file.relative_to(work_dir)),
                "golden_dir": str(golden_dir.relative_to(work_dir)),
                "inputs": input_specs,
            }
            del inputs, session
            if execution_device.type == "cuda":
                torch.cuda.empty_cache()

        golden_meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "components": component_results,
        }
        golden_meta_file = work_dir / "golden_meta_info.json"
        golden_meta_file.write_text(json.dumps(golden_meta, indent=2), encoding="utf-8")
        return str(golden_meta_file)


_ONNX_TO_TORCH_DTYPE = {
    onnx.TensorProto.FLOAT: torch.float32,
    onnx.TensorProto.FLOAT16: torch.float16,
    onnx.TensorProto.BFLOAT16: torch.bfloat16,
    onnx.TensorProto.INT64: torch.int64,
    onnx.TensorProto.INT32: torch.int32,
    onnx.TensorProto.INT8: torch.int8,
    onnx.TensorProto.BOOL: torch.bool,
}


def _build_golden_inputs(
    hmonnx_file: Path,
    component_meta: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    model = onnx.load(str(hmonnx_file), load_external_data=False)
    initializer_names = {value.name for value in model.graph.initializer}
    graph_inputs = [value for value in model.graph.input if value.name not in initializer_names]
    metadata_names = list(component_meta.get("input_names") or ())
    graph_names = [value.name for value in graph_inputs]
    if metadata_names and metadata_names != graph_names:
        raise ValueError(f"HMONNX input order differs: graph={graph_names}, metadata={metadata_names}")

    generator = torch.Generator(device=device).manual_seed(seed)
    inputs = []
    specs = []
    metadata_specs = {item["name"]: item for item in component_meta.get("inputs", ())}
    for value in graph_inputs:
        tensor_type = value.type.tensor_type
        dtype = _ONNX_TO_TORCH_DTYPE.get(tensor_type.elem_type)
        if dtype is None:
            raise TypeError(f"Unsupported ONNX input dtype {tensor_type.elem_type}: {value.name}")
        metadata_spec = metadata_specs.get(value.name)
        shape = [int(dim.dim_value) for dim in tensor_type.shape.dim]
        if metadata_spec is not None:
            shape = [int(item) for item in metadata_spec["shape"]]
        if any(item <= 0 for item in shape):
            raise ValueError(f"Cannot resolve static shape for HMONNX input {value.name}: {shape}")
        if dtype.is_floating_point:
            tensor = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        elif value.name == "attention_mask":
            tensor = torch.ones(shape, device=device, dtype=dtype)
        elif value.name == "position_ids":
            # Text-only Qwen3-VL pad semantics: valid positions are arange, pads are 1.
            seq_len = int(shape[-1])
            base = torch.arange(seq_len, device=device, dtype=dtype).view(1, 1, seq_len)
            tensor = base.expand(*shape).contiguous()
        elif value.name == "input_ids":
            tensor = torch.randint(0, 1024, shape, device=device, dtype=dtype, generator=generator)
        else:
            tensor = torch.zeros(shape, device=device, dtype=dtype)
        inputs.append(tensor)
        specs.append({"name": value.name, "shape": shape, "dtype": str(dtype).removeprefix("torch.")})
    return inputs, specs
