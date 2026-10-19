# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Static DepthSplat ONNX PTQ through the standard OtherModel workflow."""

from __future__ import annotations

import json
from pathlib import Path

from ...workflows.base import BaseOtherModelWorkflow
from ...workflows.result import ExportResult, QuantResult
from .assets import OUTPUT_NAMES, PROFILE, load_bundle, load_sample, onnx_contract, sample_contract, graph_inputs


class DepthSplatWorkflow(BaseOtherModelWorkflow):
    def quant(self, output_dir, device, config_overrides=None):
        cfg = self.workflow_config.with_overrides(config_overrides)
        if cfg.quant is not None:
            raise ValueError("DepthSplat performs graph PTQ during export; quant must be null")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(self, quant_result, output_dir, device, config_overrides=None):
        import onnx
        import torch
        from xhquant.api import DeviceType, QuantScheme, create_quant_config, ptq_quantize, xhquant_init
        from xhquant.api.ptq_export_hmonnx import _convert_model_to_quanted_model, convert_quanted_model_to_hmonnx
        from xhquant.common.types import FrontendType, PrecisionMode

        cfg = self.workflow_config.with_overrides(config_overrides)
        if cfg.quant is not None:
            raise ValueError("DepthSplat performs graph PTQ during export; quant must be null")
        settings = cfg.build_export_dict()
        if settings["target_device"] != "XH2a" or settings["profile"] != PROFILE:
            raise ValueError("DepthSplat currently requires XH2a and 2v_256x256")
        quant_type = settings["model"]["quant_scheme"]["quant_type"]
        if quant_type != "w8a8":
            raise ValueError("Only the initial w8a8 deployment configuration is registered")
        bundle_config, source, samples, _ = load_bundle(self._resolve_export_model_dir(quant_result))
        specs, output_names = onnx_contract(source, bundle_config.get("postprocess"))
        sample_specs = sample_contract(bundle_config, specs)
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        graph_path = root / "visual_2v_256x256/model.onnx"
        if graph_path.exists() or (root / "runtime_meta_info.json").exists():
            raise FileExistsError("DepthSplat export output already exists")
        graph_path.parent.mkdir(exist_ok=True)
        config_file = cfg.dump(str(root / "workflow.yaml"))
        xhquant_init(str(root / "export.log"), self.debug)
        target = torch.device(device)
        if target.type == "cuda":
            torch.cuda.set_device(target)
        calibration = []
        for sample in samples:
            feed = load_sample(sample, sample_specs)
            prepared = graph_inputs(feed, specs, bundle_config.get("preprocess"))
            calibration.append([torch.from_numpy(prepared[name]) for name in specs])
        quant_config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type))
        # This installed SDK helper sets each input's qspec dtype before graph conversion.
        # SDK 0.9.2 statically quantizes weights; activation SEFP exponents are
        # runtime data dependent. PTQ uses only the first sample for shape propagation.
        graph = _convert_model_to_quanted_model(
            str(source),
            FrontendType.ONNX,
            calibration[0],
            DeviceType.XH2a,
            quant_config,
            use_ptq=False,
            input_names=list(specs),
        )
        ptq_quantize(graph, calibration, PrecisionMode.ALIGNED, target)
        convert_quanted_model_to_hmonnx(graph, calibration[0], str(graph_path), list(specs), output_names)
        proto = onnx.load(str(graph_path), load_external_data=False)
        external = set()
        for tensor in proto.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                location = {item.key: item.value for item in tensor.external_data}.get("location")
                if not location:
                    raise ValueError("HMONNX external tensor has no location")
                path = (graph_path.parent / location).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError("Invalid or missing HMONNX external data")
                external.add(str(path.relative_to(root)))
        exported_specs = {}
        for value in proto.graph.input:
            if value.name in specs:
                exported_specs[value.name] = {
                    "shape": [dim.dim_value for dim in value.type.tensor_type.shape.dim],
                    "dtype": str(onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)),
                }
        if list(exported_specs) != list(specs):
            raise ValueError("Exported input ports changed unexpectedly")
        meta = {
            "schema_version": 1,
            "model_type": "XHDepthSplatModel",
            "profile": PROFILE,
            "target_device": "XH2a",
            "quant_type": quant_type,
            "input_names": list(specs),
            "output_names": output_names,
            "input_specs": exported_specs,
            "sample_input_specs": sample_specs,
            "hmonnx_file": str(graph_path.relative_to(root)),
            "hmonnx_external_data": sorted(external),
            "calibration_sample_count": len(samples),
            "ptq_sample_usage": "first_sample_shape_propagation",
            "activation_calibration": "runtime_sefp_shared_exponents",
            "accuracy_verified": False,
        }
        if "postprocess" in bundle_config:
            meta.update(
                schema_version=2,
                postprocess=bundle_config["postprocess"],
                host_input_specs=bundle_config["host_input_specs"],
                gaussian_output_names=list(OUTPUT_NAMES),
            )
        if "preprocess" in bundle_config:
            meta.update(schema_version=3, preprocess=bundle_config["preprocess"])
        (root / "runtime_meta_info.json").write_text(json.dumps(meta, indent=2) + "\n")
        return ExportResult(work_dir=str(root), config_file=config_file, meta=meta)

    def dump_golden(self, export_result, device, input_messages):
        from .runtime import run_artifact

        if not isinstance(input_messages, dict) or not isinstance(input_messages.get("sample"), str):
            raise ValueError("Golden requires {'sample': real_input_npz_path}")
        root = Path(export_result.work_dir)
        meta = json.loads((root / "runtime_meta_info.json").read_text())
        run_artifact(root, meta, input_messages["sample"], device, golden_dir=root / "golden")
        return str(root / "golden")
