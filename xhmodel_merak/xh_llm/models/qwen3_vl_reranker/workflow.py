"""Keep quant/export/Golden separate while reusing the LLM workflow."""

import json
import os
import shutil
from pathlib import Path

from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow


class XHQwen3VLRerankerWorkflow(BaseLLMWorkflow):
    expected_model_config_cls_name = "XHQwen3VLRerankerModelConfig"
    expected_model_cls_name = "XHQwen3VLRerankerModel"

    def quant(self, output_dir, device, config_overrides=None):
        config = self.workflow_config.with_overrides(config_overrides)
        if config.quant is None:
            return super().quant(output_dir, device, config_overrides)
        quant = config.quant
        if (
            quant.get("algorithm") not in {"gptqmodel", "autoround"}
            or quant.get("method") != "autoround"
            or quant.get("rotation", False)
        ):
            raise ValueError("Reranker supports AutoRound without rotation")
        if quant.get("quant_nontext_module", False) or int(quant.get("iters", 0)) <= 0:
            raise ValueError("Reranker requires language-only AutoRound with positive optimization iters")
        if int(quant.get("bits", 0)) != 4 or int(quant.get("group_size", 0)) != 64:
            raise ValueError("Reranker AutoRound requires bits=4 and group_size=64")
        if config.export["model"]["quant_scheme"]["quant_type"] not in {
            "w8a8h1_sefp",
            "w4a8h0_ssfp",
            "w8a16h1_sefp",
            "w4a16h0_ssfp",
        }:
            raise ValueError("Reranker AutoRound requires W8 SEFP or W4 SSFP base export with A8/A16")
        # The maintained dense mode loads AutoModelForImageTextToText and the
        # native processor. Its llm-only path also supports Qwen3-VL; no
        # quantizer, checkpoint packing or model loading is duplicated here.
        from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import quantize_with_autoround_api

        return quantize_with_autoround_api(
            model_dir=self.model_dir,
            output_dir=output_dir,
            device=device,
            quant_cfg=quant,
            export_model_cfg=config.export["model"],
            workflow_seed=self.seed,
        )

    def export(self, quant_result, output_dir, device, config_overrides=None):
        config = self.workflow_config.with_overrides(config_overrides)
        model_config = config.export["model"]
        checkpoint = Path(self._resolve_export_model_dir(quant_result))
        quant = json.loads((checkpoint / "config.json").read_text()).get("quantization_config") or {}
        # Existing integer carriers remain W4 even with the standard W8 base
        # scheme. Validate the checkpoint and kernel constraints by its source,
        # rather than only by the requested base scheme.
        if (
            config.quant is not None
            or quant
            or model_config["quant_scheme"]["quant_type"] in {"w4a8h0_ssfp", "w4a16h0_ssfp"}
        ):
            if quant.get("bits") != 4 or quant.get("group_size") != 64 or quant.get("provider") != "auto-round":
                raise ValueError("Reranker W4 export requires an AutoRound W4G64 checkpoint; run quant() first")
            if model_config["prefill_chunk_length"] != 256:
                raise ValueError("Reranker W4 export requires prefill_chunk_length=256")
        result = super().export(quant_result, output_dir, device, config_overrides)
        roots = [path for path in Path(result.work_dir).glob("hmquant_*") if path.is_dir()]
        if len(roots) != 1:
            raise ValueError("Reranker export must produce exactly one artifact directory")
        (roots[0] / "golden_meta_info.json").write_text(
            json.dumps(result.meta.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        intermediate = Path(result.work_dir) / "visual" / "onnx"
        if intermediate.is_dir():
            shutil.rmtree(intermediate)
            if not any(intermediate.parent.iterdir()):
                intermediate.parent.rmdir()
        return result

    def dump_golden(self, export_result, device, input_messages=None):
        metadata = Path(self._find_golden_meta_file(export_result))
        return dump_reranker_golden(metadata, device, input_messages)


def dump_reranker_golden(metadata, device, input_messages=None, *, golden_name="golden"):
    from PIL import Image

    from xhmodel_merak.xh_llm import AutoLLMHONNXModel

    if golden_name not in {"golden", "golden_synthetic", "golden_real"}:
        raise ValueError("Golden directory must be golden, golden_synthetic, or golden_real")
    model = AutoLLMHONNXModel.from_pretrained(str(metadata), device_map=[device], enable_golden=True)
    model.to(device)
    graphs = [Path(model.meta_info.prefill_hmonnx)]
    sessions = [model.prefill_model.hmonnx_session]
    if hasattr(model.visual, "models"):
        for gear in model.visual_meta.gears:
            graphs.append(Path(gear.hmonnx))
            sessions.append(model.visual.models[int(gear.image_token_capacity)].hmonnx_session)
    else:
        graphs.append(Path(model.visual_meta.hmonnx))
        sessions.append(model.visual.hmonnx_session)
    paths = [graph.parent for graph in graphs]
    for component, session in zip(paths, sessions, strict=True):
        golden = component / golden_name
        if golden.exists():
            shutil.rmtree(golden)
        golden.mkdir()
        session.save_golden_dir = str(golden)
        session.reset_step()
    if hasattr(model.visual, "models") and golden_name != "golden_real":
        # Each independently shaped graph gets a real step-0 baseline, even when
        # representative images route to only a subset of the configured gears.
        for gear, visual_model in model.visual.models.items():
            visual_model.forward(*model.visual._full_capacity_inputs(gear))
            visual_model.update_step()
    if input_messages is None:
        input_messages = {
            "query": {"text": "Find the red image."},
            "documents": [
                {
                    "image": Image.new(
                        "RGB",
                        (
                            getattr(model.visual_meta, "image_size_w", None) or 448,
                            getattr(model.visual_meta, "image_size_h", None) or 448,
                        ),
                        "red",
                    )
                }
            ],
        }
    scores = model.rerank(**input_messages)
    if golden_name == "golden_real":
        # Synthetic capture covers every gear. Representative real inputs
        # capture the graphs selected by the normal smallest-fit scheduler.
        executed = [graph for graph in graphs if (graph.parent / golden_name / "step_0").is_dir()]
        if len(executed) < 2:
            raise RuntimeError("Real-image Golden must execute Prefill and at least one visual graph")
        for graph in graphs:
            directory = graph.parent / golden_name
            if graph not in executed and directory.is_dir():
                directory.rmdir()
        graphs = executed
        paths = [graph.parent for graph in graphs]
    for component in paths:
        if not (component / golden_name / "step_0").is_dir():
            raise RuntimeError(f"Missing real Golden execution for {component}")
    for graph in graphs:
        link_golden_graph(graph, graph.parent / golden_name)
    return {
        "metadata": str(metadata),
        "scores": scores,
        "golden_dirs": [str(component / golden_name) for component in paths],
    }


def link_golden_graph(graph, golden_dir):
    """Use ordinary hard links: published Artifact inventories forbid symlinks."""
    import onnx

    model = onnx.load(graph, load_external_data=False)
    files = {graph.resolve()}
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                path = (graph.parent / entry.value).resolve()
                if not path.is_relative_to(graph.parent.resolve()) or not path.is_file():
                    raise ValueError("Golden graph external data must remain inside its component")
                files.add(path)
    for step in golden_dir.glob("step_*"):
        for path in files:
            link = step / path.relative_to(graph.parent.resolve())
            link.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, link)
