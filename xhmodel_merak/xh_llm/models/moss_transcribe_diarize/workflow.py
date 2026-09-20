from collections.abc import Mapping
import json
from pathlib import Path
import shutil
from typing import Any

from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow
from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult

from .configuration import XHMossTranscribeDiarizeConfig
from .model import XHMossTranscribeDiarizeModel


def publish_moss_serving_aliases(meta: dict[str, Any]) -> dict[str, Any]:
    """Expose LLM serving entry fields vllm-merak's generic parser expects."""
    if not isinstance(meta, dict):
        raise TypeError("MOSS golden metadata must be an object")
    decoder = meta.get("components", {}).get("prefill_decode", {})
    if not isinstance(decoder, dict):
        raise TypeError("MOSS golden metadata must contain components.prefill_decode")
    prefill = decoder.get("prefill")
    decode = decoder.get("decode")
    if not isinstance(prefill, dict) or not isinstance(decode, dict):
        raise TypeError("MOSS prefill/decode metadata must be objects")

    def _graph_ref(component: dict[str, Any], *, phase: str) -> str:
        for key in ("hmonnx_file", "onnx_file"):
            value = component.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError(f"MOSS {phase} metadata is missing a published graph path")

    published = dict(meta)
    published["prefill_hmonnx"] = _graph_ref(prefill, phase="prefill")
    published["decode_hmonnx"] = _graph_ref(decode, phase="decode")
    return published


def finalize_moss_merak_runtime_config(export_result: ExportResult) -> Path:
    meta_path = Path(BaseLLMWorkflow._find_golden_meta_file(export_result))
    meta = publish_moss_serving_aliases(json.loads(meta_path.read_text(encoding="utf-8")))
    model_config = meta.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError(f"MOSS golden metadata must contain an object field 'model_config': {meta_path}")
    explicit_page_attention = model_config.get("enable_page_attention")
    if isinstance(explicit_page_attention, bool):
        enable_page_attention = explicit_page_attention
    else:
        flash_attention = model_config.get("flash_attention")
        enable_page_attention = bool(
            isinstance(flash_attention, dict) and flash_attention.get("enable") is True
        )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    config = {
        "architectures": ["MerakForCausalLM"],
        "config_format": "merak_llm",
        "load_format": "merak_llm",
        "xh_model": {
            "model_type": "hmonnx",
            "meta_info": meta_path.name,
        },
        "enable_page_attention": enable_page_attention,
        "model_type": "merak_llm",
    }
    config_path = meta_path.parent / "merak_config.json"
    temporary_path = config_path.with_name(f".{config_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(config_path)
    return config_path


class MossTranscribeDiarizeWorkflow(BaseLLMWorkflow):
    expected_model_config_cls_name = "XHMossTranscribeDiarizeConfig"
    expected_model_cls_name = "XHMossTranscribeDiarizeModel"

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError(f"{type(self).__name__} uses export-time PTQ; quant must be null")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def get_model(
        self,
        quant_result: QuantResult | None = None,
        config_overrides: Mapping[str, Any] | None = None,
        work_dir: str | None = None,
    ) -> XHMossTranscribeDiarizeModel:
        if quant_result is None:
            quant_result = QuantResult(raw_model_dir=self.model_dir, skipped=True)
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_cfg = self._build_export_config(quant_result, workflow_config)
        model_cfg = XHMossTranscribeDiarizeConfig.from_dict(export_cfg["model"])
        xh_model = XHMossTranscribeDiarizeModel(model_cfg)
        if work_dir is not None:
            xh_model.work_dir = work_dir
        return xh_model

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
        finalize_moss_merak_runtime_config(export_result)
        from .packaging import copy_official_runtime_files

        copy_official_runtime_files(self.model_dir, export_result.work_dir)
        config_file = Path(export_result.config_file)
        meta = publish_moss_serving_aliases(dict(export_result.meta))
        meta["config"] = config_file.name
        (Path(export_result.work_dir) / "export_meta_info.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        serving_meta = Path(self._find_golden_meta_file(export_result))
        serving_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return ExportResult(
            work_dir=export_result.work_dir,
            config_file=export_result.config_file,
            meta=meta,
        )

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        if str(device).lower().startswith("cpu"):
            raise ValueError("MOSS HMONNX golden requires an XH2a-capable device, not CPU")
        work_dir = Path(export_result.work_dir).resolve()
        meta_file = work_dir / "export_meta_info.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"export_meta_info.json not found under {work_dir}")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        audio = _golden_audio_path(input_messages)
        from xhquant.api import HMONNXGoldenInference
        from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.hmonnx_runtime import (
            _runtime_tool_path,
            golden_dir_for_graph,
        )

        from .processing import extract_moss_log_mel

        mel, token_len = extract_moss_log_mel(audio, self.model_dir)
        golden_meta: dict[str, Any] = {
            "device": str(device),
            "audio_file": str(audio),
            "components": {},
        }
        component_paths = {
            "encode": meta["components"]["encode"]["hmonnx_file"],
            "prefill": meta["components"]["prefill_decode"]["prefill"]["hmonnx_file"],
            "decode": meta["components"]["prefill_decode"]["decode"]["hmonnx_file"],
        }
        fused = None
        for component, relative in component_paths.items():
            graph = (work_dir / relative).resolve()
            if not graph.is_file() or not graph.is_relative_to(work_dir):
                raise FileNotFoundError(f"MOSS HMONNX graph not found inside artifact: {relative}")
            session = HMONNXGoldenInference(str(graph))
            session.initialize()
            session.to(device)
            inputs = _golden_inputs(session, component, mel, fused)
            inputs = {name: value.to(device) for name, value in inputs.items()}
            golden_dir = golden_dir_for_graph(graph)
            if golden_dir.exists():
                shutil.rmtree(golden_dir)
            golden_dir.mkdir(parents=True, exist_ok=True)
            session.save_golden = True
            session.save_golden_dir = str(golden_dir)
            session.step = 0
            with _runtime_tool_path():
                outputs = session(*[inputs[name] for name in session.get_input_names()])
            if component == "encode":
                fused = _fused_embeds_from_encoder(outputs, audio, self.model_dir, work_dir, token_len)
                if fused is None:
                    raise RuntimeError("MOSS decoder golden requires Host fusion embeddings")
            golden_meta["components"][component] = {
                "hmonnx_file": str(graph.relative_to(work_dir)),
                "golden_dir": str(golden_dir.relative_to(work_dir)),
            }
        golden_file = work_dir / "golden_meta_info.json"
        golden_file.write_text(json.dumps(golden_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return str(golden_file)


def _golden_audio_path(input_messages: Any) -> Path:
    if isinstance(input_messages, Mapping):
        value = input_messages.get("audio") or input_messages.get("audio_file")
    else:
        value = input_messages
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("MOSS golden requires input_messages to contain an audio path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MOSS golden audio file not found: {path}")
    return path


def _fused_embeds_from_encoder(outputs: Any, audio: Path, model_dir: str, work_dir: Path, token_len: int):
    """Build Host fusion embeddings for decoder golden."""
    import numpy as np
    import torch

    from .host_fusion import fuse_embeddings
    from .processing import layout_prompt_ids, load_official_processor
    from .transcribe import embed_tokens

    audio_embeds = None
    if isinstance(outputs, torch.Tensor):
        audio_embeds = outputs.detach().cpu().numpy()
    elif isinstance(outputs, (tuple, list)) and outputs:
        audio_embeds = outputs[0].detach().cpu().numpy() if hasattr(outputs[0], "detach") else None
    elif isinstance(outputs, dict):
        value = outputs.get("audio_embeds") or next(iter(outputs.values()), None)
        audio_embeds = value.detach().cpu().numpy() if hasattr(value, "detach") else None
    if audio_embeds is None:
        raise RuntimeError("MOSS encoder golden did not return audio embeddings")
    if audio_embeds.ndim == 3:
        audio_embeds = audio_embeds[0]
    audio_embeds = audio_embeds[: int(token_len)]
    processor = None
    load_error: Exception | None = None
    for candidate in (model_dir, work_dir):
        try:
            processor = load_official_processor(candidate)
            break
        except Exception as error:
            load_error = error
    if processor is None:
        raise RuntimeError(
            "MOSS decoder golden requires Official processor for Host fusion"
        ) from load_error
    from .prefill_decode import MossLMHead
    from .processing import as_tokenizer

    lm_path = Path(work_dir) / "lm_head.pt"
    if not lm_path.is_file():
        raise FileNotFoundError(f"MOSS decoder golden requires lm_head.pt under {work_dir}")
    import torch as _torch

    state = _torch.load(lm_path, map_location="cpu", weights_only=True)
    from .transcribe import normalize_lm_head_state
    state = normalize_lm_head_state(state)
    weight = state["projection.weight"]
    head = MossLMHead(hidden_size=weight.shape[1], vocab_size=weight.shape[0])
    head.load_state_dict(state)
    input_ids = np.asarray(layout_prompt_ids(processor, None, int(token_len)), dtype=np.int64)
    tokenizer = as_tokenizer(processor)
    token_embeddings = embed_tokens(input_ids, head)
    audio_token_id = int(tokenizer.convert_tokens_to_ids("<|audio_pad|>"))
    mask = input_ids == audio_token_id
    if int(mask.sum()) != audio_embeds.shape[0]:
        raise ValueError(
            "MOSS decoder golden audio-pad count does not match encoder tokens: "
            f"{int(mask.sum())} vs {audio_embeds.shape[0]}"
        )
    return fuse_embeddings(token_embeddings, audio_embeds, mask)


def _graph_sequence_length(session: Any, component: str) -> int:
    try:
        shape = tuple(int(dim) for dim in session.get_input("inputs_embeds").shape)
    except Exception:
        shape = ()
    if len(shape) > 1 and shape[1] > 0:
        return int(shape[1])
    return 512 if component == "prefill" else 1


def _golden_inputs(session: Any, component: str, mel: Any, fused: Any = None) -> dict[str, Any]:
    import torch

    graph_len = _graph_sequence_length(session, component)
    defaults = {
        "inputs_embeds": (1, graph_len, 1024),
        "past_seq_length": (1,),
        "current_input_length": (1,),
    }
    inputs: dict[str, Any] = {}
    for name in session.get_input_names():
        info = session.get_input(name)
        dtype = info.dtype
        if name == "input_features":
            inputs[name] = torch.from_numpy(mel).unsqueeze(0).to(dtype=dtype)
            continue
        if name == "inputs_embeds" and fused is not None:
            shape = tuple(max(1, int(dim)) for dim in getattr(info, "shape", ())) or defaults["inputs_embeds"]
            padded = torch.zeros(shape, dtype=dtype)
            seq = min(int(fused.shape[0]), int(shape[1]))
            hidden = min(int(fused.shape[1]), int(shape[2]))
            padded[0, :seq, :hidden] = torch.from_numpy(fused[:seq, :hidden]).to(dtype=dtype)
            inputs[name] = padded
            continue
        if name in {"past_seq_length", "current_input_length"}:
            if fused is not None and component == "prefill" and name == "current_input_length":
                value = min(int(fused.shape[0]), graph_len)
            elif name == "past_seq_length" and component == "prefill":
                value = 0
            elif name == "past_seq_length":
                value = graph_len
            elif component == "prefill":
                value = graph_len
            else:
                value = 1
            inputs[name] = torch.tensor([value], dtype=dtype)
            continue
        shape = tuple(max(1, int(dim)) for dim in getattr(info, "shape", ()))
        if not shape:
            shape = defaults.get(name, (1,))
        inputs[name] = torch.zeros(shape, dtype=dtype)
    return inputs
