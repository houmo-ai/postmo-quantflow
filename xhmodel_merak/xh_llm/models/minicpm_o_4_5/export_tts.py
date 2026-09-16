from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .export_common import build_component_model, quantize_and_export, write_standard_decoder_metadata


def export_minicpm_o_4_5_tts(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, Any],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, Any]:
    model_name = model_name or "minicpm_o_4_5"
    model = build_component_model(model_dir, component_cfg)
    host = model.get_hf_model(device_map="cpu")
    text_embedding_file = work_dir / "tts_text_embedding.pt"
    code_embedding_file = work_dir / "tts_code_embedding.pt"
    text_embedding_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(host.tts.emb_text.state_dict(), text_embedding_file)
    torch.save(host.tts.emb_code[0].state_dict(), code_embedding_file)
    model.init_wrap_model(host)
    hidden_size = int(model.config.hidden_size)
    semantic_hidden_size = int(model.semantic_hidden_size)
    prefill_length = int(component_cfg["wrap_cfg"]["input_sequence_length"])
    max_length = int(component_cfg["wrap_cfg"]["max_sequence_length"])
    prefill_data = {
        "inputs_embeds": torch.zeros((1, prefill_length, hidden_size), dtype=torch.float16),
        "past_seq_length": torch.tensor([0], dtype=torch.int32),
        "current_input_length": torch.tensor([prefill_length], dtype=torch.int32),
        "attention_mask": torch.zeros((1, 1, prefill_length, max_length), dtype=torch.float16),
        "semantic_hidden_states": torch.zeros((1, prefill_length, semantic_hidden_size), dtype=torch.float16),
        "semantic_mask": torch.ones((1, prefill_length, 1), dtype=torch.float16),
    }
    # The TTS component is a small (8-layer) LLM; calibrate it with real-shaped
    # hidden-state inputs like the qwen3_tts talker calibration, instead of the
    # shape-only dummy path used for vision/audio.
    calib_data = {
        "inputs_embeds": torch.randn((1, prefill_length, hidden_size), dtype=torch.float16),
        "past_seq_length": torch.tensor([0], dtype=torch.int32),
        "current_input_length": torch.tensor([prefill_length], dtype=torch.int32),
        "attention_mask": torch.zeros((1, 1, prefill_length, max_length), dtype=torch.float16),
        "semantic_hidden_states": torch.randn((1, prefill_length, semantic_hidden_size), dtype=torch.float16),
        "semantic_mask": torch.ones((1, prefill_length, 1), dtype=torch.float16),
    }
    prefill_graph = Path(
        quantize_and_export(
            model,
            prefill_data,
            work_dir / "tts_prefill",
            f"{model_name}_tts_prefill_{target_device}_{component_cfg['quant_type']}",
            device,
            calibrate=True,
            calib_data=calib_data,
        )
    )
    model.release_exported_model()
    model.set_input_sequence_length(1)
    decode_data = {
        "inputs_embeds": torch.zeros((1, 1, hidden_size), dtype=torch.float16),
        "past_seq_length": torch.tensor([prefill_length], dtype=torch.int32),
        "current_input_length": torch.tensor([1], dtype=torch.int32),
        "attention_mask": torch.zeros((1, 1, 1, max_length), dtype=torch.float16),
        "semantic_hidden_states": torch.zeros((1, 1, semantic_hidden_size), dtype=torch.float16),
        "semantic_mask": torch.zeros((1, 1, 1), dtype=torch.float16),
    }
    decode_graph = Path(
        model.to_export_onnx(
            decode_data,
            str(work_dir / "tts_decode"),
            f"{model_name}_tts_decode_{target_device}_{component_cfg['quant_type']}",
        )[0]
    )
    result = {
        "quant_type": component_cfg["quant_type"],
        "graphs": {
            "prefill": str(prefill_graph.relative_to(work_dir)),
            "decode": str(decode_graph.relative_to(work_dir)),
        },
        "prefill_input_sequence_length": prefill_length,
        "num_hidden_layers": len(model.past_key_caches),
        "kv_cache_shape": list(model.past_key_caches[0].shape),
        "semantic_hidden_size": semantic_hidden_size,
        "tts_hidden_size": int(model.tts_hidden_size),
        "text_embedding": str(text_embedding_file.relative_to(work_dir)),
        "code_embedding": str(code_embedding_file.relative_to(work_dir)),
    }
    metadata_file = work_dir / "metadata" / "tts_meta_info.json"
    write_standard_decoder_metadata(
        metadata_file,
        model_dir=model_dir,
        artifact_name="minicpm_o_4_5_tts",
        model_type="MiniCPMO45TTSModel",
        target_device=target_device,
        prefill_graph=prefill_graph,
        decode_graph=decode_graph,
        prefill_chunk_length=prefill_length,
        num_hidden_layers=len(model.past_key_caches),
        kv_cache_shape=list(model.past_key_caches[0].shape),
        pad_token_id=0,
        copy_hf_config=False,
    )
    result["metadata"] = str(metadata_file.relative_to(work_dir))
    return result


__all__ = ["export_minicpm_o_4_5_tts"]
