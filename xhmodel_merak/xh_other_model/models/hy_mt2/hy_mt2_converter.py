# Copyright 2025 HOUMO AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import HunYuanDenseV1ForCausalLM

from xhquant.api import CacheTensor

from ...builder import wrap_llm_model
from ._export_utils import load_quant_weight, xh1_hmonnx_compatible
from .evaluation import build_padded_chat_input_ids, build_representative_prompt
from .hy_mt2_convert_config import HyMT2ConvertConfig


from xhquant.api import (  # type: ignore # isort:skip
    Config,
    ConfigDict,
    DeviceType,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    is_ssfp_quant_config,
)


class HyMT2ConverterXH2a:
    target_device = DeviceType.XH2a

    def __init__(self, config: HyMT2ConvertConfig):
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        native_model = AutoModelForCausalLM.from_pretrained(hf_model_dir, trust_remote_code=True, **kwargs)
        assert isinstance(native_model, HunYuanDenseV1ForCausalLM), (
            f"The model is not HunYuanDenseV1ForCausalLM, but {type(native_model)}"
        )

        _untie_word_embeddings_for_export(native_model)

        self.hf_model_path = hf_model_dir
        return native_model

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        config = self.config
        native_model = self.load_hf_model(hf_model_path, dtype=torch.float16, device_map="cpu")

        resume_from = self.config.quant_weight
        if resume_from is not None:
            load_quant_weight(resume_from, native_model)
        lm_head = native_model.lm_head
        if not hasattr(lm_head, "quant_weight"):
            config.quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"

        model_name = Path(hf_model_path).name
        target_device = config.quant_scheme.target_device
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        assert target_device == DeviceType.XH2a, f"Only support convert to XH2a, but got {target_device}"
        quant_type = config.quant_scheme.quant_type
        quant_config = ConfigDict(create_quant_config(config.quant_scheme))

        meta_info: Dict[str, Any] = dict(create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        meta_info["device"] = str(target_device)
        meta_info["model_name"] = model_name
        meta_info["hf_model_path"] = hf_model_path
        meta_info["quant_scheme"] = config.quant_scheme.to_dict()
        meta_info["quant_weight"] = resume_from

        work_dir = Path(output_dir)
        hf_config_dir = work_dir / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        hf_config_files = [
            "config.json",
            "configuration.json",
            "generation_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "chat_template.jinja",
        ]
        for cfg_file in hf_config_files:
            src_file = Path(hf_model_path) / cfg_file
            dst_file = hf_config_dir / cfg_file
            if src_file.exists():
                shutil.copyfile(src_file, dst_file)
            else:
                logger.warning(f"{src_file} not exists, skip copy")
        meta_info["hf_config"] = str(hf_config_dir.relative_to(work_dir))

        token_embedding = native_model.model.get_input_embeddings()
        token_embedding_file = work_dir / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        from ._model import register_wrap_modules as hy_mt2_register_wrap_modules

        hy_mt2_register_wrap_modules(native_model)

        wrap_cfg = Config(
            dict(
                max_sequence_length=context_length,
                input_sequence_length=input_sequence_length,
                use_cache=True,
                num_logits_to_keep=config.num_logits_to_keep,
                kv_cache=dict(cache_axis=2),
            )
        )
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        wraped_model = wrap_llm_model(native_model, wrap_cfg)
        num_hidden_layers = wraped_model.model.config.num_hidden_layers
        head_dim = wraped_model.model.layers[0].self_attn.head_dim
        num_decoder_layers = num_hidden_layers
        kv_cache_shape = [
            1,
            wraped_model.model.config.num_key_value_heads,
            wrap_cfg.max_sequence_length,
            head_dim,
        ]
        meta_info["kv_cache"] = dict(shape=kv_cache_shape, num_decoder_layers=num_decoder_layers)

        past_key_caches = []
        past_value_caches = []
        for _ in range(num_decoder_layers):
            past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
            past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))

        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
        calibration_prompt = build_representative_prompt()
        input_ids_t, calibration_token_count = build_padded_chat_input_ids(
            tokenizer,
            calibration_prompt,
            input_sequence_length,
        )
        meta_info["calibration"] = {
            "prompt": calibration_prompt,
            "token_count": calibration_token_count,
            "sequence_length": input_sequence_length,
        }
        inputs_embeds = token_embedding(input_ids_t)
        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor([calibration_token_count], dtype=torch.int32)

        inputs = (
            inputs_embeds,
            past_seq_length_t,
            current_input_length_t,
            past_key_caches,
            past_value_caches,
        )
        input_names = ["inputs_embeds", "past_seq_length", "current_input_length"]
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_value_cache_{layer_idx}")
        output_names = ["logits"]

        prefix = f"{model_name}-{target_device}-{context_length // 1024}k-{quant_type}"
        prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        logger.info("********************* start export prefill model *********************")
        if prefill_onnx_file.exists():
            logger.warning(f"{prefill_onnx_file} already exists, skip export")
            quanted_model = None
        else:
            quanted_model = convert_fx_model_to_quanted_model(
                wraped_model,
                inputs,
                target_device,
                quant_config=quant_config,
            )
            input_names = xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(quanted_model, inputs, str(prefill_onnx_file), input_names, output_names)
            logger.info(f"Export Prefill model to {prefill_onnx_file}")

        logger.info("********************* start export decode model *********************")
        decode_inputs = (
            inputs_embeds[:, calibration_token_count - 1 : calibration_token_count, :],
            past_seq_length_t,
            torch.ones_like(current_input_length_t),
            past_key_caches,
            past_value_caches,
        )
        wrap_cfg.input_sequence_length = 1
        decode_onnx_file = work_dir / "hmonnx" / "decode" / f"{prefix}_decoder.onnx"
        decode_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["decode_onnx"] = str(decode_onnx_file.relative_to(work_dir))
        if decode_onnx_file.exists():
            logger.warning(f"{decode_onnx_file} already exists, skip export")
        else:
            if quanted_model is None:
                quanted_model = convert_fx_model_to_quanted_model(
                    wraped_model,
                    inputs,
                    target_device,
                    quant_config=quant_config,
                )
            quanted_model.update_cfg(wrap_cfg)
            input_names = xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_model,
                decode_inputs,
                str(decode_onnx_file),
                input_names,
                output_names,
            )
            logger.info(f"Export decode model to {decode_onnx_file}")
        (work_dir / "meta.json").write_text(
            json.dumps(meta_info, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

    @classmethod
    def convert(cls, hf_model_path: str, config: HyMT2ConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
            if not hasattr(hf_config, "quantization_config"):
                assert config.quant_weight is not None and Path(config.quant_weight).exists()
        HyMT2ConverterXH2a(config)._convert(hf_model_path, output_dir)


def _untie_word_embeddings_for_export(model: Any) -> None:
    """Clone tied output weights because HMONNX exports each parameter independently."""
    if not bool(getattr(model.config, "tie_word_embeddings", False)):
        return
    model.tie_weights()
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("tie_word_embeddings is enabled but the model has no output embedding weight")
    original_weight = output_embeddings.weight
    output_embeddings.weight = torch.nn.Parameter(
        original_weight.detach().clone(),
        requires_grad=original_weight.requires_grad,
    )
    model.config.tie_word_embeddings = False
