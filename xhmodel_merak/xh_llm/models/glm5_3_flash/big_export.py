# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Stream one GLM dense/MoE FFN at a time through the placeholder exporter."""

import re

import torch
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP, Glm5NextTextMoE

from ...big_hf_model_helper import BigHFModelExportHelper
from .checkpoint import Glm53Checkpoint
from .modeling import StaticFFNPlaceholder, StaticMoE


class Glm53BigHFModel(BigHFModelExportHelper):
    PLACEHOLDER_TYPES = ["Glm5NextTextMoE", "Glm5NextTextMLP"]

    @classmethod
    def initialize_process_worker(cls):
        # FP8 block rows are small elementwise operations. A large intra-op
        # pool per GPU worker oversubscribes the host and stalls expert loading.
        torch.set_num_threads(1)

    def _load_weight_mapping(self, hf_model):
        reader = Glm53Checkpoint(self._hf_model_dir)
        self._weight_mapping.weight_map = self._read_weight_map()
        self._weight_mapping.weight_dir = self._hf_model_dir
        skip = []
        for index, layer in enumerate(hf_model.language_model.layers):
            layer.mlp._glm53_placeholder = True
            skip.append(f"layers.{index}.mlp")
        reader.load_module(hf_model.language_model, "model.language_model.", skip_prefixes=skip)
        reader.load_module(hf_model.lm_head, "lm_head.")
        self._weight_mapping.tensor_id_to_param_names = {id(p): n for n, p in hf_model.named_parameters() if p.is_meta}

    def register_layer_as_placeholder(self, hf_model):
        # Only the identity marker is a leaf. Real dense/shared MLPs must remain
        # traceable inside each independently exported FFN, including serial export.
        self._placeholder_types = {StaticFFNPlaceholder}
        super().register_layer_as_placeholder(hf_model)

    def prepare_materialized_non_placeholder_model(self, hf_model, quantization_config):
        # Original FP8 matrices were already dequantized by the strict reader.
        pass

    @staticmethod
    def _preprocess_quantized_hf_model(hf_model, hf_model_dir=None, **kwargs):
        # Preserve the raw checkpoint layout; no GPTQ/AutoRound producer here.
        pass

    def _load_place_holder_module_once(self, empty_hf_model_for_placeholder, module_target):
        match = re.fullmatch(r"layers\.(\d+)\.mlp", module_target)
        if match is None:
            raise ValueError(f"unexpected GLM FFN placeholder target {module_target}")
        reader = Glm53Checkpoint(self._hf_model_dir)
        index = int(match[1])
        sparse = reader.text_config.mlp_layer_types[index] == "sparse"
        with torch.device("meta"):
            source = Glm5NextTextMoE(reader.text_config) if sparse else Glm5NextTextMLP(reader.text_config)
        reader.load_module(source, f"model.language_model.layers.{index}.mlp.")
        return StaticMoE(source) if sparse else source
