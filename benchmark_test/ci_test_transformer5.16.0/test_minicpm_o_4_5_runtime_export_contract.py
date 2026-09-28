from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


CONFIG_DIR = Path("configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5")


@pytest.mark.parametrize(
    "name",
    (
        "minicpm_o_4_5_xh2a_w4a8_gptq.yaml",
        "minicpm_o_4_5_xh2a_w8a8_gptq.yaml",
    ),
)
def test_release_configs_define_each_export_mode(name):
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.workflow import MiniCPMO45Workflow

    config = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
    export = config["export"]
    components = export["components"]

    assert export["mode"] == "offline"
    assert set(components) == set(MiniCPMO45Workflow.SUPPORTED_COMPONENTS)
    assert "tts_projection" not in components
    assert components["tts"]["input_names"][-2:] == ["semantic_hidden_states", "semantic_mask"]

    for mode in ("offline", "session", "stream"):
        MiniCPMO45Workflow._validate_export_config({**export, "mode": mode}, config["quant"])


def test_tts_graph_fuses_masked_normalized_semantics_and_audio_head():
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5._tts_model_impl import _LlamaModel

    class Graph:
        use_cache = False
        num_logits_to_keep = 0
        layers = []
        rotary_emb = SimpleNamespace(cos_cached=torch.zeros(1), sin_cached=torch.zeros(1))
        cos_slice = staticmethod(lambda value, _past: value)
        sin_slice = staticmethod(lambda value, _past: value)
        norm = torch.nn.Identity()
        semantic_projector = torch.nn.Linear(4, 3, bias=True)
        audio_code_head = torch.nn.Linear(3, 5, bias=False)
        normalize_projected_hidden = True

    graph = Graph()
    with torch.no_grad():
        graph.semantic_projector.weight.copy_(
            torch.eye(3, 4, dtype=graph.semantic_projector.weight.dtype)
        )
        graph.semantic_projector.bias.zero_()
        graph.audio_code_head.weight.copy_(
            torch.arange(1, 16, dtype=graph.audio_code_head.weight.dtype).reshape(5, 3) / 16
        )
    embeddings = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    semantic = torch.tensor([[[1.0, 0.0, 0.5, -1.0], [3.0, 2.0, 1.0, 0.0]]])
    mask = torch.tensor([[[1.0], [0.0]]])
    output = _LlamaModel.forward(
        graph,
        inputs_embeds=embeddings,
        past_seq_length=torch.tensor([0], dtype=torch.int32),
        current_input_length=torch.tensor([2], dtype=torch.int32),
        attention_mask=torch.zeros((1, 1, 2, 2)),
        semantic_hidden_states=semantic,
        semantic_mask=mask,
    ).last_hidden_state
    projected = torch.nn.functional.normalize(graph.semantic_projector(semantic), p=2, dim=-1) * mask

    torch.testing.assert_close(output, graph.audio_code_head(embeddings + projected))
    torch.testing.assert_close((embeddings + projected)[:, 1], embeddings[:, 1])
