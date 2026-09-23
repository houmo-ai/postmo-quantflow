"""Family export orchestration must return SDK artifacts without final graph rewriting."""
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def sdk_export(tmp_path):
    from xhquant.api import (to_frontend_graph, to_quant_graph, ptq_quantize,
                             PrecisionMode, to_export_hmonnx_from_quanted_graph)
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(128, 128, bias=False).half()
        def forward(self, hidden):
            return self.lm_head(hidden)
    args = [torch.zeros(1, 1, 128).half()]
    graph = to_quant_graph(to_frontend_graph(Head().eval(), "TorchFX", args), "XH2a", {"quant_type": "w8a8h1_sefp"})
    ptq_quantize(graph, [args], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    directory = tmp_path / "sdk"
    directory.mkdir()
    to_export_hmonnx_from_quanted_graph(graph, args, str(directory / "head.onnx"), save_as_external_data=True)
    return directory


@pytest.mark.parametrize("family", ["qwen", "macaron", "deepseek"])
@pytest.mark.parametrize("low_memory", [False, True])
def test_family_returns_sdk_files_unchanged(tmp_path, monkeypatch, sdk_export, family, low_memory):
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
    from xhmodel_merak.xh_llm.models.macaron_v1_tall.macaron_v1_tall_model import XHMacaronV1TallModel
    from xhmodel_merak.xh_llm.models.deepseek_v4.deepseek_v4_model import XHDeepSeekV4Model
    from xhmodel_merak.xh_llm.types import ExportData, LLMModelMeta, LLMModelState, ModelSwitcher
    from xhmodel_merak.xh_llm import utils

    monkeypatch.setenv("HUGE_MODEL_EXPORT_ENABLED", "1" if low_memory else "0")
    assert utils.is_huge_model_export_enabled() == low_memory
    parent = {"qwen": XHQwen3_5Model, "macaron": XHMacaronV1TallModel, "deepseek": XHDeepSeekV4Model}[family]
    # Supply the expensive model-specific export stages with a real SDK result.
    # The family export_hmonnx method itself runs unchanged in both branches.
    class ExportHarness(parent):
        def __init__(self):
            self.config = SimpleNamespace(lora=None, context_max_length=64)
            self.visual = None
            self._state = LLMModelState.QUANTED_ALIGNED
            self._quanted_model = ModelSwitcher({"prefill": SimpleNamespace(fixed=lambda: None),
                                                "decode": SimpleNamespace(fixed=lambda: None)})
            self.calls = []
        def get_export_info(self, output_dir):
            info = ExportData()
            info.exported_dir = output_dir
            info.meta = LLMModelMeta(prefill_hmonnx="prefill/head.onnx", decode_hmonnx="decode/head.onnx")
            self.info = info
            return info
        def _export_visual_hmonnx_impl(self, info):
            pass
        def to_wrap(self):
            pass
        def _write_sdk_files(self, info, mode):
            self.calls.append(mode)
            for role in ("prefill", "decode"):
                folder = Path(info.exported_dir) / role
                shutil.copytree(sdk_export, folder)
                setattr(info.meta, role + "_hmonnx_md5", hashlib.md5((folder / "head.onnx").read_bytes()).hexdigest())
            return info.meta
        def _export_language_hmonnx_impl(self, info, **kwargs):
            return self._write_sdk_files(info, "standard")
        def _export_big_language_hmonnx(self, info):
            return self._write_sdk_files(info, "low_memory")
        def _export_hmonnx(self, info):
            return self._write_sdk_files(info, "standard")
    model = ExportHarness()
    output = tmp_path / "export"
    model.export_hmonnx(str(output))
    assert model.calls == ["low_memory" if low_memory else "standard"]
    for source in sdk_export.iterdir():
        for role in ("prefill", "decode"):
            assert (output / role / source.name).read_bytes() == source.read_bytes()
