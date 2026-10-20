"""Exercise the real publication hook with small executable graphs and draft closure."""
import hashlib
import json
import shutil
import tarfile

import numpy as np
import pytest

from test_glm53_shared_weights import evaluate, write_graph
from xhmodel_merak.model_integrations.deepseek_v4.build import normalize_export


@pytest.mark.parametrize("dspark", [False, True])
def test_target_hardlinks_preserve_outputs_and_independent_draft(tmp_path, dspark):
    stage = tmp_path / "stage"
    root = stage / "hmquant_tiny"
    graphs = {role: write_graph(root, role, bias)
              for role, bias in (("prefill", [1, 2]), ("decode", [3, 4]))}
    metadata = {role + "_hmonnx": path for role, path in graphs.items()}
    metadata.update({role + "_onnx": path for role, path in graphs.items()})
    draft = {}
    if dspark:
        for role in ("dspark_context", "dspark_context_decode", "dspark_draft_decode", "dspark_markov"):
            path = write_graph(root, role, [5, 6])
            draft[role] = {"path": path, "type": "hmonnx"}
        metadata["dspark"] = {"components": draft}
    expected = {role: evaluate(root / path) for role, path in graphs.items()}
    original_draft = {p.relative_to(root): p.read_bytes() for role in draft
                      for p in (root / role).iterdir()}
    (root / "golden_meta_info.json").write_text(json.dumps(metadata))
    assert normalize_export(stage, None, None) == "hmquant_tiny"
    updated = json.loads((stage / "golden_meta_info.json").read_text())
    assert updated.get("dspark") == metadata.get("dspark")
    assert not (stage / "weights.bin").exists()
    for role in graphs:
        assert not (stage / role / "original.bin").exists()
    # tar stores one payload and restores the two directory entries as hardlinks.
    archive = tmp_path / "model.tar"
    with tarfile.open(archive, "w") as out:
        out.add(stage, arcname="model")
    shutil.rmtree(stage)
    with tarfile.open(archive) as inp:
        assert inp.getmember("model/prefill/weights.bin").islnk() or inp.getmember("model/decode/weights.bin").islnk()
        inp.extractall(tmp_path / "relocated", filter="data")
    moved = tmp_path / "relocated/model"
    assert (moved / "prefill/weights.bin").samefile(moved / "decode/weights.bin")
    for role in graphs:
        assert updated[role + "_onnx"] == updated[role + "_hmonnx"]
        path = moved / updated[role + "_hmonnx"]
        assert hashlib.md5(path.read_bytes()).hexdigest() == updated[role + "_hmonnx_md5"]
        np.testing.assert_array_equal(evaluate(path), expected[role])
    for relative, data in original_draft.items():
        assert (moved / relative).read_bytes() == data
        assert not (moved / relative).samefile(moved / "prefill/weights.bin")


@pytest.mark.parametrize("decode_width", [1, 6])
def test_small_deepseek_hmonnx_hardlinks_preserve_logits_and_cache(tmp_path, decode_width):
    """Real three-layer MoE target; ordinary decode and DSpark verify export."""
    import copy
    import torch
    from test_deepseek_v4_dspark import _case
    from xhquant.api import (HMONNXInference, PrecisionMode, ptq_quantize,
                            to_export_hmonnx_from_quanted_graph, to_frontend_graph, to_quant_graph)
    from xhmodel_merak.model_integrations.shared_weights import compact_language_weights
    from xhmodel_merak.xh_llm.models.deepseek_v4.cache_abi import DeepSeekV4CacheABI
    from xhmodel_merak.xh_llm.models.deepseek_v4.full_model import StaticDeepSeekV4ForCausalLM
    from xhmodel_merak.xh_llm.models.deepseek_v4.runtime import DeepSeekV4CacheMixin, DeepSeekV4DataPreprocess
    from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import DeepSeekV4StaticCacheSpec

    target, spec, *_ = _case()
    metadata, cases = {}, {}
    for role, width in (("prefill", 8), ("decode", decode_width)):
        model = StaticDeepSeekV4ForCausalLM.from_hf(
            copy.deepcopy(target), input_sequence_length=width, max_context_length=64,
            auxiliary_layer_ids=(0, 1, 2) if decode_width == 6 else (),
            return_all_logits=decode_width == 6,
            swa_backing_length=spec.cache_backing_length).half()
        cache_spec = DeepSeekV4StaticCacheSpec(
            max_context_length=64, prefill_chunk_length=8, decode_sequence_length=decode_width,
            sliding_window=16, latent_head_dim=128, index_topk=4)
        cache = DeepSeekV4CacheMixin(DeepSeekV4CacheABI(spec=cache_spec, layer_types=("sliding_attention",) * 3))
        processor = DeepSeekV4DataPreprocess(token_embedding=target.model.embed_tokens,
                                            input_sequence_length=width, cache_mixin=cache)
        args = processor({"input_ids": torch.arange(1, min(width, 5) + 1).reshape(1, -1), "past_seq_length": 0})
        graph = to_quant_graph(to_frontend_graph(model, "TorchFX", args), "XH2a", spec.quant_scheme(8))
        flat = list(args[:-1]) + list(args[-1])
        ptq_quantize(graph, [flat], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
        cache.reset_kv_cache()
        directory = tmp_path / role
        directory.mkdir()
        path = directory / "model.onnx"
        to_export_hmonnx_from_quanted_graph(graph, flat, str(path), save_as_external_data=True)
        cache.reset_kv_cache()
        session = HMONNXInference(str(path)).to("cpu")
        output = session(*flat)
        values = tuple(output.values()) if isinstance(output, dict) else output
        cases[role] = (cache, flat, [v.clone() for v in values])
        metadata[role + "_hmonnx"] = path.relative_to(tmp_path).as_posix()
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2
    from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader

    loader = MultiHMONNXLoader({role: str(tmp_path / metadata[role + "_hmonnx"]) for role in cases})
    v2_expected = {}
    for role, (cache, flat, _) in cases.items():
        cache.reset_kv_cache()
        path = str(tmp_path / metadata[role + "_hmonnx"])
        # Match the CPU quantization and inputs even on GPU-equipped CI workers.
        runtime = HMONNXInferenceV2.from_onnx_graph(
            path, loader.graphs[role], HMONNXInferenceConfig(exec_devices=["cpu"]))
        output = runtime(*flat)
        values = tuple(output.values()) if isinstance(output, dict) else output
        v2_expected[role] = [v.clone() for v in values]
    (tmp_path / "golden_meta_info.json").write_text(json.dumps(metadata))
    report = compact_language_weights(tmp_path)
    assert report["reused_tensors"] > 0
    moved = tmp_path / "moved"
    moved.mkdir()
    for role in cases:
        (tmp_path / role).rename(moved / role)
    for role, (cache, flat, expected) in cases.items():
        cache.reset_kv_cache()
        session = HMONNXInference(str(moved / report["graphs"][role])).to("cpu")
        output = session(*flat)
        values = tuple(output.values()) if isinstance(output, dict) else output
        for actual, reference in zip(values, expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    loader = MultiHMONNXLoader({role: str(moved / report["graphs"][role]) for role in cases})
    assert loader.initializer_pool.summary()["shared_saved_nbytes"] > 0
    for role, (cache, flat, _) in cases.items():
        cache.reset_kv_cache()
        runtime = HMONNXInferenceV2.from_onnx_graph(
            str(moved / report["graphs"][role]), loader.graphs[role],
            HMONNXInferenceConfig(exec_devices=["cpu"]))
        output = runtime(*flat)
        values = tuple(output.values()) if isinstance(output, dict) else output
        for actual, reference in zip(values, v2_expected[role], strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
