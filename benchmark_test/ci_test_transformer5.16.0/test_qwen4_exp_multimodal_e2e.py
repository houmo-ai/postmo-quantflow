"""Same-checkpoint Qwen4 vision/text HMONNX numerical and request contracts."""

from __future__ import annotations

import inspect
import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration, Qwen4ExpVisionConfig


QUANT = {"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True}
_NATIVE_INITIALIZERS = {name: getattr(torch.nn.init, name) for name in ("kaiming_uniform_", "uniform_", "normal_")}


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    from transformers.models.qwen4_exp import modeling_qwen4_exp

    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ENABLE_HMINFERENCE_V2", "1")
        # HF's optional FLA/causal-conv packages may be installed in GPU CI.
        # This CPU oracle uses the HF functions' actual PyTorch implementations.
        for name in (
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
            "causal_conv1d_fn",
            "causal_conv1d_update",
        ):
            implementation = inspect.unwrap(getattr(modeling_qwen4_exp, name))
            assert implementation.__module__ == modeling_qwen4_exp.__name__
            patch.setattr(modeling_qwen4_exp, name, implementation)
        yield
    torch.set_num_threads(previous)


def _save_processor(directory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

    tokens = [
        "<pad>",
        "<eos>",
        "<unk>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "describe",
    ]
    vocabulary = {token: index for index, token in enumerate(tokens)}
    vocabulary.update({f"t{index}": index for index in range(len(tokens), 32)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        additional_special_tokens=tokens[3:7],
        model_input_names=["input_ids", "attention_mask"],
    )
    template = (
        "{% for message in messages %}{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
        "{% elif item['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
        "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endfor %} describe"
    )
    geometry = dict(patch_size=2, temporal_patch_size=2, merge_size=2)
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            **geometry,
            min_pixels=16,
            max_pixels=64,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        ),
        video_processor=Qwen3VLVideoProcessor(**geometry, size={"shortest_edge": 16, "longest_edge": 64}),
        tokenizer=tokenizer,
        chat_template=template,
    )
    processor.save_pretrained(directory)
    return processor


@dataclass
class _Source:
    path: Path
    native: Qwen4ExpForConditionalGeneration
    processor: object
    assistant: object = None


@contextmanager
def _native_initialization():
    import transformers.modeling_utils as modeling_utils

    # A previous packed-checkpoint load can disable initialization process-wide.
    # Scope the real initializers to construction and preserve the caller's state.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(modeling_utils, "_init_weights", True, raising=False)
        for name, initializer in _NATIVE_INITIALIZERS.items():
            patch.setattr(torch.nn.init, name, initializer)
        yield


@_native_initialization()
def _make_checkpoint(path, *, mtp=False):
    from test_qwen4_exp_workflow_quant_adapter import _tiny_stateful_model

    torch.manual_seed(54317)
    text = _tiny_stateful_model().config
    text.max_position_embeddings = 64
    text.eos_token_id, text.pad_token_id = 1, 0
    # All T/H/W axes must affect real rotary channels in this small fixture.
    text.rope_parameters.update(partial_rotary_factor=1.0, mrope_section=[2, 1, 1])
    assistant = None
    if mtp:
        text.mtp = {"num_hidden_layers": 1, "layer_types": ["full_attention"]}
    visual = Qwen4ExpVisionConfig(
        depth=1,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        out_hidden_size=16,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
    )
    config = Qwen4ExpConfig(
        text_config=text.to_dict(),
        vision_config=visual.to_dict(),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    config._attn_implementation = "eager"
    native = Qwen4ExpForConditionalGeneration(config).eval().half()
    native.save_pretrained(path)
    if mtp:
        from safetensors import safe_open
        from safetensors.torch import save_file

        from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_model import Qwen4ExpMTPModel

        assistant = Qwen4ExpMTPModel(text).eval().half()
        save_file(
            {"mtp." + name: value.contiguous() for name, value in assistant.state_dict().items()},
            path / "mtp.safetensors",
        )
        with safe_open(path / "model.safetensors", framework="pt") as reader:
            mapping = {name: "model.safetensors" for name in reader.keys()}
        mapping.update({"mtp." + name: "mtp.safetensors" for name in assistant.state_dict()})
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return _Source(path, native, _save_processor(path), assistant)


def _write_video(path):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width, stream.height, stream.pix_fmt = 12, 8, "yuv420p"
        for color in (30, 120, 210):
            frame = av.VideoFrame.from_ndarray(np.full((8, 12, 3), color, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _processor_inputs(processor, kind="image", *, video_path=None):
    from transformers.video_utils import VideoMetadata

    pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    image = Image.fromarray(pixels if kind == "image_video" else pixels[:, :4])
    content, options = [], {}
    if kind in {"image", "two_images", "image_video"}:
        content.append({"type": "image", "image": image})
    if kind == "two_images":
        content.append({"type": "image", "image": Image.fromarray(255 - pixels[:4])})
    if kind in {"video", "image_video"}:
        frames = np.stack([pixels[:4, :4], 255 - pixels[:4, :4]] * 2)
        content.append({"type": "video", "video": frames})
        options = {
            "do_sample_frames": False,
            "video_metadata": [VideoMetadata(total_num_frames=4, fps=2, frames_indices=[0, 1, 2, 3])],
        }
    if kind == "video_file":
        assert video_path is not None
        content.append({"type": "video", "video": str(video_path)})
    content.append({"type": "text", "text": "describe"})
    result = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=options,
    )
    result.pop("video_metadata", None)
    assert result["input_ids"].shape[1] + 4 < 64
    assert not result["input_ids"].eq(0).any()
    return result


def _positions(native, inputs):
    return native.model.get_rope_index(
        inputs["input_ids"],
        inputs["mm_token_type_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        attention_mask=inputs.get("attention_mask"),
    )[0]


def _native_inputs(inputs):
    return {name: value.half() if value.is_floating_point() else value for name, value in inputs.items()}


def test_native_checkpoint_isolates_loader_initialization_guard(monkeypatch, tmp_path):
    import transformers.modeling_utils as modeling_utils

    expected = _make_checkpoint(tmp_path / "expected", mtp=True)

    def skip(*args, **kwargs):
        return None

    monkeypatch.setattr(modeling_utils, "_init_weights", False, raising=False)
    for name in _NATIVE_INITIALIZERS:
        monkeypatch.setattr(torch.nn.init, name, skip)
    actual = _make_checkpoint(tmp_path / "guarded", mtp=True)
    for a, b in ((actual.native, expected.native), (actual.assistant, expected.assistant)):
        assert a.state_dict().keys() == b.state_dict().keys()
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0, msg=name)
    assert modeling_utils._init_weights is False
    assert all(getattr(torch.nn.init, name) is skip for name in _NATIVE_INITIALIZERS)


def _assert_numerical(actual, expected, *, cosine=0.99, atol=3e-3):
    assert actual.shape == expected.shape
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert torch.isfinite(a).all() and torch.count_nonzero(b)
    similarity = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    assert similarity > cosine, f"cosine={similarity.item()}, max_abs={(a - b).abs().max().item()}"
    torch.testing.assert_close(a, b, rtol=5e-2, atol=atol)


def _export_package(root, *, mtp):
    source = _make_checkpoint(root / "checkpoint", mtp=mtp)
    visual = {
        "enable": True,
        "image_token_gears": [2, 4],
        "quant_scheme": {**QUANT, "quant_type": "w8a16h1_sefp" if mtp else "w8a8h1_sefp"},
        "dtype": "float16",
    }
    config = dict(
        hf_model=str(source.path),
        model_name="qwen4_multimodal_tiny",
        context_max_length=64,
        prefill_chunk_length=4,
        quant_scheme=QUANT,
        quant_device="cpu",
        dtype="float16",
        use_fused_gdr=True,
        ple_quant_type="fp16",
        visual_config=visual,
    )
    output = root / "export"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HUGE_MODEL_EXPORT_ENABLED", "0")
        patch.setenv("TORCH_COMPILE_DISABLE", "0")
        if mtp:
            from xhmodel_merak.xh_llm.models.qwen4_exp.serving_workflow import export_qwen4_exp_serving_package

            shared_head = {**QUANT, "nodes": {"lm_head": {"quant_type": "w8a16h1_sefp"}}}
            config.update(
                spec_decode_mode="mtp",
                quant_scheme=shared_head,
                mtp_config={"quant_scheme": shared_head},
            )
            export_qwen4_exp_serving_package(
                model_dir=source.path,
                output_dir=output,
                model_config=config,
                device="cpu",
                low_memory=False,
            )
        else:
            from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig
            from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_model import XHQwen4ExpModel

            XHQwen4ExpModel(XHQwen4ExpModelConfig(**config)).export_hmonnx(str(output))
    meta = json.loads((output / "golden_meta_info.json").read_text())
    assert meta["qwen4_exp_vision"]["abi_version"] == 2
    assert meta["qwen4_exp_vision"]["layout"] == "components"
    assert all(len(gear["components"]) == 3 for gear in meta["qwen4_exp_vision"]["gears"])
    import onnx

    for gear in meta["qwen4_exp_vision"]["gears"]:
        for part in gear["components"]:
            graph = onnx.load(output / part["hmonnx"], load_external_data=False).graph
            linears = [node for node in graph.node if node.op_type == "Linear"]
            assert linears, part["name"]
            for node in linears:
                attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
                assert attrs["hmfp_weight_man_bit"] == 8
                assert attrs["hmfp_act_man_bit"] == (16 if mtp else 8)
    moved = root / "relocated"
    output.rename(moved)
    shutil.rmtree(source.path)
    assert not output.exists() and not source.path.exists()
    return moved, source


@pytest.fixture(scope="module")
def ordinary_package(tmp_path_factory):
    return _export_package(tmp_path_factory.mktemp("qwen4_mm_export"), mtp=False)


def _load_package(root):
    from xhmodel_merak.xh_llm.hmonnx.auto_llm_hmonnx_model import AutoLLMHONNXModel

    return AutoLLMHONNXModel.from_pretrained(str(root / "golden_meta_info.json"), device_map=["cpu"])


def _fused_and_positions(model, source, inputs):
    from xhmodel_merak.xh_llm.models.qwen4_exp.multimodal import fuse_multimodal_embeddings, multimodal_rope_positions

    native = source.native
    for pixels_name, grid_name, method, expected_fn in (
        ("pixel_values", "image_grid_thw", model.vision_encoder.encode_many, native.get_image_features),
        ("pixel_values_videos", "video_grid_thw", model.vision_encoder.encode_videos, native.get_video_features),
    ):
        if pixels_name not in inputs:
            continue
        actual = method(inputs[pixels_name], inputs[grid_name])
        expected = expected_fn(inputs[pixels_name].half(), inputs[grid_name]).pooler_output
        if pixels_name == "pixel_values":
            assert all(value.ndim == 3 and value.shape[0] == 1 for value in actual)
            actual = torch.cat(actual, dim=1).squeeze(0)
        else:
            assert all(value.ndim == 2 for value in actual)
            actual = torch.cat(actual, dim=0)
        _assert_numerical(actual, torch.cat(expected, dim=0))
    positions = _positions(native, inputs)
    actual_positions, _ = multimodal_rope_positions(
        native.config,
        inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        mm_token_type_ids=inputs["mm_token_type_ids"],
        attention_mask=inputs.get("attention_mask"),
    )
    torch.testing.assert_close(actual_positions, positions, rtol=0, atol=0)
    fused = fuse_multimodal_embeddings(
        inputs["input_ids"],
        model.embed_tokens,
        model.vision_encoder,
        native.config,
        **{
            key: inputs[key]
            for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
            if key in inputs
        },
    )
    return fused, positions


def _prefill(model, ids, fused, positions):
    from xhmodel_merak.xh_llm.models.qwen4_exp.generation import _allocate_page_caches, _page_step

    model.reset_internal_state()
    model.set_prefill()
    caches = _allocate_page_caches(model)
    hidden = []
    width = model.get_input_sequence_length()
    for start in range(0, ids.shape[1], width):
        logits = _page_step(
            model,
            caches,
            ids[:, start : start + width],
            start,
            inputs_embeds=fused[:, start : start + width],
            position_ids=positions[:, :, start : start + width],
        )
        if getattr(model, "last_hidden_states", None) is not None:
            hidden.append(model.last_hidden_states[:, : min(width, ids.shape[1] - start)].clone())
    return logits, caches, torch.cat(hidden, dim=1) if hidden else None


def _assert_actual_capture(capture, *, all_visual_gears):
    capture.finish()
    manifest = json.loads(capture.manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert {step["component"] for step in manifest["steps"]} == {"prefill", "decode", "visual"}
    visual = {step["graph"] for step in manifest["steps"] if step["component"] == "visual"}
    capacities = (2, 4) if all_visual_gears else (2,)
    assert visual == {
        f"visual_m{capacity}_{name}" for capacity in capacities for name in ("stem", "block_00", "merger")
    }
    for step in manifest["steps"]:
        directory = capture.root / step["directory"]
        for entry in (*step["inputs"], *step["outputs"], *step["runtime_before"], *step["runtime_after"]):
            array = np.load(directory / entry["file"], allow_pickle=False, mmap_mode="r")
            assert list(array.shape) == entry["shape"]
            assert str(array.dtype) == entry["dtype"]
        # SDK node intermediates must be present as well as graph boundaries.
        boundary_files = {entry["file"] for entry in (*step["inputs"], *step["outputs"])}
        assert any(
            path.name not in boundary_files and not path.name.startswith("runtime_") for path in directory.glob("*.npy")
        )


@pytest.mark.parametrize("kind", ["image", "two_images", "video", "image_video", "video_file"])
@torch.inference_mode()
def test_real_multimodal_export_relocate_native_numerical_and_generate(ordinary_package, kind, tmp_path):
    from xhmodel_merak.xh_llm.models.qwen4_exp.generation import _page_step

    root, source = ordinary_package
    model = _load_package(root)
    processor = model.get_processor()
    assert processor is model.get_processor()
    video_path = _write_video(tmp_path / "short.mp4") if kind == "video_file" else None
    inputs = _processor_inputs(processor, kind, video_path=video_path)
    fused, positions = _fused_and_positions(model, source, inputs)
    assert positions.shape == (3, 1, inputs["input_ids"].shape[1])
    if "image_grid_thw" in inputs:
        assert not torch.equal(positions[0], positions[1])
    actual, caches, _ = _prefill(model, inputs["input_ids"], fused, positions)
    reference = source.native(**_native_inputs(inputs), use_cache=True)
    pixels_name = "pixel_values" if "pixel_values" in inputs else "pixel_values_videos"
    changed = {**inputs, pixels_name: -inputs[pixels_name]}
    altered = source.native(**_native_inputs(changed), use_cache=False).logits
    assert torch.isfinite(reference.logits).all()
    assert (reference.logits - altered).abs().max() > 1e-5
    _assert_numerical(actual, reference.logits[:, -1], cosine=0.98, atol=1e-2)
    first = actual.argmax(-1).reshape(1, 1)
    next_positions = torch.full((3, 1, 1), int(positions.max()) + 1, dtype=torch.long)
    model.set_decode()
    actual = _page_step(model, caches, first, inputs["input_ids"].shape[1], position_ids=next_positions)
    reference = source.native(
        input_ids=first,
        past_key_values=reference.past_key_values,
        position_ids=next_positions,
        use_cache=True,
    )
    _assert_numerical(actual, reference.logits[:, -1], cosine=0.98, atol=1e-2)
    expected_ids = torch.cat((inputs["input_ids"], first, actual.argmax(-1).reshape(1, 1)), dim=1)
    kwargs = dict(inputs, max_new_tokens=2, do_sample=False, eos_token_id=[])
    capture = None
    if kind in {"image", "image_video"}:
        from xhmodel_merak.xh_llm.models.qwen4_exp.golden_runtime import GoldenCapture

        capture = GoldenCapture(tmp_path / "golden", ["prefill", "decode", "visual"])
        capture.add_ple_resources(model.external_ple)
    generated = model.generate(**kwargs, _capture=capture)
    if capture is not None:
        _assert_actual_capture(capture, all_visual_gears=kind == "image_video")
        assert model.vision_encoder.capture is None
    torch.testing.assert_close(generated, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(model.generate(**kwargs), generated, rtol=0, atol=0)
    grid_name = "image_grid_thw" if "image_grid_thw" in inputs else "video_grid_thw"
    bad = inputs[grid_name].clone()
    bad[0, 2] += 2
    with pytest.raises(ValueError, match="grid|patch|image|video|placeholder|feature"):
        model.generate(**{**kwargs, grid_name: bad})
    torch.testing.assert_close(model.generate(**kwargs), generated, rtol=0, atol=0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        model.generate(**kwargs, inputs_embeds=fused)


def _native_assistant_step(source, embedding, target_hidden, positions, cache):
    """Use the unrewritten assistant's native HF blocks and rotary/mask math."""
    from transformers.masking_utils import create_causal_mask

    assistant = source.assistant
    mask = create_causal_mask(
        config=assistant.config,
        inputs_embeds=embedding,
        attention_mask=None,
        past_key_values=cache,
        position_ids=None,
        allow_is_causal_skip=False,
    )
    previous = getattr(cache, "position_ids", None)
    full_positions = positions if previous is None else torch.cat((previous, positions), dim=-1)
    cache.position_ids = full_positions
    rotary = assistant.rotary_emb(embedding, full_positions)
    hidden = assistant.fuse(embedding, target_hidden)
    for layer in assistant.layers:
        hidden = layer(hidden, position_embeddings=rotary, attention_mask=mask, conv_mask=None, past_key_values=cache)
    return source.native.lm_head(assistant.hyper_connection_mixer(hidden)), hidden
