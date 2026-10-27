"""Five-gear export, smallest-fit routing and query chunking with native ViTs."""

from copy import deepcopy
import json
from pathlib import Path

from _minicpm_v47_fixture import minicpmv47_modeling  # noqa: F401

import pytest
import torch

from test_minicpm_dynamic_vision import _native_features, _native_parts, _run_vision
from xhmodel_merak.xh_llm.minicpm_v_series.token_capacity_vision import (
    MiniCPMTokenCapacityVisionExportModel, prepare_token_capacity_vision_inputs,
)
from xhmodel_merak.xh_llm.minicpm_v_series.token_capacity_vision_export import (
    _convert_hmonnx, _hmonnx_inputs, vision_input_names,
)
from xhmodel_merak.xh_llm.minicpm_v_series.vision_gears import (
    DEFAULT_PATCH_TOKEN_GEARS, image_token_gears_from_patches,
    factor_patch_grid, normalize_image_token_gears,
    output_token_divisor, select_vision_profile, vision_profile_entries,
)


def _metadata(mode, gears=None):
    if gears is None:
        gears = image_token_gears_from_patches(DEFAULT_PATCH_TOKEN_GEARS, mode)
    divisor = output_token_divisor(mode)
    return {
        'schema_version': 2, 'downsample_mode': mode, 'token_unit': 'post_spatial_merge',
        'image_token_gears': list(gears), 'routing_policy': 'smallest_fit', 'overflow_policy': 'reject',
        'gears': [dict(downsample_mode=mode, image_token_capacity=g, token_capacity=g * divisor,
                       hmonnx=f'visual_{mode}_m{g}/model.onnx', positions_per_side=8) for g in gears],
    }


@pytest.mark.parametrize('mode', ('4x', '16x'))
def test_default_gears_cover_equal_input_capacities_and_smallest_fit(mode):
    profile = _metadata(mode)
    divisor = output_token_divisor(mode)
    assert [e['token_capacity'] for e in vision_profile_entries(profile, mode)] == [384, 784, 1536, 2816, 6144]
    assert profile['image_token_gears'] == (
        [96, 196, 384, 704, 1536] if mode == '4x' else [24, 49, 96, 176, 384]
    )
    previous = 0
    for gear in profile['image_token_gears']:
        for count in (previous + 1, gear):
            grid = factor_patch_grid(count * divisor, mode)
            assert grid[0] * grid[1] == count * divisor
            assert select_vision_profile(profile, mode, grid)['image_token_capacity'] == gear
        previous = gear
    with pytest.raises(ValueError, match='exceeding capacity'):
        select_vision_profile(profile, mode, factor_patch_grid((previous + 1) * divisor, mode))
    with pytest.raises(ValueError, match='divisible'):
        select_vision_profile(profile, mode, (3, 3))
    legacy = dict(profile['gears'][2])
    legacy.pop('image_token_capacity')
    assert select_vision_profile(legacy, mode, factor_patch_grid(divisor, mode)) is legacy


@pytest.mark.parametrize('value', ([], [0], [-1], [True], [1.5], ['96']))
def test_gears_reject_invalid_capacities(value):
    with pytest.raises(ValueError, match='positive integers'):
        normalize_image_token_gears(value)


@pytest.mark.parametrize('mode', ('4x', '16x'))
def test_shared_patch_gears_require_alignment_for_both_modes(mode):
    with pytest.raises(ValueError, match='divisible by 16'):
        image_token_gears_from_patches([384, 788], mode)


@pytest.mark.parametrize('field,value', (
    ('token_unit', 'input_patch'), ('routing_policy', 'nearest'), ('overflow_policy', 'truncate'),
    ('image_token_gears', [96, 196]), ('schema_version', 1),
))
def test_gear_metadata_rejects_inconsistent_contracts(field, value):
    profile = _metadata('16x')
    profile[field] = value
    with pytest.raises(ValueError):
        vision_profile_entries(profile, '16x')


@pytest.mark.parametrize('mode', ('4x', '16x'))
def test_chunked_attention_retains_all_keys_and_padding_mask_in_hmonnx(tmp_path, mode):
    from xhquant.api import HMONNXInference

    torch.set_num_threads(2)
    vision, merger = _native_parts('4_7')
    wrapper = MiniCPMTokenCapacityVisionExportModel(deepcopy(vision), deepcopy(merger), 64, mode)
    wrapper.window_chunk_size = 3  # Exercise a partial final batch of local windows.
    for layer in wrapper.encoder_layers:
        layer.self_attn.query_chunk_size = 8
    grid = (4, 8)
    pixels = torch.randn(1, 3, 14, 32 * 14)
    names = vision_input_names(mode)
    inputs = prepare_token_capacity_vision_inputs(pixels, grid, 64, mode, 8)[:len(names)]
    with torch.inference_mode():
        native = _native_features(vision, merger, pixels, grid, mode)
        floating = wrapper(*inputs)[:, :native.shape[1]]
        torch.testing.assert_close(floating, native, atol=1e-5, rtol=1e-5)
    graph = tmp_path / 'chunked.onnx'
    _convert_hmonnx(wrapper, graph, inputs, names, 'XH2a', 'w8a8h1_sefp')
    session = HMONNXInference(str(graph))
    session.to('cpu')
    session.exec_device = torch.device('cpu')
    probabilities = []
    for module in session.node_modules:
        if type(module).__name__ == 'Softmax':
            module.register_forward_hook(lambda m, a, o: probabilities.append(next(iter(o.values())).detach()))
    with torch.inference_mode():
        output = _run_vision(session, _hmonnx_inputs(inputs, names))[:, :native.shape[1]]
    assert torch.nn.functional.cosine_similarity(output.float().flatten(), native.flatten(), dim=0) > .98
    global_chunks = [p for p in probabilities if p.shape[-1] == 64]
    assert len(global_chunks) >= 8
    assert all(p.shape[-2] <= 8 and p[..., 32:].count_nonzero() == 0 for p in global_chunks)


@pytest.fixture(scope='module')
def exported_gears(tmp_path_factory, minicpmv47_modeling):
    MiniCPMV4_7Config = minicpmv47_modeling.MiniCPMV4_7Config
    MiniCPMV4_7ForConditionalGeneration = minicpmv47_modeling.MiniCPMV4_7ForConditionalGeneration
    from xhmodel_merak.workflows import AutoWorkflow

    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp('minicpm_five_gears')
    vision, _ = _native_parts('4_7')
    config = MiniCPMV4_7Config(
        vision_config=vision.config, insert_layer_id=0,
        text_config=dict(model_type='qwen3_5_text', hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=1, num_key_value_heads=1,
                         head_dim=64, vocab_size=64, layer_types=['full_attention'] * 2),
    )
    model = MiniCPMV4_7ForConditionalGeneration(config).float().eval()
    source = root / 'source'
    model.save_pretrained(source)
    repo = Path(__file__).resolve().parents[2]
    workflow = AutoWorkflow.from_config(str(source), str(repo / (
        'configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/minicpm_v_4_7_xh2a_w8a8.yaml')))
    overrides = {'export.components.llm': False, 'export.vision.patch_token_gears': [16, 32, 48, 64, 128],
                 'export.components.vision_4x.validation_target': [4, 4],
                 'export.components.vision_16x.validation_target': [4, 4]}
    quant = workflow.quant(str(root / 'quant'), 'cpu', overrides)
    result = workflow.export(quant, str(root / 'export'), 'cpu', overrides)
    return model, result


@pytest.mark.parametrize('mode', ('4x', '16x'))
def test_real_five_graph_export_routes_mixed_slices_in_original_order(exported_gears, mode):
    from xhmodel_merak.model_integrations.minicpm_v_4_7.artifact import describe_export, validate_runtime_contract
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.inference import MiniCPMV47HMONNXRuntime

    model, result = exported_gears
    root = Path(result.work_dir)
    meta = json.loads((root / 'export_meta_info.json').read_text())
    entries = describe_export(meta)
    assert sum(entry['type'] == 'hmonnx' for entry in entries.values()) == 10
    assert sum(entry['type'] == 'json' for entry in entries.values()) == 2

    def resolve_file(value, label):
        path = root / value
        assert path.is_file(), label
        return path
    validate_runtime_contract(meta, components=meta['components'], resolve_file=resolve_file)
    runtime = MiniCPMV47HMONNXRuntime(root / 'export_meta_info.json', device='cpu')
    divisor = output_token_divisor(mode)
    scale = 4 if mode == '4x' else 1
    counts = [scale * count for count in (5, 1, 4, 2, 3)]
    grids = [factor_patch_grid(count * divisor, mode) for count in counts]
    pixels = [torch.randn(1, 3, 14, count * divisor * 14) for count in counts]
    with torch.inference_mode():
        expected = torch.cat([
            _native_features(model.model.vision_tower, model.model.merger, p, grid, mode).squeeze(0)
            for p, grid in zip(pixels, grids, strict=True)
        ])
        actual = runtime._extract_features(torch.cat(pixels, dim=-1), torch.tensor(grids), mode)
    assert actual.shape == (sum(counts), 64)
    offset = 0
    for count in counts:
        cosine = torch.nn.functional.cosine_similarity(
            actual[offset:offset + count].float().flatten(), expected[offset:offset + count].flatten(), dim=0)
        assert cosine > .98
        offset += count
    assert set(runtime._vision_sessions) == {(mode, capacity) for capacity in (16, 32, 48, 64, 128)}
    runtime._release_vision_session(mode)
    assert not runtime._vision_sessions


def test_every_gear_can_generate_golden_without_being_selected_by_prompt(exported_gears):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.inference import MiniCPMV47HMONNXRuntime

    _, result = exported_gears
    root = Path(result.work_dir)
    runtime = MiniCPMV47HMONNXRuntime(root / 'export_meta_info.json', device='cpu', enable_golden=True)
    with torch.inference_mode():
        runtime.run_all_vision_gears()
    for mode, profile in runtime.meta['vision'].items():
        for entry in profile['gears']:
            graph = root / entry['hmonnx']
            assert (graph.parent / 'step_0').is_dir()
    assert len(runtime._vision_golden_steps) == 10
    assert not runtime._vision_sessions
