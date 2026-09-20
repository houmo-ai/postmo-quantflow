"""YAML -> adapter -> producer recipe/CLI contract; does not rerun quantization."""
import json

import pytest
import yaml

from test_transformer516_recipe_contract import (
    REPO_ROOT, _assert_command_matches_parser, _modelzoo_adapters, _require_callable,
)


def test_qwen38_hessian_mode_reaches_producer(tmp_path):
    config = yaml.safe_load((REPO_ROOT / 'configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/'
                            'qwen3_8_27b_full_xh_hessian.yaml').read_text())
    qwen, _ = _modelzoo_adapters()
    kwargs = qwen.build_qwen35_autoround_kwargs(
        hf_model_dir='unused', output_dir=str(tmp_path / 'out'), device='cuda:0',
        quant_cfg=config['quant'], export_model_cfg=config['export']['model'], workflow_seed=42,
    )
    assert kwargs['data_type'] == 'int_sym_xh_hessian'
    assert 'optimization_mode' not in kwargs and 'xh_hessian_config' not in kwargs
    result = _require_callable('qwen35_autoround', 'quantize_qwen35_autoround')(**kwargs, dry_run=True)
    command = result.provenance['command']
    _assert_command_matches_parser(command)
    assert command[command.index('--data_type') + 1] == 'int_sym_xh_hessian'
    assert command[command.index('--optimization_mode') + 1] == 'xh_hessian'
    resolved = json.loads(command[command.index('--xh_hessian_config') + 1])
    assert resolved['validation_samples'] == 16 and resolved['candidates'] == 33
    original = yaml.safe_load((REPO_ROOT / 'configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/'
                              'qwen3_8_27b_full.yaml').read_text())
    original['quant']['data_type'] = 'int_sym_xh_hessian'
    # Solver selection changes only the dtype; export settings are independent.
    assert config['quant'] == original['quant']


def test_original_qwen38_config_remains_standard(tmp_path):
    config = yaml.safe_load((REPO_ROOT / 'configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/'
                            'qwen3_8_27b_full.yaml').read_text())
    qwen, _ = _modelzoo_adapters()
    kwargs = qwen.build_qwen35_autoround_kwargs(
        hf_model_dir='unused', output_dir=str(tmp_path / 'out'), device='cuda:0',
        quant_cfg=config['quant'], export_model_cfg=config['export']['model'], workflow_seed=42,
    )
    assert 'optimization_mode' not in kwargs and 'xh_hessian_config' not in kwargs


@pytest.mark.parametrize('legacy_field', ['optimization_mode', 'xh_hessian_config'])
def test_backend_solver_options_are_not_silently_ignored_in_workflow(tmp_path, legacy_field):
    qwen, _ = _modelzoo_adapters()
    with pytest.raises(ValueError, match='quant.data_type=int_sym_xh_hessian'):
        qwen.build_qwen35_autoround_kwargs(
            hf_model_dir='unused', output_dir=str(tmp_path / 'out'), device='cuda:0',
            quant_cfg={legacy_field: {}}, export_model_cfg={}, workflow_seed=42,
        )
