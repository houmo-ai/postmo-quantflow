"""LoRA child exports must be directly recognizable by vLLM-Merak."""
import json

import pytest

from xhmodel_merak.xh_llm.models.qwen3_5.workflow import finalize_qwen35_merak_runtime_config
from xhmodel_merak.xh_llm.workflows.result import ExportResult


def test_lora_views_receive_their_own_runtime_entry(tmp_path):
    root = tmp_path / 'hmquant_qwen36'
    root.mkdir()
    original = {'model_config': {'flash_attention': {'enable': True}}, 'lora_adapters': []}
    for step, enabled in [(4000, False), (4500, True)]:
        child = root / 'lora' / f'checkpoint-{step}'
        child.mkdir(parents=True)
        meta = {'model_config': {'flash_attention': {'enable': enabled}},
                'active_lora': {'name': f'checkpoint-{step}'}}
        path = child / 'golden_meta_info.json'
        path.write_text(json.dumps(meta))
        original['lora_adapters'].append({'meta_file': str(path.relative_to(root))})
    (root / 'golden_meta_info.json').write_text(json.dumps(original))
    returned = finalize_qwen35_merak_runtime_config(ExportResult(str(tmp_path), 'unused.yaml'))
    assert returned == root / 'merak_config.json'
    for directory, enabled in [(root, True), (root/'lora/checkpoint-4000', False), (root/'lora/checkpoint-4500', True)]:
        config = json.loads((directory / 'merak_config.json').read_text())
        assert config['model_type'] == 'merak_llm'
        assert config['architectures'] == ['MerakForCausalLM']
        assert config['xh_model'] == {'model_type': 'hmonnx', 'meta_info': 'golden_meta_info.json'}
        assert config['enable_page_attention'] is enabled
        assert (directory / config['xh_model']['meta_info']).is_file()
    assert json.loads((root / 'golden_meta_info.json').read_text()) == original


@pytest.mark.parametrize('child_path', ['golden_meta_info.json', '../outside.json'])
def test_lora_entry_rejects_non_child_metadata(tmp_path, child_path):
    root = tmp_path / 'hmquant_qwen36'
    root.mkdir()
    (root / 'golden_meta_info.json').write_text(json.dumps({
        'model_config': {}, 'lora_adapters': [{'meta_file': child_path}],
    }))
    with pytest.raises(ValueError, match='child'):
        finalize_qwen35_merak_runtime_config(ExportResult(str(tmp_path), 'unused.yaml'))
