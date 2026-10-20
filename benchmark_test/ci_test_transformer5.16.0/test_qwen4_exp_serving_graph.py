"""Qwen4 static layout and rotary index graph normalization."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def _static_layout_fixture():
    nodes = [
        helper.make_node('Reshape', ['x', 'flat_shape'], ['flat'], name='flatten'),
        helper.make_node('Reshape', ['flat', 'shape'], ['back'], name='inverse'),
        helper.make_node('Reshape', ['x', 'shape'], ['same'], name='noop'),
        helper.make_node('Transpose', ['back'], ['transposed'], name='transpose', perm=[0, 2, 1]),
        helper.make_node('Transpose', ['transposed'], ['restored'], name='inverse_transpose', perm=[0, 2, 1]),
        helper.make_node('Cast', ['mask'], ['float_mask'], name='mask_cast', to=TensorProto.FLOAT16),
        helper.make_node('Unsqueeze', ['float_mask', 'axes'], ['shared_mask'], name='shared_mask'),
        helper.make_node('Unsqueeze', ['mask', 'axes'], ['bool_mask'], name='duplicate_mask'),
        helper.make_node('Cast', ['bool_mask'], ['duplicate_float'], name='duplicate_cast', to=TensorProto.FLOAT16),
        helper.make_node('Add', ['restored', 'same'], ['added']),
        helper.make_node('Mul', ['added', 'duplicate_float'], ['masked']),
        helper.make_node('Mul', ['flat', 'flat'], ['flat_squared']),
        helper.make_node('Add', ['masked', 'shared_mask'], ['result']),
    ]
    shapes = {'flat': [1, 8], 'back': [1, 2, 4], 'same': [1, 2, 4], 'transposed': [1, 4, 2],
              'restored': [1, 2, 4], 'float_mask': [1, 2], 'shared_mask': [1, 2, 1],
              'bool_mask': [1, 2, 1], 'duplicate_float': [1, 2, 1], 'added': [1, 2, 4], 'masked': [1, 2, 4]}
    graph = helper.make_graph(
        nodes, 'static_layout',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT16, [1, 2, 4]),
         helper.make_tensor_value_info('mask', TensorProto.BOOL, [1, 2])],
        [helper.make_tensor_value_info('result', TensorProto.FLOAT16, [1, 2, 4]),
         helper.make_tensor_value_info('flat_squared', TensorProto.FLOAT16, [1, 8])],
        initializer=[numpy_helper.from_array(np.array(shape, dtype=np.int64), name)
                     for name, shape in [('shape', [1, 2, 4]), ('flat_shape', [1, 8]), ('axes', [-1])]],
        value_info=[helper.make_tensor_value_info(name, TensorProto.BOOL if name == 'bool_mask'
                                                 else TensorProto.FLOAT16, shape) for name, shape in shapes.items()],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])


def test_static_layout_cleanup_preserves_mask_values_and_shared_branch():
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import simplify_qwen4_static_layouts

    model = _static_layout_fixture()
    inputs = {'x': np.arange(8, dtype=np.float16).reshape(1, 2, 4), 'mask': np.array([[True, False]])}
    before = ReferenceEvaluator(model).run(None, inputs)
    removed = simplify_qwen4_static_layouts(model)
    assert set(removed) == {'inverse', 'noop', 'inverse_transpose', 'transpose', 'duplicate_cast', 'duplicate_mask'}
    assert any(node.name == 'flatten' for node in model.graph.node)  # Still used by flat_squared.
    onnx.checker.check_model(model)
    for actual, expected in zip(ReferenceEvaluator(model).run(None, inputs), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert simplify_qwen4_static_layouts(model) == ()


@pytest.mark.parametrize('boundary', ['unknown_domain', 'bad_shape', 'overridable_shape', 'unknown_attr',
                                     'bad_allowzero', 'public_output', 'cache_alias', 'symbolic_shape'])
def test_static_layout_cleanup_keeps_unproven_reshape(boundary):
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import simplify_qwen4_static_layouts

    model = _static_layout_fixture()
    node = next(node for node in model.graph.node if node.name == 'noop')
    if boundary == 'unknown_domain':
        node.domain = 'unrelated.custom'
    elif boundary == 'bad_shape':
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.array([2, 4], dtype=np.int64), 'shape'))
    elif boundary == 'overridable_shape':
        model.graph.input.append(helper.make_tensor_value_info('shape', TensorProto.INT64, [3]))
    elif boundary == 'unknown_attr':
        node.attribute.append(helper.make_attribute('private_mode', 1))
    elif boundary == 'bad_allowzero':
        node.attribute.append(helper.make_attribute('allowzero', 2))
    elif boundary == 'public_output':
        model.graph.output.append(helper.make_tensor_value_info('same', TensorProto.FLOAT16, [1, 2, 4]))
    elif boundary == 'cache_alias':
        model.graph.node.append(helper.make_node('KVcache', ['same'], ['cache_state'], domain='ai.houmo.xh2a'))
    else:
        info = next(value for value in model.graph.value_info if value.name == 'same')
        info.type.tensor_type.shape.dim[1].dim_param = 'dynamic'
    simplify_qwen4_static_layouts(model)
    assert any(node.name == 'noop' for node in model.graph.node)


def test_static_layout_cleanup_preserves_public_reshape_source_and_unknown_cast_rounding():
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import simplify_qwen4_static_layouts

    model = _static_layout_fixture()
    cast = next(node for node in model.graph.node if node.name == 'duplicate_cast')
    cast.attribute.append(helper.make_attribute('round_mode', 'UNKNOWN'))
    simplify_qwen4_static_layouts(model)
    assert any(node.name == 'duplicate_cast' for node in model.graph.node)
    # A public previous view cannot be removed; do not silently rewire only.
    model = helper.make_model(helper.make_graph(
        [helper.make_node('Reshape', ['x', 'flat_shape'], ['flat'], name='public_flatten'),
         helper.make_node('Reshape', ['flat', 'matrix_shape'], ['matrix'], name='private_matrix'),
         helper.make_node('Add', ['matrix', 'matrix'], ['result'])], 'public_view',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT16, [1, 2, 4])],
        [helper.make_tensor_value_info('flat', TensorProto.FLOAT16, [1, 8]),
         helper.make_tensor_value_info('result', TensorProto.FLOAT16, [2, 4])],
        initializer=[numpy_helper.from_array(np.array(shape, dtype=np.int64), name)
                     for name, shape in [('flat_shape', [1, 8]), ('matrix_shape', [2, 4])]],
        value_info=[helper.make_tensor_value_info('matrix', TensorProto.FLOAT16, [2, 4])],
    ))
    original = model.SerializeToString()
    assert simplify_qwen4_static_layouts(model) == ()
    assert model.SerializeToString() == original


def test_static_layout_file_cleanup_reaches_fixed_point_after_private_branch_is_exposed(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import normalize_qwen4_static_layout_graph

    model = helper.make_model(helper.make_graph(
        [helper.make_node('Reshape', ['x', 'grouped_shape'], ['grouped'], name='reshape'),
         helper.make_node('Transpose', ['grouped'], ['transposed'], name='first_transpose', perm=[0, 2, 1]),
         helper.make_node('Transpose', ['transposed'], ['restored'], name='second_transpose', perm=[0, 2, 1]),
         helper.make_node('Reshape', ['restored', 'original_shape'], ['back'], name='back'),
         helper.make_node('Add', ['back', 'back'], ['result'])], 'fixed_point',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT16, [1, 2, 4])],
        [helper.make_tensor_value_info('result', TensorProto.FLOAT16, [1, 2, 4])],
        initializer=[numpy_helper.from_array(np.array(shape, dtype=np.int64), name)
                     for name, shape in [('original_shape', [1, 2, 4]), ('grouped_shape', [1, 4, 2])]],
        value_info=[helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape)
                    for name, shape in [('grouped', [1, 4, 2]), ('transposed', [1, 2, 4]),
                                        ('restored', [1, 4, 2]), ('back', [1, 2, 4])]],
    ), opset_imports=[helper.make_opsetid('', 18)])
    path = tmp_path / 'fixed_point.onnx'
    onnx.save(model, path)
    data = {'x': np.arange(8, dtype=np.float16).reshape(1, 2, 4)}
    expected = ReferenceEvaluator(model).run(None, data)
    assert set(normalize_qwen4_static_layout_graph(path)) == {'first_transpose', 'second_transpose', 'reshape', 'back'}
    assert normalize_qwen4_static_layout_graph(path) == ()
    for actual, reference in zip(ReferenceEvaluator(onnx.load(path)).run(None, data), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)


def _sdk_gather_index_normalization():
    """Real SDK negative-index normalization shared by two rotary tables."""
    shape = [2, 3]
    nodes = [
        helper.make_node('Cast', ['ids'], ['wide'], name='widen', to=TensorProto.INT64),
        helper.make_node('Less', ['wide', 'zero'], ['negative'], name='is_negative'),
        helper.make_node('Add', ['wide', 'capacity'], ['shifted'], name='wrap_negative'),
        helper.make_node('Where', ['negative', 'shifted', 'wide'], ['normalized'], name='normalize'),
        helper.make_node('Less', ['normalized', 'zero'], ['still_negative'], name='invalid_negative'),
        helper.make_node('Where', ['still_negative', 'capacity', 'normalized'], ['indices'], name='sentinel'),
        helper.make_node('GatherElements', ['cos_table', 'indices'], ['cos'], name='cos_lookup', axis=0),
        helper.make_node('GatherElements', ['sin_table', 'indices'], ['sin'], name='sin_lookup', axis=0),
    ]
    return helper.make_model(helper.make_graph(
        nodes, 'sdk_rotary_indices',
        [helper.make_tensor_value_info('ids', TensorProto.INT32, shape)],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in ['cos', 'sin']],
        initializer=[
            numpy_helper.from_array(np.array(0, dtype=np.int64), 'zero'),
            numpy_helper.from_array(np.array(8, dtype=np.int64), 'capacity'),
            numpy_helper.from_array(np.arange(24, dtype=np.float32).reshape(8, 3), 'cos_table'),
            numpy_helper.from_array(-np.arange(24, dtype=np.float32).reshape(8, 3), 'sin_table'),
        ],
        value_info=[
            helper.make_tensor_value_info(name, dtype, shape)
            for name, dtype in [('wide', TensorProto.INT64), ('negative', TensorProto.BOOL),
                                ('shifted', TensorProto.INT64), ('normalized', TensorProto.INT64),
                                ('still_negative', TensorProto.BOOL), ('indices', TensorProto.INT64)]
        ],
    ), opset_imports=[helper.make_opsetid('', 18)])


def test_rotary_index_lowering_preserves_positive_and_negative_gathers():
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import rewrite_gather_index_normalization

    model = _sdk_gather_index_normalization()
    ids = np.array([[-8, -1, 0], [1, 6, 7]], dtype=np.int32)
    expected = ReferenceEvaluator(model).run(None, {'ids': ids})
    assert rewrite_gather_index_normalization(model) == 1
    assert rewrite_gather_index_normalization(model) == 0
    assert not any(n.op_type == 'Cast' and helper.get_attribute_value(n.attribute[0]) == TensorProto.INT64
                   for n in model.graph.node)
    onnx.checker.check_model(model)
    for actual, reference in zip(ReferenceEvaluator(model).run(None, {'ids': ids}), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)


def test_rotary_index_lowering_preserves_invalid_sentinel_at_int32_extremes():
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import rewrite_gather_index_normalization

    model = _sdk_gather_index_normalization()
    assert rewrite_gather_index_normalization(model) == 1
    # Evaluate the proven index subgraph itself: invalid indices deliberately
    # remain invalid, rather than being silently clipped to a valid table row.
    retained = [node for node in model.graph.node if node.op_type != 'GatherElements']
    del model.graph.node[:]
    model.graph.node.extend(retained)
    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info('indices', TensorProto.INT32, [2, 3]))
    ids = np.array([[np.iinfo(np.int32).min, -9, -8], [-1, 8, np.iinfo(np.int32).max]], dtype=np.int32)
    wide = ids.astype(np.int64)
    normalized = np.where(wide < 0, wide + 8, wide)
    expected = np.where(normalized < 0, 8, normalized)
    actual = ReferenceEvaluator(model).run(None, {'ids': ids})[0]
    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('mismatch', ['source_width', 'axis_size', 'dynamic_axis', 'public_index',
                                      'shared_add', 'shared_cast', 'other_consumer', 'wrong_sentinel',
                                      'override_capacity', 'override_zero'])
def test_rotary_index_lowering_rejects_unproven_or_observable_chains(mismatch):
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import rewrite_gather_index_normalization

    model = _sdk_gather_index_normalization()
    if mismatch == 'source_width':
        model.graph.input[0].type.tensor_type.elem_type = TensorProto.INT64
    elif mismatch == 'axis_size':
        model.graph.initializer[2].dims[0] = 9
    elif mismatch == 'dynamic_axis':
        del model.graph.initializer[2]
        model.graph.input.append(helper.make_tensor_value_info('cos_table', TensorProto.FLOAT, ['rows', 3]))
    elif mismatch == 'public_index':
        model.graph.output.append(helper.make_tensor_value_info('indices', TensorProto.INT64, [2, 3]))
    elif mismatch in {'shared_add', 'shared_cast', 'other_consumer'}:
        source = {'shared_add': 'shifted', 'shared_cast': 'wide', 'other_consumer': 'indices'}[mismatch]
        model.graph.node.append(helper.make_node('Identity', [source], ['observed'], name='other_user'))
        model.graph.output.append(helper.make_tensor_value_info('observed', TensorProto.INT64, [2, 3]))
    elif mismatch.startswith('override_'):
        name = 'capacity' if mismatch == 'override_capacity' else 'zero'
        model.graph.input.append(helper.make_tensor_value_info(name, TensorProto.INT64, []))
    else:
        model.graph.initializer.append(numpy_helper.from_array(np.array(7, dtype=np.int64), 'bad_sentinel'))
        model.graph.node[5].input[1] = 'bad_sentinel'
    before = model.SerializeToString()
    assert rewrite_gather_index_normalization(model) == 0
    assert model.SerializeToString() == before


def test_rotary_index_file_lowering_retains_shared_constants_and_external_weights(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import normalize_gather_index_graph

    model = _sdk_gather_index_normalization()
    model.graph.node.append(helper.make_node('Identity', ['capacity'], ['public_capacity'], name='capacity_user'))
    model.graph.output.append(helper.make_tensor_value_info('public_capacity', TensorProto.INT64, []))
    path = tmp_path / 'rotary.onnx'
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location='weight.bin', size_threshold=0)
    external_before = (tmp_path / 'weight.bin').read_bytes()
    ids = {'ids': np.array([[-8, -1, 0], [1, 6, 7]], dtype=np.int32)}
    expected = ReferenceEvaluator(str(path)).run(None, ids)
    assert normalize_gather_index_graph(path) == 1
    assert normalize_gather_index_graph(path) == 0
    assert (tmp_path / 'weight.bin').read_bytes() == external_before
    onnx.checker.check_model(str(path))
    for actual, reference in zip(ReferenceEvaluator(str(path)).run(None, ids), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
    saved = onnx.load(path, load_external_data=False)
    assert next(t for t in saved.graph.initializer if t.name == 'capacity').data_type == TensorProto.INT64
