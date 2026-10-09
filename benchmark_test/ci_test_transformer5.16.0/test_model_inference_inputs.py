"""Numeric input-schema constraints shared by registered model recipes."""

from types import SimpleNamespace

import pytest

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_inference.inputs import validate_inputs


@pytest.mark.parametrize("value, divisor", [(32, 32), (1024, 32), (0, 32), (0.3, 0.1), (1.25, 0.25)])
def test_input_schema_accepts_integer_and_decimal_multiples(value, divisor):
    recipe = SimpleNamespace(
        model_id="numeric",
        modalities=("text",),
        input_schema={"properties": {"size": {"type": "number", "multipleOf": divisor}}},
    )
    assert validate_inputs(recipe, {"size": value}) == {"size": value}


@pytest.mark.parametrize("value, divisor", [(1000, 32), (31, 32), (0.31, 0.1), (1.26, 0.25)])
def test_input_schema_rejects_nonmultiples_before_loading(value, divisor):
    recipe = SimpleNamespace(
        model_id="numeric",
        modalities=("text",),
        input_schema={"properties": {"size": {"type": "number", "multipleOf": divisor}}},
    )
    with pytest.raises(ModelError, match="multiple of"):
        validate_inputs(recipe, {"size": value})
