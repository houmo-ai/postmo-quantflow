"""Explicit Qwen4 entry into the shared build API."""

from __future__ import annotations

import ast
import inspect
import textwrap
from contextlib import contextmanager
from functools import wraps
from typing import Any


MODEL_ID = "qwen3.8-flash-next"


@contextmanager
def _rtn_dependency_scope():
    """Allow this model's quant-rtn declaration in the existing publisher."""
    from xhmodel_merak.model_build import executor

    original = executor.write_dependency_files
    if getattr(original, "_qwen4_rtn_scope", False):
        yield
        return
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    expected = {"build", "inference", "checkpoint_build", "quant-gptq", "quant-autoround", "quant-qat"}
    matches = [
        node for node in ast.walk(tree) if isinstance(node, ast.Set)
        and all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.elts)
        and {item.value for item in node.elts} == expected
    ]
    if len(matches) != 1:
        raise RuntimeError("Qwen4 RTN is incompatible with the installed dependency publisher")
    matches[0].elts.append(ast.Constant(value="quant-rtn"))
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), original.__code__.co_filename, "exec"), namespace)
    adapted = namespace[original.__name__]

    @wraps(original)
    def write(root, model_id, variant):
        return (adapted if model_id == MODEL_ID else original)(root, model_id, variant)

    write._qwen4_rtn_scope = True
    executor.write_dependency_files = write
    try:
        yield
    finally:
        executor.write_dependency_files = original


def build_qwen4_exp_model(*, model_id: str = MODEL_ID, **kwargs: Any):
    """Build Qwen4 with checkpoint compatibility scoped to this call.

    The dedicated CLI runs this in its own process. Importing this module does
    not modify the shared builder, and the compatibility scope restores its
    functions after success or failure. All build arguments and the returned
    BuildResult belong to the public build_model API.
    """
    if model_id != MODEL_ID:
        raise ValueError(f"Qwen4 build entry requires model_id={MODEL_ID!r}, got {model_id!r}")

    from dataclasses import replace

    from xhmodel_merak.model_build.executor import execute_build
    from xhmodel_merak.model_build.resolver import resolve_build_plan

    from .checkpoint_patch import qwen4_exp_checkpoint_scope

    with qwen4_exp_checkpoint_scope(), _rtn_dependency_scope():
        plan = resolve_build_plan(model_id=model_id, **kwargs)
        if plan.quant_method is not None:
            method = plan.quant_method
            # The public plan selects the producer; bind that choice to the
            # model workflow without extending the shared builder.
            replacements = {
                **plan.config_replacements,
                "quant.method": method,
                "quant.iters": 0 if method == "rtn" else 200,
                "quant.disable_opt_rtn": True if method == "rtn" else None,
            }
            plan = replace(plan, config_replacements=replacements)
        return execute_build(plan)


__all__ = ["MODEL_ID", "build_qwen4_exp_model"]
