# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Shared local inputs and calibration budget for tiny real models."""
import copy
from dataclasses import replace
from _unified_small_model import write_calibration

def prepare_context(case, root):
    from xhmodel_merak.model_catalog.registry import get_catalog

    catalog = get_catalog()
    recipe = catalog._recipes[case.model_id]
    source = root / "source"
    calibration, image = write_calibration(root)
    kwargs = {
        "model_id": case.model_id,
        "model_dir": str(source),
        "output_dir": str(root / "artifact"),
        "device": "cuda:0",
    }
    inputs = {"prompt": "hello world", "max_new_tokens": 2}
    replacements = dict(recipe.config_replacements)
    family = case.fixture
    return catalog, recipe, source, calibration, image, kwargs, inputs, replacements, family


def configure_calibration(case, calibration, monkeypatch):
    """Change test data and calibration budget, never bypass quant/export calls."""
    import xhmodel_merak.model_build.api as api

    if case.fixture == "minicpm":
        import datasets

        def local_wikitext(name, config, *, split):
            assert (name, config, split) == ("Salesforce/wikitext", "wikitext-2-raw-v1", "train")
            return datasets.Dataset.from_dict({"text": ["hello world " * 32] * 2})

        # The legacy workflow owns a fixed dataset request. Replace only that
        # external data boundary; quantization itself executes unchanged.
        monkeypatch.setattr(datasets, "load_dataset", local_wikitext)
        return

    original = api.resolve_build_plan

    def resolve(**kwargs):
        plan = original(**kwargs)
        replacements = copy.deepcopy(plan.config_replacements)
        quant = replacements.get("quant")
        fields = {
            "calibration.dataset": str(calibration),
            "calibration.nsamples": 2,
            "calibration.seqlen": 64,
            "runtime.batch_size": 1,
        }
        if case.fixture in ("qwen_next", "glm", "ling"):
            fields["calibration.jsonl"] = fields.pop("calibration.dataset")
        if case.fixture == "qwen_next" and plan.quant_method == "gptq":
            # A random tiny checkpoint is not a WikiText benchmark. Keep real
            # quantization/reload and public inference, without hub downloads
            # or the example script's default 128-token demonstration.
            fields.update({"validation.skip_ppl": True, "validation.max_new_tokens": 2})
        if plan.quant_method == "autoround":
            fields["iters"] = 4
        if case.fixture == "paddleocr":
            fields = {"nsamples": 2, "iters": 4}
        for key, value in fields.items():
            # A resource binding may already occupy this dotted path after the
            # whole quant mapping. Ensure the fixture data remains the last write.
            replacements.pop("quant." + key, None)
            if isinstance(quant, dict):
                node = quant
                parts = key.split(".")
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = value
            else:
                replacements["quant." + key] = value
        if isinstance(quant, dict) and quant.get("mtp"):
            quant["mtp"].update(validation_nsamples=1, iters=4)
        return replace(plan, config_replacements=replacements)

    monkeypatch.setattr(api, "resolve_build_plan", resolve)
