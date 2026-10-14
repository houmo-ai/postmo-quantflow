"""Validate checkpoint reuse and resolve the bundled draft without loading tensors."""
import json
from pathlib import Path

from xhmodel_merak.errors import invalid_parameter


def resolve_config(*, recipe, variant, model_dir, sources, workflow, quant_checkpoint=None, precision=None):
    root = Path(quant_checkpoint or model_dir)
    config = json.loads((root / "config.json").read_text())
    if config.get("quantization_config", {}).get("quant_method") not in {"gptq", "auto-round", "auto_round", "autoround"}:
        raise invalid_parameter(recipe.model_id, "model_dir", str(root),
                                "this workflow exports an existing W8/E4 GPTQModel/AutoRound checkpoint")
    if variant.speculative_mode == "dspark" and not (root / "dspark" / "dspark_meta_info.json").is_file():
        raise invalid_parameter(recipe.model_id, "model_dir", str(root),
                                "the checkpoint must include its producer-preserved dspark/ component")
    return {}
