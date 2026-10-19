"""Decode-only ACEP/SAFE wiring and expert routing artifacts for Qwen MoE."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from xhquant.api import get_xhquant_logger
from xhquant.nn.modules.moeblock import MoeBlock


def normalize_dynamic_prune(config):
    if config is None:
        return {"mode": "off"}
    if not isinstance(config, Mapping):
        raise TypeError("dynamic_prune must be a mapping")
    config = dict(config)
    allowed = {
        "mode",
        "threshold",
        "s_scalar_path",
        "auto_s_scalar",
        "s_scalar_output",
        "s_scalar_compute_device",
        "s_scalar_eps",
        "overwrite_s_scalar",
        "algorithm",
        "calibration",
        "ace_priors_path",
        "exfold_artifact_path",
        "max_projection_loss",
        "even_fold_policy",
    }
    if unknown := set(config) - allowed:
        raise ValueError(f"Unknown dynamic_prune options: {sorted(unknown)}")
    config.setdefault("mode", "decode")
    if config["mode"] not in ("off", "decode"):
        raise ValueError("dynamic_prune.mode must be 'off' or 'decode'; prefill pruning is unsupported")
    threshold = float(config.get("threshold", 0.85))
    if not 0 < threshold <= 1:
        raise ValueError("dynamic_prune.threshold is ACEP coverage and must be in (0, 1]")
    config["threshold"] = threshold
    eps = float(config.get("s_scalar_eps", 1e-8))
    if not 0 < eps < float("inf"):
        raise ValueError("dynamic_prune.s_scalar_eps must be positive and finite")
    config["s_scalar_eps"] = eps
    config.setdefault("algorithm", "acep")
    if config["algorithm"] not in ("acep", "safe"):
        raise ValueError("dynamic_prune.algorithm must be acep or safe")
    safe_only = ("ace_priors_path", "exfold_artifact_path", "even_fold_policy", "max_projection_loss", "calibration")
    if config["algorithm"] != "safe":
        # Switching the default SAFE YAML to ACEP carries its fixed policy and
        # calibration block along. Those defaults belong to SAFE and are dropped;
        # an explicit artifact path is still a conflicting override.
        if config.get("ace_priors_path") or config.get("exfold_artifact_path"):
            raise ValueError("SAFE options require algorithm=safe")
        for key in safe_only:
            config.pop(key, None)
    if config["algorithm"] == "safe":
        if "calibration" in config:
            from .safe_calibration import normalize_calibration

            config["calibration"] = normalize_calibration(config["calibration"])
        config.setdefault("even_fold_policy", "nearest_mass")
        if config["even_fold_policy"] != "nearest_mass":
            raise ValueError("SAFE even_fold_policy is fixed at nearest_mass")
        limit = float(config.get("max_projection_loss", 1.0))
        if limit != 1.0:
            raise ValueError("SAFE max_projection_loss is fixed at 1.0")
        config["max_projection_loss"] = limit
        if eps != 1e-8:
            raise ValueError("SAFE requires s_scalar_eps=1e-8 for its denominator safety floor")
    return config


def _source_fingerprint(model_dir):
    """Bind automatic caches to the checkpoint index/config and shard revisions."""
    root = Path(model_dir).resolve()
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        path = root / name
        if path.exists():
            digest.update(path.read_bytes())
    for path in sorted(root.glob("*.safetensors")):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def validate_scalars(payload, expert_counts):
    """Require one finite, positive score vector for every exported body layer."""
    if not isinstance(payload, Mapping):
        raise ValueError("s_scalar must map body layer indices to expert score vectors")
    scalars = {str(key): torch.as_tensor(value, dtype=torch.float32).cpu() for key, value in payload.items()}
    if set(scalars) != set(expert_counts):
        raise ValueError(f"s_scalar layer IDs must match body layers {sorted(expert_counts)}; got {sorted(scalars)}")
    for key, count in expert_counts.items():
        value = scalars[key]
        if value.shape != (count,) or not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
            raise ValueError(f"s_scalar layer {key} must contain {count} finite positive scores")
    return scalars


@torch.no_grad()
def compute_layer_scalars(gamma, moeblock, device, eps=1e-8):
    """Method1 on the consumed weights and effective (1 + HF weight) norm."""
    gamma = gamma.to(device=device, dtype=torch.float32)
    scores = []
    for expert in range(moeblock.expert_gate_proj_weight.shape[0]):
        gate = moeblock.expert_gate_proj_weight[expert].to(device=device, dtype=torch.float32).T
        up = moeblock.expert_up_proj_weight[expert].to(device=device, dtype=torch.float32).T
        down = moeblock.expert_down_proj_weight[expert].to(device=device, dtype=torch.float32).T
        norm_up = torch.linalg.vector_norm(gamma[:, None] * up)
        norm_gate = torch.linalg.vector_norm(gamma[:, None] * gate)
        score_up = torch.linalg.vector_norm(gamma[:, None] * (norm_up * (gate @ down)))
        score_gate = torch.linalg.vector_norm(gamma[:, None] * (norm_gate * (up @ down)))
        scores.append(torch.sqrt(score_up * score_gate + eps).cpu())
    scores = torch.stack(scores)
    return scores / (scores.mean() + eps)


def prepare_scalars(language_model, config, model_dir, work_dir):
    """Use wrapped body modules, avoiding fused-name matching and MTP collisions."""
    layers = {}
    for index, layer in enumerate(language_model.layers):
        block = getattr(layer.mlp, "moeblock", None)
        if not isinstance(block, MoeBlock) or block.expert_gate_proj_weight is None:
            raise ValueError("Decode ACEP requires regular export with materialized MoeBlock weights")
        block._acep_layer_idx = str(index)
        layers[str(index)] = (layer.post_attention_layernorm.norm.weight, block)
    counts = {key: block.expert_gate_proj_weight.shape[0] for key, (_, block) in layers.items()}
    if not counts:
        raise ValueError("No body MoE layers found for decode ACEP")
    source = _source_fingerprint(model_dir)
    if config.get("algorithm") == "safe":
        from .safe_prune import prepare_safe_scalars

        return prepare_safe_scalars(counts, config, source, work_dir)
    eps = config.get("s_scalar_eps", 1e-8)
    explicit = config.get("s_scalar_path")
    path = Path(explicit or config.get("s_scalar_output") or Path(work_dir) / "moe_acep_s_scalar.pt")
    logger = get_xhquant_logger()
    if explicit or (path.exists() and not config.get("overwrite_s_scalar", False)):
        payload = (
            json.loads(path.read_text())
            if path.suffix == ".json"
            else torch.load(path, map_location="cpu", weights_only=True)
        )
        if isinstance(payload, Mapping) and "scalars" in payload:
            if payload.get("method") != "acep_method1_effective_norm_v1" or payload.get("source") != source:
                raise ValueError(f"s_scalar cache does not match this checkpoint/method: {path}; regenerate it")
            if payload.get("eps") != eps:
                raise ValueError(f"s_scalar cache does not match s_scalar_eps={eps}: {path}; regenerate it")
            payload = payload["scalars"]
        elif not explicit:
            raise ValueError(f"Automatic s_scalar cache has no source provenance: {path}; regenerate it")
        logger.info(f"Using decode ACEP s_scalar: {path}")
        return validate_scalars(payload, counts), str(path)
    if not config.get("auto_s_scalar", True):
        raise ValueError("Decode ACEP requires s_scalar_path or auto_s_scalar=true")
    device = config.get("s_scalar_compute_device", "auto")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scalars = {}
    for key, (gamma, block) in layers.items():
        logger.info(f"Computing decode ACEP s_scalar for body layer {key}/{len(layers)} on {device}")
        scalars[key] = compute_layer_scalars(gamma, block, device, eps)
    scalars = validate_scalars(scalars, counts)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = {"method": "acep_method1_effective_norm_v1", "source": source, "eps": eps, "scalars": scalars}
    if path.suffix == ".json":
        payload["scalars"] = {key: value.tolist() for key, value in scalars.items()}
        temporary.write_text(json.dumps(payload, indent=2))
    else:
        torch.save(payload, temporary)
    temporary.replace(path)
    logger.info(f"Saved decode ACEP s_scalar for {len(scalars)} body layers to {path}")
    return scalars, str(path)


def configure_decode_graph(frontend_model, scalars, threshold):
    """Separate per-graph attributes while sharing all expert weight storage."""
    from ..qwen3_5.qwen3_5_llm_model import _copy_model_shared_params

    for phase in ("prefill", "decode"):
        graph = frontend_model[phase]
        found = set()
        for name, block in list(graph.named_modules()):
            if not isinstance(block, MoeBlock):
                continue
            key = getattr(block, "_acep_layer_idx", None)
            if key not in scalars or key in found:
                raise ValueError(f"Invalid body MoE mapping in {phase}: {name}, layer={key}")
            found.add(key)
            if phase == "decode":
                block = _copy_model_shared_params(block)
                graph.set_submodule(name, block)
                block.prune_threshold = threshold
                block.s_scalar = scalars[key].to(device=block.expert_gate_proj_weight.device)
            else:
                block.prune_threshold = None
                block.s_scalar = None
        if found != set(scalars):
            raise ValueError(f"Incomplete MoE layers in {phase}: expected {sorted(scalars)}, got {sorted(found)}")
