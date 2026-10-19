"""由 ModelZoo 消费 SAFE 标定产物，离线融合先验并准备 SDK 的 s_scalar 输入。"""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch

from xhquant.common.moe_safe import validate_safe_data


def pack_safe_data(amp, aimer, coefficient, projection_loss):
    """在导出前融合 AMP/AIMER 并打包一层数据，不增加 MoE 的输入槽。

    amp/aimer 为 [E]，coefficient/projection_loss 为 [E, E]。
    系数 C[s,t] 表示用 C[s,t] * 专家 t 的输出近似专家 s 的输出，不能交换源、目标轴。
    返回独立、连续的 FP16 数据包；配对禁用标记只在包内修改，不改调用方的原表。
    """
    amp, aimer, coefficient, projection_loss = [
        torch.as_tensor(value, dtype=torch.float32, device="cpu")
        for value in (amp, aimer, coefficient, projection_loss)
    ]
    count = amp.numel()
    if count < 1 or amp.shape != (count,) or aimer.shape != (count,):
        raise ValueError("SAFE AMP/AIMER must be nonempty matching expert vectors")
    if not all(bool(torch.isfinite(prior).all() & (prior > 0).all()) for prior in (amp, aimer)):
        raise ValueError("SAFE AMP/AIMER priors must be finite and positive")
    if coefficient.shape != (count, count) or projection_loss.shape != coefficient.shape:
        raise ValueError("SAFE coefficient/loss must be [source expert, target expert]")
    if not bool(torch.isfinite(coefficient).all() & torch.isfinite(projection_loss).all()):
        raise ValueError("SAFE coefficient/loss must be finite")
    if bool((projection_loss < 0).any()):
        raise ValueError("SAFE projection loss must be non-negative")
    amp, aimer = amp.half(), aimer.half()
    if not all(bool(torch.isfinite(prior).all() & (prior > 0).all()) for prior in (amp, aimer)):
        raise ValueError("SAFE AMP/AIMER priors must remain finite and positive in float16")
    # 融合本身也用 FP16；先分别开方再相乘，避免 AMP*AIMER 的中间乘积上溢/下溢。
    scalar = amp.sqrt() * aimer.sqrt()
    # 必须在损失降精度之前屏蔽超阈值配对，防止 1.0001 舍入到 1.0 后被误判为安全。
    # 65504 是 FP16 最大有限数，替代旧 1e30 哨兵；不将无穷大写入导出常量。
    disabled = projection_loss > 1.0
    coefficient = coefficient.masked_fill(disabled, 0).half()
    projection_loss = projection_loss.masked_fill(disabled, torch.finfo(torch.float16).max).half()
    data = torch.cat((scalar[:, None], coefficient, projection_loss), dim=1)
    # 对角系数设为 1，但对角损失设为禁用值，避免将“折叠到自己”当成有效替代。
    data[:, 1 : 1 + count].diagonal().fill_(1)
    data[:, 1 + count : 1 + 2 * count].diagonal().fill_(torch.finfo(torch.float16).max)
    validate_safe_data(data, count)
    return data.contiguous()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cache(payload, counts, source, settings):
    if not isinstance(payload, Mapping) or payload.get("method") != "safe_exfold_v4":
        raise ValueError("SAFE needs a float16 safe_exfold_v4 cache; rebuild from raw prior/ExFold artifacts")
    if payload.get("source") != source:
        raise ValueError("SAFE packed cache does not match this checkpoint")
    if payload.get("settings") != settings:
        raise ValueError("SAFE packed cache settings disagree; rebuild from prior/ExFold artifacts")
    scalars = payload.get("scalars", {})
    if set(scalars) != set(counts):
        raise ValueError("SAFE packed cache layer IDs must match exported body layers")
    for key, count in counts.items():
        validate_safe_data(scalars[key], count)
    return scalars


def prepare_safe_scalars(counts, config, source, work_dir):
    """Require real pairwise projection data; a scalar vector cannot substitute."""
    settings = {
        "max_projection_loss": config.get("max_projection_loss", 1.0),
        "even_fold_policy": config.get("even_fold_policy", "nearest_mass"),
    }
    if settings != {"max_projection_loss": 1.0, "even_fold_policy": "nearest_mass"}:
        raise ValueError("SAFE settings are fixed: max_projection_loss=1.0, even_fold_policy=nearest_mass")
    priors_path, artifact_path = config.get("ace_priors_path"), config.get("exfold_artifact_path")
    if config.get("s_scalar_path") and (priors_path or artifact_path):
        raise ValueError("SAFE accepts either s_scalar_path or raw prior/ExFold files, not both")
    if not priors_path and not artifact_path and config.get("s_scalar_path"):
        path = Path(config["s_scalar_path"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return _validate_cache(payload, counts, source, settings), str(path)
    if not priors_path or not artifact_path:
        raise ValueError("SAFE requires both ace_priors_path and exfold_artifact_path, or a packed SAFE s_scalar_path")
    priors = torch.load(priors_path, map_location="cpu", weights_only=True)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(priors, Mapping) or not all(isinstance(priors.get(k), Mapping) for k in ("amp", "aimer")):
        raise ValueError("SAFE priors must contain amp and aimer layer tables")
    tables = {key: {str(i): value for i, value in priors[key].items()} for key in ("amp", "aimer")}
    if set(tables["amp"]) != set(tables["aimer"]) or not set(counts).issubset(tables["amp"]):
        raise ValueError("SAFE AMP/AIMER layer IDs must agree and cover every exported body layer")
    if not isinstance(artifact, Mapping) or not {"coeff", "loss", "expert_norms"}.issubset(artifact):
        raise ValueError("SAFE ExFold artifact requires coeff, loss and expert_norms")
    coeff, loss, norms = [torch.as_tensor(artifact[k], dtype=torch.float32) for k in ("coeff", "loss", "expert_norms")]
    if (
        coeff.ndim != 3
        or coeff.shape[1] != coeff.shape[2]
        or loss.shape != coeff.shape
        or norms.shape != coeff.shape[:2]
    ):
        raise ValueError("SAFE ExFold tensors must have [layer, source, target] and [layer, expert] shapes")
    if (
        not all(bool(torch.isfinite(t).all()) for t in (coeff, loss, norms))
        or bool((loss < 0).any())
        or bool((norms <= 0).any())
    ):
        raise ValueError("SAFE ExFold tensors must be finite, loss non-negative and norms positive")
    scalars = {}
    for key, count in counts.items():
        layer = int(key)
        if layer >= coeff.shape[0] or coeff.shape[1] != count:
            raise ValueError(f"SAFE ExFold layer/expert count disagrees at body layer {key}")
        scalars[key] = pack_safe_data(
            tables["amp"][key],
            tables["aimer"][key],
            coeff[layer],
            loss[layer],
        )
    payload = {
        "method": "safe_exfold_v4",
        "source": source,
        "settings": settings,
        "scalars": scalars,
        "ace_priors_sha256": _sha256(priors_path),
        "exfold_sha256": _sha256(artifact_path),
    }
    _validate_cache(payload, counts, source, settings)
    path = Path(config.get("s_scalar_output") or Path(work_dir) / "moe_safe_s_scalar_v4.pt")
    if path.suffix != ".pt" or path.resolve() in {Path(priors_path).resolve(), Path(artifact_path).resolve()}:
        raise ValueError("SAFE packed output must be a separate .pt file")
    if path.exists() and not config.get("overwrite_s_scalar", False):
        previous = torch.load(path, map_location="cpu", weights_only=True)
        _validate_cache(previous, counts, source, settings)
        if any(previous.get(k) != payload[k] for k in ("ace_priors_sha256", "exfold_sha256")):
            raise ValueError("SAFE calibration artifacts changed; set overwrite_s_scalar=true to regenerate")
        return previous["scalars"], str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return scalars, str(path)
