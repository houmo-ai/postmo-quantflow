# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Portable HMONNX-only policy; FP32 Euler state and request-local graph caches."""

import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .processor import WallProcessor


def read_metadata(root):
    from xhmodel_merak.model_artifact.files import confined_file

    root = Path(root).resolve()
    from .config import normalize_prefill_config

    meta = normalize_prefill_config(
        json.loads(confined_file(root, "runtime_meta_info.json", "runtime metadata").read_text())
    )
    if meta.get("schema_version") != 1 or meta.get("kind") != "wall_oss" or meta.get("model_id") != "wall-oss-0.5":
        raise ValueError("invalid Wall deployment identity")
    if meta.get("instruction_truncation", "error") not in ("right", "error"):
        raise ValueError("invalid Wall instruction truncation policy")
    from .build import export_entries, validate_config

    cfg = validate_config(meta["deployment_config"])
    meta["deployment_config"] = cfg
    from .config import validate_source_config

    source = json.loads(confined_file(root, "wall_config.json", "model configuration").read_text())["model_config"]
    validate_source_config(source)
    head_dim = source["hidden_size"] // source["num_attention_heads"]
    expected = dict(
        context_length=cfg["context_length"],
        prefill_chunk_length=cfg["prefill_chunk_length"],
        visual_capacity=cfg["visual_capacity"],
        horizon=32,
        action_dim=26,
        layers=source["num_hidden_layers"],
        kv_heads=source["num_key_value_heads"],
        head_dim=head_dim,
        target_device="XH2a",
        cache_policy="inplace_replace_action",
        quant_type=cfg["model"]["quant_scheme"]["quant_type"],
    )
    if any(meta.get(k) != v for k, v in expected.items()) or set(meta["graphs"]) != {"visual", "prefill", "denoise"}:
        raise ValueError("Wall metadata and effective configuration disagree")
    for name, entry in export_entries(root, meta).items():
        if entry["type"] != "tokenizer":
            confined_file(root, entry["path"], name)
    audit_graphs(root, meta)
    return meta


def audit_graphs(root, meta):
    """Physical ports and precision must agree with the published contract."""
    from collections import Counter

    import onnx

    c, p, v = meta["context_length"], meta["prefill_chunk_length"], meta["visual_capacity"]
    arch = json.loads((root / "wall_config.json").read_text())["model_config"]
    hidden, ah = arch["hidden_size"], arch["dim_inputs"][1]
    layers, kv, dim = meta["layers"], meta["kv_heads"], meta["head_dim"]
    vc = arch["vision_config"]
    vd = vc["hidden_size"] // vc["num_heads"]
    weight_bits, activation_bits = {"w16a16_sefp": (16, 16), "w8a16_sefp": (8, 16), "w8a8_sefp": (8, 8)}[
        meta["quant_type"]
    ]
    specifications = {
        "visual": [(1, v, 1176), (v,), (v // 4,), (1, 1, v, vd), (1, 1, v, vd), (1, v, 1), (1, v, 2), (1, v, 2)],
        "prefill": [(1, p, hidden), (1, 1, p, dim), (1, 1, p, dim), (1,), (1,)] + [(1, kv, c, dim)] * (2 * layers),
        "denoise": [(1, 32, 26), (1, 32, 26), (1, ah), (1, 1, 32, dim), (1, 1, 32, dim), (1,), (1,)]
        + [(1, kv, c, dim)] * (2 * layers),
    }
    for role, shapes in specifications.items():
        graph = onnx.load(str(root / meta["graphs"][role]), load_external_data=False).graph
        actual = [tuple(d.dim_value for d in value.type.tensor_type.shape.dim) for value in graph.input]
        if actual != shapes or len(graph.output) != 1:
            raise ValueError(f"{role} physical graph ports disagree with metadata")
        integer_ports = {"visual": {1, 2, 6, 7}, "prefill": {3, 4}, "denoise": {5, 6}}[role]
        if any(
            value.type.tensor_type.elem_type
            != (onnx.TensorProto.INT32 if i in integer_ports else onnx.TensorProto.FLOAT16)
            for i, value in enumerate(graph.input)
        ):
            raise ValueError(f"{role} graph input dtypes disagree with the runtime")
        output_shape = tuple(d.dim_value for d in graph.output[0].type.tensor_type.shape.dim)
        expected_output = {"visual": (1, v // 4, hidden), "prefill": (1, p, hidden), "denoise": (1, 32, 26)}[role]
        if output_shape != expected_output or graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT16:
            raise ValueError(f"{role} graph output disagrees with the runtime")
        counts = Counter(node.op_type for node in graph.node)
        if counts["FlashAttention"] != (vc["depth"] if role == "visual" else layers):
            raise ValueError(f"{role} must contain every attention layer")
        if counts["KVcache"] != (0 if role == "visual" else 2 * layers):
            raise ValueError(f"{role} cache producer count disagrees with the contract")
        for node in graph.node:
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if node.op_type == "Linear" and (
                attrs.get("hmfp_weight_man_bit") != weight_bits
                or attrs.get("hmfp_act_man_bit") != activation_bits
                or attrs.get("hmfp_weight_nshare") != 64
                or attrs.get("hmfp_act_nshare") != 64
            ):
                raise ValueError(f"{role} Linear precision/group disagrees with the package")
            if node.op_type == "FlashAttention" and any(
                attrs.get(name) != activation_bits for name in ("q_bits", "k_bits", "v_bits", "p_bits")
            ):
                raise ValueError(f"{role} attention precision disagrees with the package")
            if node.op_type == "FlashAttention" and (
                attrs.get("num_heads") != (vc["num_heads"] if role == "visual" else arch["num_attention_heads"])
                or attrs.get("num_kv_heads") != (vc["num_heads"] if role == "visual" else kv)
                or attrs.get("is_causal") != (0 if role == "visual" else 1)
            ):
                raise ValueError(f"{role} attention geometry disagrees with the package")


class WallRuntime:
    def __init__(self, root, *, device="cuda:0", golden_step=None):
        from safetensors.torch import load_file

        self.root = Path(root).resolve()
        self.meta = read_metadata(root)
        self.device = torch.device(device)
        self.processor = WallProcessor(
            self.root,
            prefix_capacity=self.meta["context_length"],
            visual_capacity=self.meta["visual_capacity"],
            truncate_instruction=self.meta.get("instruction_truncation", "error") == "right",
        )
        weight = load_file(str(self.root / "quant_embedding.safetensors"))["embedding.weight"]
        arch = self.processor.config["model_config"]
        self.hidden, self.action_hidden = arch["hidden_size"], arch["dim_inputs"][1]
        if tuple(weight.shape) != (self.processor.config["embedding_rows"], self.hidden):
            raise ValueError("invalid embedding shape")
        self.embedding = weight.to(self.device)
        self.sessions = {name: self._session(name) for name in self.meta["graphs"]}
        self.golden_step, self.captured = golden_step, set()
        self.last_trace = []

    def _session(self, name, golden=False):
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

        path = self.root / self.meta["graphs"][name]
        session = HMONNXInferenceV2(
            str(path), session_config=HMONNXInferenceConfig(exec_devices=[str(self.device)], enable_golden=golden)
        )
        if name != "visual" and session.enable_inplace_cache_updates() != 2 * self.meta["layers"]:
            raise ValueError("every Wall layer must update both KV ports in place")
        return session

    def _call(self, name, *args):
        if self.golden_step is not None and name not in self.captured:
            path = (self.root / self.meta["graphs"][name]).parent / f"step_{self.golden_step}"
            if path.exists():
                raise ValueError(f"refusing to overwrite Golden {path}")
            session = self._session(name, True)
            session.save_golden_dir = str(path)
            output = session(*args).clone()
            for linked in (self.root / self.meta["graphs"][name]).parent.iterdir():
                if linked.is_file() and (linked.suffix == ".onnx" or "external_data" in linked.name):
                    (path / linked.name).symlink_to(Path("..") / linked.name)
            self.captured.add(name)
            del session
            gc.collect()
            torch.cuda.empty_cache()
            return output
        return self.sessions[name](*args).clone()

    @torch.inference_mode()
    def predict(
        self,
        *,
        instruction,
        images,
        state,
        norm_key="x2_normal",
        dof_mask=None,
        state_mask=None,
        seed=0,
        steps=10,
        noise=None,
    ):
        from xhquant.core import CacheTensor

        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be a positive integer")
        data = self.processor.prepare(instruction, images, state, norm_key, state_mask)
        dtype = torch.float16

        def move(x):
            return x.to(self.device, dtype=dtype if x.is_floating_point() else torch.int32)

        ids = data["ids"].to(self.device)
        embeddings = F.embedding(ids, self.embedding).half()
        features = []
        for patches, geometry, grid in zip(data["patches"], data["geometry"], data["grid"], strict=True):
            output = self._call("visual", move(patches), *[move(x) for x in geometry])
            features.append(output[:, : int(grid.prod()) // 4])
        image_mask = ids == self.processor.image_token_id
        merged = torch.cat(features, 1)
        if image_mask.sum() != merged.shape[1]:
            raise ValueError("image token/feature mismatch")
        embeddings[image_mask] = merged.reshape(-1, self.hidden)
        cos, sin = (move(x) for x in data["rope"])
        p = embeddings.shape[1]
        chunk = self.meta["prefill_chunk_length"]
        h = 32
        caches = tuple(
            CacheTensor(
                torch.zeros(
                    1,
                    self.meta["kv_heads"],
                    self.meta["context_length"],
                    self.meta["head_dim"],
                    device=self.device,
                    dtype=dtype,
                )
            )
            for _ in range(2 * self.meta["layers"])
        )

        def scalar(v):
            return torch.tensor([v], device=self.device, dtype=torch.int32)

        for start in range(0, p, chunk):
            length = min(chunk, p - start)
            values = (
                F.pad(embeddings[:, start : start + length], (0, 0, 0, chunk - length)),
                F.pad(cos[:, :, start : start + length], (0, 0, 0, chunk - length), value=1),
                F.pad(sin[:, :, start : start + length], (0, 0, 0, chunk - length)),
                scalar(start),
                scalar(length),
                *caches,
            )
            self._call("prefill", *values)
        mask = np.ones(26, np.float32) if dof_mask is None else np.asarray(dof_mask, np.float32)
        if mask.shape != (26,) or not np.isin(mask, [0, 1]).all():
            raise ValueError("DOF mask must contain 26 boolean entries")
        dof = torch.tensor(mask, device=self.device)[None, None].expand(1, h, 26)
        if noise is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            action = torch.randn((1, h, 26), generator=generator, device=self.device, dtype=torch.float32)
        else:
            action = torch.as_tensor(noise, device=self.device, dtype=torch.float32).clone()
            if action.shape != (1, h, 26) or not torch.isfinite(action).all():
                raise ValueError("noise must be finite [1,32,26]")
        padding = torch.tensor(
            self.processor.normalize(np.zeros(26, np.float32), norm_key, "action"), device=self.device
        )[None, None]
        vpadding = padding - action
        scheduler = self.processor.config["model_config"]["noise_scheduler"]
        times = torch.linspace(0, 1, steps + 1, device=self.device, dtype=torch.float32)
        shift = scheduler.get("time_shift", 1.0)
        times = shift * times / (1 + (shift - 1) * times) * scheduler.get("s", 0.999)
        inv = torch.exp(
            torch.arange(self.action_hidden // 2, device=self.device, dtype=torch.float32)
            * (-np.log(10000) / (self.action_hidden // 2 - 1))
        )
        prefix_before = [cache.data[:, :, :p].clone() for cache in caches]
        self.last_trace = [action.cpu().clone()]
        for t, tnext in zip(times[:-1], times[1:], strict=True):
            angle = t * inv
            time = torch.cat((angle.sin(), angle.cos()))[None]
            velocity = self._call(
                "denoise",
                action.half(),
                dof.half(),
                time.half(),
                cos[:, :, p : p + h],
                sin[:, :, p : p + h],
                scalar(p),
                scalar(h),
                *caches,
            ).float()
            velocity = velocity * dof + vpadding * (1 - dof)
            action = action + (tnext - t) * velocity
            if not torch.isfinite(action).all():
                raise ValueError("nonfinite action trajectory")
            self.last_trace.append(action.cpu().clone())
        if any(
            not torch.equal(before, cache.data[:, :, :p]) for before, cache in zip(prefix_before, caches, strict=True)
        ):
            raise ValueError("denoising modified the immutable observation prefix")
        self.last_cache_checks = dict(
            prefix_unchanged=True, action_offset=p, action_length=h, cache_capacity=self.meta["context_length"]
        )
        stats = self.processor.normalizers["action"][norm_key]
        result = (action + 1) / 2 * torch.tensor(stats["delta"], device=self.device) + torch.tensor(
            stats["min"], device=self.device
        )
        return result[0].cpu().numpy(), dict(
            steps=steps,
            prefix_tokens=p,
            image_grids=data["grid"].tolist(),
            resized_sizes=data["sizes"],
            norm_key=norm_key,
            cache=self.last_cache_checks,
            truncation=data["truncation"],
        )
