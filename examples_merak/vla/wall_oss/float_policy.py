"""Reusable FP32 split-policy oracle for exported Wall numerical evaluations."""

import gc

import numpy as np
import torch
from torch.nn import functional as F


class FloatPolicy:
    def __init__(self, model_dir, resource_dir, device):
        from safetensors.torch import load_file

        from xhmodel_merak.xh_other_model.models.wall_oss.graph import ActionDenoiser, ExpertStack
        from xhmodel_merak.xh_other_model.models.wall_oss.processor import WallProcessor
        from xhmodel_merak.xh_other_model.models.wall_oss.source import load_source, load_visual
        from xhmodel_merak.xh_other_model.models.wall_oss.vision import PackedVision

        self.device = torch.device(device)
        self.processor = WallProcessor(resource_dir)
        self.embedding = load_file(str(self.processor.root / "quant_embedding.safetensors"))["embedding.weight"].to(
            device=device, dtype=torch.float32
        )
        self.visual = PackedVision(load_visual(model_dir), 1024, sdk=False).to(device).eval()
        source = load_source(model_dir)
        self.prefix = (
            ExpertStack(source, expert=0, query_capacity=256, cache_capacity=2048, sdk=False).to(device).eval()
        )
        self.denoise = ActionDenoiser(source, query_capacity=32, cache_capacity=2048, sdk=False).to(device).eval()
        del source
        gc.collect()

    @torch.inference_mode()
    def predict(self, data, noise, values):
        features = []
        for patches, geometry, grid in zip(data["patches"], data["geometry"], data["grid"], strict=True):
            result = self.visual(patches.to(self.device), *[x.to(self.device) for x in geometry])
            features.append(result[:, : int(grid.prod()) // 4])
        vision = torch.cat(features, 1).reshape(-1, 2048)
        ids = data["ids"].to(self.device)
        embeddings = F.embedding(ids, self.embedding)
        embeddings[ids == self.processor.image_token_id] = vision
        p = ids.shape[1]
        cos, sin = [x.to(self.device) for x in data["rope"]]
        cache = tuple(torch.zeros(1, 2, 2048, 128, device=self.device) for _ in range(72))
        for start in range(0, p, 256):
            n = min(256, p - start)
            _, cache = self.prefix(
                F.pad(embeddings[:, start : start + n], (0, 0, 0, 256 - n)),
                F.pad(cos[:, :, start : start + n], (0, 0, 0, 256 - n), value=1),
                F.pad(sin[:, :, start : start + n], (0, 0, 0, 256 - n)),
                torch.tensor([start], device=self.device),
                torch.tensor([n], device=self.device),
                cache,
            )
        action = noise.to(self.device).clone()
        dof = torch.tensor(values["dof_mask"], device=self.device, dtype=torch.float32)[None, None].expand(1, 32, 26)
        padding = torch.tensor(
            self.processor.normalize(np.zeros(26, np.float32), values["norm_key"], "action"), device=self.device
        )[None, None]
        vpadding = padding - action
        scheduler = self.processor.config["model_config"]["noise_scheduler"]
        times = torch.linspace(0, 1, values["steps"] + 1, device=self.device)
        shift = scheduler.get("time_shift", 1.0)
        times = shift * times / (1 + (shift - 1) * times) * scheduler.get("s", 0.999)
        trajectory = [action.cpu().clone()]
        for t, tn in zip(times[:-1], times[1:], strict=True):
            angle = t * torch.exp(torch.arange(512, device=self.device) * (-np.log(10000) / 511))
            time = torch.cat((angle.sin(), angle.cos()))[None]
            velocity, _ = self.denoise(
                action,
                dof,
                time,
                cos[:, :, p:],
                sin[:, :, p:],
                torch.tensor([p], device=self.device),
                torch.tensor([32], device=self.device),
                cache,
            )
            velocity = velocity * dof + vpadding * (1 - dof)
            action = action + (tn - t) * velocity
            trajectory.append(action.cpu().clone())
        stats = self.processor.normalizers["action"][values["norm_key"]]
        actions = (action.cpu() + 1) / 2 * torch.tensor(stats["delta"]) + torch.tensor(stats["min"])
        return dict(
            trajectory=torch.stack(trajectory),
            vision=vision.cpu(),
            embeddings=embeddings.cpu(),
            actions=actions,
        )
