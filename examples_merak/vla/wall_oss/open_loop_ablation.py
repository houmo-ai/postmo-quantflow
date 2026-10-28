"""Replay fixed paired probes with an existing alternative precision package."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .open_loop import identity, input_hash, numeric_metrics, write_json


def main():
    from xhmodel_merak.xh_other_model.models.wall_oss.runtime import WallRuntime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    identifiers = set(map(int, args.ids.split(",")))
    runtime = WallRuntime(args.artifact_dir, device=args.device)
    name = runtime.meta["quant_type"]
    out = run / "ablation" / name
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in manifest["samples"]:
        if sample["id"] not in identifiers:
            continue
        expected = torch.load(run / "float" / f"{sample['id']:03d}.pt", weights_only=True)
        native = torch.load(run / "official" / f"{sample['id']:03d}.pt", weights_only=True)
        images = [Image.open(p).convert("RGB") for p in sample["images"]]
        data = runtime.processor.prepare(
            sample["instruction"], images, sample["state"], sample["norm_key"], sample["state_mask"]
        )
        if input_hash(data) != native["input_hash"] or native["protocol_hash"] != manifest["protocol_hash"]:
            raise ValueError("ablation inputs differ from the registered paired protocol")
        call = {k: sample[k] for k in ("instruction", "state", "norm_key", "state_mask", "dof_mask", "seed", "steps")}
        actions, stats = runtime.predict(**call, images=images, noise=native["noise"])
        trajectory = torch.stack(runtime.last_trace)
        mask = np.asarray(sample["dof_mask"], bool)
        record = dict(
            id=sample["id"],
            final=numeric_metrics(trajectory[-1], expected["trajectory"][-1], mask),
            trajectory=numeric_metrics(trajectory[1:], expected["trajectory"][1:], mask),
        )
        torch.save(
            dict(actions=torch.from_numpy(actions), trajectory=trajectory, stats=stats), out / f"{sample['id']:03d}.pt"
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    report = dict(
        precision=name,
        samples=len(records),
        protocol_hash=manifest["protocol_hash"],
        artifact_dir=str(args.artifact_dir.resolve()),
        artifact_files=identity(args.artifact_dir),
        selection="five lowest W8A8 cosine cases plus one median-RMSE case; diagnostic subset, not a full-suite score",
        cases=records,
    )
    write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
