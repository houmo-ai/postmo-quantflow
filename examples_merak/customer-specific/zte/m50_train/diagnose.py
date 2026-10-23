"""Reproduce ReLU selection and FP16 Adam errors without changing training."""
import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

from common import (DEFAULT_CUSTOMER_ROOT, REPO_ROOT, configure_customer_sources,
                    prepare_output, repo_path, load_customer_modules)


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="work_dirs/zte_train/runs/diagnostics")
    parser.add_argument("--customer-sources", default=str(DEFAULT_CUSTOMER_ROOT))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.steps < 1 or args.seed < 0:
        parser.error("steps must be positive and seed nonnegative")
    try:
        files = configure_customer_sources(repo_path(args.customer_sources))
        output = prepare_output(args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(REPO_ROOT))
    from trainer import Experiment, dataset, PARAMETER_NAMES
    ReferenceAdam = load_customer_modules()[0].ReferenceAdam
    from runtime import XH2Runtime, adam_probe_step
    import torch
    from xhquant.backend.xh2a import functions as xf

    protocol = {"scope": "VP/NL ReLU; constant-gradient Adam; C16 with same-gradient FP32 Adam shadow",
                "seed": args.seed, "steps": args.steps, "batch": 32,
                "customer_sources": {k: str(v) for k, v in files.items()},
                "shadow": "monitor only; never feeds the training graph"}
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    rt = XH2Runtime()
    scalar = lambda value: torch.tensor(value, device="cuda", dtype=torch.float16)
    x = torch.tensor([-2., -.1, 0., .001, .1, 1., 2., 8., 16., 32.], device="cuda", dtype=torch.float16)
    activation = {"x": x.tolist(), "vp": xf.relu_xh2a_vp_default(x).tolist(),
                  "nl": xf.relu_xh2a_nl_default(x, *rt.luts["relu"]).tolist(),
                  "fp32_reference": x.float().relu().tolist()}
    (output/"relu_minimal.json").write_text(json.dumps(activation, indent=2))
    scalar_rows = []
    for gradient in (1e-5, 1e-4, .001, .002, .003, .004, .005, .006, .01, .1):
        g = scalar(gradient)
        w, m, v, c1, c2 = scalar(.05), scalar(0), scalar(0), scalar(0), scalar(0)
        master = torch.nn.Parameter(w.float().clone())
        optimizer = ReferenceAdam([master])
        for step in range(1, args.steps+1):
            before = master.detach().clone()
            master.grad = g.float().clone()
            optimizer.step()
            values = adam_probe_step(w, g, m, v, c1, c2)
            new, m, v, c1, c2 = values[:5]
            scalar_rows.append({"gradient": g.item(), "step": step, "w": w.item(),
                                "m": m.item(), "v": v.item(), "numerator": values[7].item(),
                                "denominator": values[8].item(), "requested_update": values[9].item(),
                                "actual_update": (w.float()-new.float()).item(),
                                "same_gradient_fp32_update": (before-master).item()})
            w = new
    write_csv(output/"adam_constant_gradient.csv", scalar_rows)
    experiment = Experiment(SimpleNamespace(variant="C", bits=16, seed=args.seed,
                                           diagnostic_override=None, relu_path="nl", output=output))
    masters = [torch.nn.Parameter(p.detach().float().clone()) for p in experiment.visible_parameters()]
    optimizer = ReferenceAdam(masters)
    x, y = dataset(576, 20261022)
    x, y = x.cuda(), y.cuda()
    rows, per_layer = [], []
    for step in range(args.steps):
        indices = (torch.arange(32, device="cuda")+step*32) % len(y)
        old = [p.detach().clone() for p in experiment.visible_parameters()]
        row = experiment.train_step(x[indices], y[indices])
        before = [p.detach().clone() for p in masters]
        for p, node in zip(masters, experiment.model.nodes):
            p.grad = node.grad.tensor.float().clone()
        optimizer.step()
        details = []
        for name, a, b, fp0, fp1, v in zip(PARAMETER_NAMES, old, experiment.visible_parameters(),
                                         before, masters, experiment.model.v):
            actual, expected = a.float()-b.float(), fp0-fp1.detach()
            details.append({"step": step+1, "parameter": name,
                            "v_zero_fraction": (v.tensor==0).float().mean().item(),
                            "low_actual_update_abs_max": actual.abs().max().item(),
                            "fp32_same_gradient_update_abs_max": expected.abs().max().item(),
                            "low_update_over_001_fraction": (actual.abs()>.01).float().mean().item(),
                            "low_update_zero_fraction": (actual==0).float().mean().item(),
                            "fp32_update_zero_fraction": (expected==0).float().mean().item()})
        per_layer.extend(details)
        row["max_low_update"] = max(d["low_actual_update_abs_max"] for d in details)
        row["max_same_gradient_fp32_update"] = max(d["fp32_same_gradient_update_abs_max"] for d in details)
        rows.append(row)
    write_csv(output/"optimizer_shadow_curve.csv", rows)
    write_csv(output/"optimizer_shadow_layers.csv", per_layer)
    result = {"relu": activation,
              "scalar_adam_max_update": max(abs(r["actual_update"]) for r in scalar_rows),
              "full_model_low_max_update": max(r["max_low_update"] for r in rows),
              "full_model_same_gradient_fp32_max_update": max(r["max_same_gradient_fp32_update"] for r in rows),
              "full_model_initial_loss": rows[0]["train_total_loss"],
              "full_model_final_loss": rows[-1]["train_total_loss"],
              "full_model_final_l2k": rows[-1]["train_l2_kernel"]}
    (output/"summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
