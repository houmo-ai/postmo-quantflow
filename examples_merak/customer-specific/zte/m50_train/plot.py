"""Plot existing runs; no Torch, xhquant, GPU or customer code is loaded."""
import argparse
import csv
import json

from common import prepare_output, repo_path


def load_run(directory):
    summary = json.loads((directory/"summary.json").read_text())
    protocol = summary["protocol"]
    rows = list(csv.DictReader((directory/"curve.csv").open()))
    if not rows:
        raise ValueError(f"No completed training steps: {directory}")
    initial = json.loads((directory/"initial_metrics.json").read_text())
    variant, bits, seed = protocol["variant"], protocol["matmul_bits"], protocol["seed"]
    label = f"A FP32 (seed {seed})" if variant=="A" else f"{variant} SEFP{bits} + FP{'32' if variant=='B' else '16'} update (seed {seed})"
    diagnostics = []
    if protocol.get("diagnostic_override"):
        diagnostics.append(protocol["diagnostic_override"])
    if variant != "A" and protocol.get("relu_path", "nl") != "nl":
        diagnostics.append("relu="+protocol["relu_path"])
    if diagnostics:
        label += " [DIAGNOSTIC: " + ", ".join(diagnostics) + "]"
    colors = {("A",16):"#222222",("A",8):"#222222",("B",16):"#1565c0",
              ("C",16):"#c62828",("B",8):"#2e7d32",("C",8):"#ef6c00"}
    return {"directory":directory,"summary":summary,"rows":rows,"initial":initial,
            "label":label,"color":colors[(variant,bits)],"style":"--" if bits==8 and variant!="A" else "-"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Training result directories relative to repository root")
    parser.add_argument("--output", default="work_dirs/zte_train/plots", help="New/empty plot directory")
    args = parser.parse_args()
    try:
        runs = [load_run(repo_path(path)) for path in args.runs]
        output = prepare_output(args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})

    def finish(fig, axes, name, title):
        for axis in axes:
            axis.set_xlabel("Training step")
            axis.grid(alpha=.2)
        fig.suptitle(title)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=min(3,len(runs)),
                   bbox_to_anchor=(.5,0))
        fig.tight_layout(rect=(0,.15,1,.94))
        for suffix in ("png","svg"):
            fig.savefig(output/f"{name}.{suffix}",dpi=180,bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1,3,figsize=(16,4.9))
    for run in runs:
        rows=run["rows"];steps=[int(row["step"]) for row in rows]
        values=([float(row["train_class_loss"]) for row in rows],
                [float(row["train_l2_kernel"])+float(row["train_l2_dense"]) for row in rows],
                [float(row["train_total_loss"]) for row in rows])
        for axis, series in zip(axes,values):
            axis.plot(steps,series,color=run["color"],ls=run["style"],label=run["label"],lw=1.4)
    for axis,title in zip(axes,("Label loss: training NLL only","Regularization: K L2 + Dense L2","Total loss: recorded training objective")):
        axis.set_title(title)
    for axis in (axes[0],axes[2]):
        axis.set_yscale("symlog",linthresh=1e-3)
    finish(fig,axes,"loss_components","GPU numerical simulation: label, regularization and total loss are distinct")

    fig, grid = plt.subplots(2,2,figsize=(12,7))
    for run in runs:
        for axis,key,title,scale in ((grid[0,0],"train_eval_ce","Training-set eval NLL",1),
                                     (grid[0,1],"val_ce","Validation NLL",1),
                                     (grid[1,0],"train_eval_accuracy","Training-set eval accuracy (%)",100),
                                     (grid[1,1],"val_accuracy","Validation accuracy (%)",100)):
            rows=[row for row in run["rows"] if row.get(key)]
            steps=[0]+[int(row["step"]) for row in rows]
            values=[scale*run["initial"][key]]+[scale*float(row[key]) for row in rows]
            axis.plot(steps,values,color=run["color"],ls=run["style"],label=run["label"])
            axis.set_title(title)
    finish(fig,list(grid.flat),"validation","Eval uses running statistics and no Dropout; NLL excludes regularization")

    fig,axes=plt.subplots(1,2,figsize=(12,4.9))
    for run in runs:
        rows=run["rows"];steps=[int(row["step"]) for row in rows]
        for axis,key in zip(axes,("gradient_zero_fraction","visible_parameter_update_fraction")):
            axis.plot(steps,[100*float(row[key]) for row in rows],color=run["color"],ls=run["style"],label=run["label"])
    axes[0].set_title("Exact-zero parameter gradients (%)")
    axes[1].set_title("Represented parameters changed at each step (%)")
    finish(fig,axes,"gradients_updates","A low update fraction alone does not prove numerical failure")
    comparison=[]
    for run in runs:
        row=run["rows"][-1]
        comparison.append({"run":str(run["directory"]),"label":run["label"],
                           "protocol":run["summary"]["protocol"],
                           "final_training_label_loss":float(row["train_class_loss"]),
                           "final_training_regularization":float(row["train_l2_kernel"])+float(row["train_l2_dense"]),
                           "final_training_total_loss":float(row["train_total_loss"]),
                           "final_validation":run["summary"]["final"],
                           "classification_gate_passed":run["summary"]["passed"],
                           "note":"classification gate is distinct from full regularized objective behavior"})
    (output/"comparison.json").write_text(json.dumps(comparison,indent=2))
    print(output)


if __name__=="__main__":
    main()
