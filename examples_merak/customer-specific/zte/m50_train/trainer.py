"""Customer full-model controls and XH2 explicit-DAG training experiments.

Run each variant in a fresh process. Outputs/protocol.json is written before
any model result. No product source or customer source is modified.
"""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

from runtime import (PARAMETER_NAMES, FloatRuntime, XH2Runtime,
                                make_model, loss_parts)
import torch
from common import (REPO_ROOT, EXAMPLE_DIR, load_customer_modules, verify_customer_sources,
                    dataset, rng_state, restore_rng, git_revision)


def measure(a, b):
    a, b = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    return {"max_abs": (a-b).abs().max().item(),
            "relative_l2": ((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
            "cosine": torch.nn.functional.cosine_similarity(a,b,dim=0,eps=1e-12).item()}


def stats(tensors):
    a = torch.cat([v.detach().reshape(-1) for v in tensors])
    return {"finite": bool(a.isfinite().all()), "zero_fraction": (a==0).float().mean().item(),
            "saturation_fraction": (a.abs()>=65504).float().mean().item(),
            "abs_max": a.abs().max().item(), "l2": a.float().norm().item()}


def seeded_reference(seed):
    reference, _ = load_customer_modules()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return reference.SequenceClassifier(normalization="variance").cuda()


def check_float(seed, x, y, out):
    results = []
    for dropout in (False, True):
        native = seeded_reference(seed)
        rt = FloatRuntime()
        model = make_model(rt, native, seed)
        saved_rng = rng_state()
        native.train()
        lp = native(x, dropout=dropout)
        loss = native.loss(lp, y)
        loss.backward()
        restore_rng(saved_rng)
        with torch.no_grad():
            manual_lp = model.forward(rt.upload(x.reshape(-1,13)), True, dropout)
            manual_loss = model.loss(manual_lp, rt.upload(torch.nn.functional.one_hot(y,9)))
            rt.phase = "backward"
            manual_loss.backward()
        gradients = {n: measure(node.grad.tensor, p.grad) for n,node,p in zip(PARAMETER_NAMES,model.nodes,native.parameters())}
        grad_ok = all(torch.allclose(node.grad.tensor.reshape_as(p.grad),p.grad,atol=1e-5,rtol=1e-3)
                      for node,p in zip(model.nodes,native.parameters()))
        row = {"dropout": dropout, "log_probabilities": measure(manual_lp.a.tensor,lp),
               "loss": measure(manual_loss.a.tensor,loss), "gradients": gradients,
               "running_mean": measure(model.batch_statistics[0].tensor,native.saved_mean/.01),
               "passed": grad_ok and torch.allclose(manual_lp.a.tensor,lp,atol=1e-5,rtol=1e-4)
                         and torch.allclose(manual_loss.a.tensor,loss.reshape(1,1),atol=1e-5,rtol=1e-4)}
        expected_mean = .01*model.batch_statistics[0].tensor
        expected_var = 1+.01*(model.batch_statistics[1].tensor-1)
        row["running_state_matches"] = (torch.allclose(expected_mean.reshape(-1),native.saved_mean,atol=1e-6,rtol=1e-4)
                                        and torch.allclose(expected_var.reshape(-1),native.saved_spread,atol=1e-6,rtol=1e-4))
        row["passed"] = bool(row["passed"] and row["running_state_matches"])
        results.append(row)
    result = {"seed":seed,"checks":results,"all_passed":all(r["passed"] for r in results),
              "criterion":"all parameter gradients allclose atol=1e-5 rtol=1e-3; lp/loss atol=1e-5 rtol=1e-4; both dropout off/on"}
    (out/"float_gradient_check.json").write_text(json.dumps(result,indent=2))
    if not result["all_passed"]:
        raise RuntimeError("Customer explicit backward disagrees with native autograd")
    print("FP32 explicit-DAG vs original autograd: PASS (Dropout off/on)",flush=True)


class Experiment:
    def __init__(self, args):
        self.args = args
        reference, _ = load_customer_modules()
        self.reference = seeded_reference(args.seed)
        self.rt = None
        if args.variant == "A":
            self.optimizer = reference.ReferenceAdam(self.reference.parameters())
        else:
            self.rt = XH2Runtime(args.bits, args.diagnostic_override,args.relu_path)
            self.model = make_model(self.rt, self.reference, args.seed)
            if args.variant == "B":
                self.masters = [torch.nn.Parameter(p.detach().clone().reshape(a.shape))
                                for p,a in zip(self.reference.parameters(),self.model.parameters)]
                self.optimizer = reference.ReferenceAdam(self.masters)
                self.bn_master = [self.reference.saved_mean.detach().clone().reshape(1,-1),
                                  self.reference.saved_spread.detach().clone().reshape(1,-1)]
        self.step = 0

    def visible_parameters(self):
        return list(self.reference.parameters()) if self.rt is None else [p.tensor for p in self.model.parameters]

    def checkpoint(self):
        state = {"step":self.step,"rng":rng_state(),
                 "contract":{"variant":self.args.variant,"bits":self.args.bits,"seed":self.args.seed,
                             "diagnostic_override":self.args.diagnostic_override,"relu_path":self.args.relu_path},
                 "minimum":self.reference.minimum.clone(),"span":self.reference.span.clone()}
        if self.rt is None:
            state.update(native=copy.deepcopy(self.reference.state_dict()),optimizer=copy.deepcopy(self.optimizer.state_dict()))
        else:
            state["states"] = [a.tensor.clone() for a in self.model.all_states()]
            if self.args.variant == "B":
                state.update(masters=[p.detach().clone() for p in self.masters],
                             optimizer=copy.deepcopy(self.optimizer.state_dict()),
                             bn_master=[v.clone() for v in self.bn_master])
        return state

    def restore(self, state):
        if state["contract"] != {"variant":self.args.variant,"bits":self.args.bits,"seed":self.args.seed,
                                 "diagnostic_override":self.args.diagnostic_override,"relu_path":self.args.relu_path}:
            raise ValueError("Checkpoint experiment contract does not match")
        self.reference.minimum.copy_(state["minimum"])
        self.reference.span.copy_(state["span"])
        self.step = state["step"]
        if self.rt is None:
            self.reference.load_state_dict(state["native"])
            self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        else:
            self.model.step = self.step
            self.model.minimum.tensor = state["minimum"].reshape(1,-1).to(self.rt.dtype).clone()
            self.model.span.tensor = state["span"].reshape(1,-1).to(self.rt.dtype).clone()
            for a,t in zip(self.model.all_states(), state["states"]):
                a.tensor = t.clone()
            if self.args.variant == "B":
                for p,t in zip(self.masters,state["masters"]):
                    p.data.copy_(t)
                self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
                self.bn_master = [v.clone() for v in state["bn_master"]]
        restore_rng({k:v.cpu() for k,v in state["rng"].items()})

    @torch.no_grad()
    def evaluate(self, x, y):
        outputs = []
        if self.rt is not None:
            self.rt.trace = None
        # B32 fixed inference shape, independent from B32 training statistics.
        for bx in x.split(32):
            if self.rt is None:
                self.reference.eval()
                outputs.append(self.reference(bx))
            else:
                self.rt.phase = "eval"
                outputs.append(self.model.forward(self.rt.upload(bx.reshape(-1,13)),False).a.tensor)
        lp = torch.cat(outputs).float()
        return {"ce":torch.nn.functional.nll_loss(lp,y).item(),
                "accuracy":(lp.argmax(1)==y).float().mean().item(),
                "output_finite":bool(lp.isfinite().all()),
                "exp_log_probability_sum_mean":lp.exp().sum(1).mean().item()}

    @torch.no_grad()
    def low_update(self):
        rt,model = self.rt,self.model
        rt.phase = "update"
        op = lambda a,b,k: rt.binary(a,b,k)
        const = lambda v: rt.full((1,1),v)
        add = lambda a,b:op(a,b,0)
        sub = lambda a,b:op(a,b,1)
        mul = lambda a,b:op(a,b,2)
        div = lambda a,b:op(a,b,3)
        c1 = rt.wrap(model.c.tensor[:,0:1])
        c2 = rt.wrap(model.c.tensor[:,1:2])
        c1 = add(c1,mul(sub(const(1),c1),const(.1)))
        c2 = add(c2,mul(sub(const(1),c2),const(.001)))
        rate = div(mul(const(1e-4),rt.unary(c2,4)),c1)
        for state,target in zip((model.bn_mean,model.bn_var),model.batch_statistics):
            state.assign(add(state,mul(sub(target,state),const(.01))))
        detail = {}
        for name,p,node,m,v in zip(PARAMETER_NAMES,model.parameters,model.nodes,model.m,model.v):
            g = node.grad
            m.assign(add(m,mul(sub(g,m),const(.1))))
            square = mul(g,g)
            v_increment = mul(sub(square,v),const(.001))
            v.assign(add(v,v_increment))
            numerator = mul(rate,m)
            denominator = add(rt.unary(v,4),const(1e-7))
            update = div(numerator,denominator)
            new = sub(p,update)
            if self.step == 0:
                detail[name] = {"gradient":stats([g.tensor]),"square":stats([square.tensor]),
                                "v_increment":stats([v_increment.tensor]),
                                "numerator":stats([numerator.tensor]),"update":stats([update.tensor]),
                                "nonzero_update_absorbed_fraction":((update.tensor!=0)&(new.tensor==p.tensor)).float().mean().item()}
            p.assign(new)
        model.c.tensor = torch.cat([c1.tensor,c2.tensor,torch.zeros_like(c1.tensor)],1)
        return detail

    def train_step(self, x, y, trace=False):
        old = [p.detach().clone() for p in self.visible_parameters()]
        if self.rt is None:
            self.reference.train()
            self.optimizer.zero_grad(set_to_none=True)
            lp = self.reference(x)
            loss = self.reference.loss(lp,y)
            ce = torch.nn.functional.nll_loss(lp,y)
            l2k = .01*self.reference.kernel.square().sum()
            l2d = .001*self.reference.dense.square().sum()
            loss.backward()
            grads = [p.grad for p in self.reference.parameters()]
            gradstats = stats(grads)
            if not gradstats["finite"] or not bool(loss.isfinite()):
                raise FloatingPointError("nonfinite native loss/gradient")
            self.optimizer.step()
            extra = {}
        else:
            rt,model = self.rt,self.model
            rt.phase = "forward"
            rt.trace = [] if trace else None
            before_rng = rng_state() if trace else None
            with torch.no_grad():
                lp_node = model.forward(rt.upload(x.reshape(-1,13)),True)
                loss_node,parts = loss_parts(model,lp_node,rt.upload(torch.nn.functional.one_hot(y,9)))
                rt.phase = "backward"
                loss_node.backward()
                lp,loss = lp_node.a.tensor,loss_node.a.tensor
                ce,l2k,l2d = (v.a.tensor for v in parts)
                grads = [n.grad.tensor for n in model.nodes]
                gradstats = stats(grads)
                if not gradstats["finite"] or not bool(loss.isfinite().all()):
                    (self.args.output/"failure_diagnostics.json").write_text(json.dumps({
                        "step":self.step,"loss":loss.item(),
                        "gradients":{n:stats([g]) for n,g in zip(PARAMETER_NAMES,grads)},
                        "batch_mean":stats([model.batch_statistics[0].tensor]),
                        "batch_variance":stats([model.batch_statistics[1].tensor])},indent=2))
                    raise FloatingPointError("nonfinite XH2 loss/gradient; no optimizer step")
                if trace:
                    after_rng = rng_state()
                    control_rt = FloatRuntime()
                    control_rt.trace = []
                    control = make_model(control_rt,self.reference,self.args.seed)
                    for a,b in zip(control.parameters,model.parameters):
                        a.tensor = b.tensor.float().clone()
                    restore_rng(before_rng)
                    control_lp = control.forward(control_rt.upload(x.reshape(-1,13).half().float()),True)
                    control_loss = control.loss(control_lp,control_rt.upload(torch.nn.functional.one_hot(y,9)))
                    control_rt.phase = "backward"
                    control_loss.backward()
                    restore_rng(after_rng)
                    assert [k for k,_ in rt.trace] == [k for k,_ in control_rt.trace]
                    (self.args.output/"key_layer_errors.json").write_text(json.dumps({
                        "reference":"FP32 explicit graph on identical FP16-represented initial parameters/input and identical dropout mask",
                        "log_probabilities":measure(lp,control_lp.a.tensor),
                        "gradients":{n:measure(a,b.grad.tensor) for n,a,b in zip(PARAMETER_NAMES,grads,control.nodes)},
                        "intermediates":[{"index":i,"operation":k,"shape":list(a.shape),**measure(a,b),
                                          "xh2_stats":stats([a])} for i,((k,a),(_,b)) in enumerate(zip(rt.trace,control_rt.trace))]
                    },indent=2))
                    rt.trace = None
                if self.args.variant == "B":
                    for p,g in zip(self.masters,grads):
                        p.grad = g.float().clone()
                    self.optimizer.step()
                    for a,p in zip(model.parameters,self.masters):
                        a.tensor = p.detach().half().clone()
                    for master,state,target in zip(self.bn_master,(model.bn_mean,model.bn_var),model.batch_statistics):
                        master.add_(.01*(target.tensor.float()-master))
                        state.tensor = master.half().clone()
                    extra = {}
                else:
                    extra = self.low_update()
                model.step += 1
        self.step += 1
        new = self.visible_parameters()
        parameterstats = stats(new)
        if not parameterstats["finite"]:
            raise FloatingPointError("nonfinite weights after update")
        if self.rt is not None and not stats([a.tensor for a in self.model.all_states()])["finite"]:
            raise FloatingPointError("nonfinite model/optimizer state after update")
        total = sum(p.numel() for p in new)
        changed = sum((a!=b).sum().item() for a,b in zip(old,new))
        row = {"step":self.step,"train_total_loss":loss.item(),"train_class_loss":ce.item(),
               "train_l2_kernel":l2k.item(),"train_l2_dense":l2d.item(),
               "train_batch_accuracy":(lp.argmax(1)==y).float().mean().item(),
               "gradient_finite":gradstats["finite"],"gradient_zero_fraction":gradstats["zero_fraction"],
               "gradient_saturation_fraction":gradstats["saturation_fraction"],
               "gradient_l2":gradstats["l2"],"visible_parameter_update_fraction":changed/total,
               "parameter_finite":parameterstats["finite"],
               "parameter_saturation_fraction":parameterstats["saturation_fraction"]}
        if self.step == 1:
            (self.args.output/"first_step_diagnostics.json").write_text(json.dumps({
                "gradients":{n:stats([g]) for n,g in zip(PARAMETER_NAMES,grads)},
                "low_optimizer":extra,"operations":self.rt.counts if self.rt else None},indent=2))
        return row


def equal_nested(a,b):
    if isinstance(a,torch.Tensor):
        return torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal_nested(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
    return a==b


def run(args):
    SOURCE = verify_customer_sources()["reference"]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    protocol = {"variant":args.variant,"seed":args.seed,"steps":args.steps,"batch":32,
                "train_size":576,"validation_size":288,"data_seeds":[20261022,20261023],
                "normalization":"variance, biased along batch, epsilon .001",
                "lr":1e-4,"eps":1e-7,"dropout":.2,"gradient_seed":1,
                "master_and_optimizer_dtype":"FP32" if args.variant in ("A","B") else "FP16",
                "forward_backward":"customer torch autograd FP32" if args.variant=="A" else "customer explicit Node DAG, XH2 ALIGNED primitives",
                "matmul_bits":args.bits,"dynamic_sefp_group":64,"fast_matmul":False,
                "mean_boundary":"customer torch mean" if args.variant=="A" else "customer sum then div, not SDK dedicated early-scaled mean",
                "acceptance":{"final_validation_accuracy_min":.8,"relative_validation_ce_drop_min":.3,
                              "finite_all_steps":True,"changed_weights":True,"exact_reload_resume":True},
                "diagnostic_override":args.diagnostic_override,
                "relu_path":args.relu_path,
                "dropout_rng":"GPU Torch FP32 dropout" if args.variant=="A" else "GPU Torch FP32 mask generation, then exact FP16 0/1.25 representation; XH2 mask multiply in forward/backward",
                "gradient_convention":"original PyTorch autograd" if args.variant=="A" else "analytic surrogate using LUT output, evaluated via XH2 primitives; no automatic float backward",
                "scope":"GPU numerical simulation on fixed easy synthetic data; no M50 device or business accuracy claim",
                "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2))
    (args.output/"code").mkdir()
    for script in tuple(EXAMPLE_DIR / name for name in ("train.py", "trainer.py", "runtime.py", "common.py", "customer_sources.json")):
        (args.output/"code"/script.name).write_bytes(script.read_bytes())
    import xhquant
    sdk = Path(xhquant.__file__).resolve().parents[1]
    source_paths = [p for subtree in (sdk/"xhquant/backend/xh2a",sdk/"xhquant/backend/hmops/accu",
                                     sdk/"xhquant/quantization/xh2a/quantizer")
                    for p in subtree.rglob("*") if p.is_file() and p.suffix in (".py",".cu",".cpp",".h")]
    revisions = {"sdk":git_revision(sdk),"repository":git_revision(REPO_ROOT)}
    provenance = {"sdk_revision":revisions["sdk"]["revision"],
                  "repository_revision":revisions["repository"]["revision"],
                  "revision_diagnostics":revisions,
                  "sdk_source_sha256":{str(p.relative_to(sdk)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                  "customer_node_sha256":hashlib.sha256(verify_customer_sources()["autograd"].read_bytes()).hexdigest(),
                  "customer_files":{k:str(v) for k,v in verify_customer_sources().items()},
                  "reduction_binary_limit":"Uses SDK GPU hsum; inspect loader stderr for rebuild/binary diagnostics. This example has no native float reduction fallback."}
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2))
    train_x,train_y = dataset(576,20261022)
    val_x,val_y = dataset(288,20261023)
    train_x,train_y,val_x,val_y = [v.cuda() for v in (train_x,train_y,val_x,val_y)]
    check_float(args.seed,train_x[:32],train_y[:32],args.output)
    if args.check_only:
        return
    experiment = Experiment(args)
    initial = experiment.evaluate(val_x,val_y)
    initial_weights = [p.detach().clone() for p in experiment.visible_parameters()]
    rows,errors = [],[]
    initial_eval = {"step":0,**{"val_"+k:v for k,v in initial.items()},
                    **{"train_eval_"+k:v for k,v in experiment.evaluate(train_x,train_y).items()}}
    (args.output/"initial_metrics.json").write_text(json.dumps(initial_eval,indent=2))
    started = time.perf_counter()
    path = args.output/"curve.csv"
    try:
        for step in range(args.steps):
            indices = (torch.arange(32,device="cuda")+step*32)%len(train_y)
            row = experiment.train_step(train_x[indices],train_y[indices],trace=(step==0))
            if (step+1)%25==0 or step==0 or step+1==args.steps:
                row.update({"val_"+k:v for k,v in experiment.evaluate(val_x,val_y).items()})
                row.update({"train_eval_"+k:v for k,v in experiment.evaluate(train_x,train_y).items()})
                print(json.dumps({"variant":args.variant,"bits":args.bits,**row}),flush=True)
            rows.append(row)
            # Persist each step, including evidence before a later failure.
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with path.open("w") as f:
                writer = csv.DictWriter(f,fieldnames=fields)
                writer.writeheader();writer.writerows(rows)
    except Exception as error:
        errors.append({"step":experiment.step,"type":type(error).__name__,"message":str(error)})
        import traceback
        (args.output/"failure.txt").write_text(traceback.format_exc())
        print("RUN FAILED "+str(error),flush=True)
    seconds = time.perf_counter()-started
    try:
        final = experiment.evaluate(val_x,val_y)
    except Exception as error:
        errors.append({"step":experiment.step,"phase":"final_evaluation","type":type(error).__name__,"message":str(error)})
        final = {"ce":float("nan"),"accuracy":0.,"output_finite":False}
    changed = sum((p!=old).sum().item() for p,old in zip(experiment.visible_parameters(),initial_weights))
    checkpoint = experiment.checkpoint()
    torch.save(checkpoint,args.output/"checkpoint.pt")
    reload_equal = resume_equal = False
    if not errors:
        try:
            saved = torch.load(args.output/"checkpoint.pt",map_location="cuda",weights_only=True)
            eval_before = experiment.evaluate(val_x,val_y)
            # A new instance validates actual reconstruction, not only overwriting
            # the original object's state. RNG restored after its initialization.
            restored = Experiment(args)
            restored.restore(saved)
            reload_equal = equal_nested(eval_before,restored.evaluate(val_x,val_y))
            bx,by = train_x[:32],train_y[:32]
            row_a = experiment.train_step(bx,by)
            state_a = experiment.checkpoint()
            restored.restore(saved)
            row_b = restored.train_step(bx,by)
            resume_equal = equal_nested(row_a,row_b) and equal_nested(state_a,restored.checkpoint())
            experiment.restore(saved)
        except Exception as error:
            errors.append({"step":experiment.step,"phase":"resume_verification",
                           "type":type(error).__name__,"message":str(error)})
            import traceback
            (args.output/"failure.txt").write_text(traceback.format_exc())
            print("RESUME VERIFICATION FAILED "+str(error),flush=True)
    drop = (initial["ce"]-final["ce"])/max(abs(initial["ce"]),1e-12)
    result = {"protocol":protocol,"initial":initial,"final":final,"relative_validation_ce_drop":drop,
              "completed_steps":len(rows),"changed_parameters":changed,"reload_equal":reload_equal,
              "resume_equal":resume_equal,"seconds":seconds,"errors":errors,
              "passed":not errors and len(rows)==args.steps and drop>=.3 and final["accuracy"]>=.8
                        and changed>0 and reload_equal and resume_equal,
              "environment":{"torch":torch.__version__,"cuda":torch.version.cuda,
                             "triton":__import__("triton").__version__,"gpu":torch.cuda.get_device_name(),
                             "python":platform.python_version(),"CUDA_VISIBLE_DEVICES":os.getenv("CUDA_VISIBLE_DEVICES"),
                             "CUBLAS_WORKSPACE_CONFIG":os.getenv("CUBLAS_WORKSPACE_CONFIG"),
                             "tf32":False,"deterministic":True},
              "script_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               tuple(EXAMPLE_DIR / name for name in ("train.py", "trainer.py", "runtime.py", "common.py"))},
              "command":sys.argv}
    (args.output/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if errors:
        raise SystemExit("Training execution failed; inspect summary.json and failure.txt")
