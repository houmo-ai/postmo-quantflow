"""Isolated adapter for the unmodified customer reverse-mode Node DAG.

FloatRuntime is a validation control. XH2Runtime calls the SDK ALIGNED
numerical primitives; no torch autograd is used for its forward/backward.
Structural indexing, allocations, masks and monitoring remain on the host/GPU.
"""
import os
import sys
from functools import lru_cache
from types import SimpleNamespace

os.environ["USE_FAST_TRITON_MATMUL"] = "0"
os.environ["USE_TRITON_MATMUL"] = "1"
from common import REPO_ROOT, load_customer_modules
sys.path.insert(0, str(REPO_ROOT))

import torch
@lru_cache(maxsize=1)
def _xh2_backend():
    # CPU contracts exercise FloatRuntime without importing CUDA SDK modules.
    # Real XH2 execution always loads and validates the actual SDK below.
    from xhquant.backend.xh2a import functions
    from xhquant.backend.xh2a.functions.vp import arithmetic
    from xhquant.backend.xh2a.functions.nl.base import generate_lut_table, verbose_lut
    from xhquant.backend.xh2a.functions.te import xh2a_matmul_triton
    from xhquant.backend.xh2a.ir.operators.matmul import matmul_xh2a_default
    from xhquant.backend.xh2a.ir.operators.activate import generate_relu_lut_table
    return SimpleNamespace(functions=functions, arithmetic=arithmetic,
                           implementation=xh2a_matmul_triton,
                           generate_lut_table=generate_lut_table, verbose_lut=verbose_lut,
                           matmul=matmul_xh2a_default, relu_table=generate_relu_lut_table)

PARAMETER_NAMES = ("kernel", "recurrent", "bias", "dense", "dense_bias",
                   "gamma", "beta", "head", "head_bias")


class Array:
    def __init__(self, rt, tensor):
        self.rt, self.tensor = rt, tensor

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def size(self):
        return self.tensor.numel()

    def assign(self, other):
        self.tensor = other.tensor.clone()

    def get(self):
        return self.tensor.detach().cpu().numpy()


class FloatRuntime:
    dtype = torch.float32

    def __init__(self, bits=16):
        self.bits = bits
        self.constants = {}
        self.trace = None
        self.phase = "forward"
        self.counts = {}

    def wrap(self, tensor):
        return Array(self, tensor)

    def upload(self, tensor, *_):
        return self.wrap(torch.as_tensor(tensor, device="cuda", dtype=self.dtype).clone())

    def full(self, shape, value, *_):
        key = (tuple(shape), float(value))
        if key not in self.constants:
            self.constants[key] = self.wrap(torch.full(shape, value, device="cuda", dtype=self.dtype))
        # These constants are immutable in the Node graph; Model states are
        # assigned new tensors, never changed through cached constant storage.
        return self.wrap(self.constants[key].tensor)

    def record(self, kind, tensor):
        key = self.phase + ":" + kind
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.trace is not None and kind in ("matmul", "rsqrt", "relu", "exp", "log"):
            self.trace.append((key, tensor.detach().clone()))
        return self.wrap(tensor)

    def binary(self, a, b, op):
        a, b = a.tensor, b.tensor
        out = (lambda: a+b, lambda: a-b, lambda: a*b, lambda: a/b)[op]()
        return self.record(("add", "sub", "mul", "div")[op], out)

    def unary(self, a, op):
        a = a.tensor
        functions = (torch.sigmoid, torch.tanh, torch.exp, torch.log, torch.sqrt,
                     torch.rsqrt, torch.neg, torch.relu, lambda x: (x > 0).to(x.dtype))
        return self.record(("sigmoid", "tanh", "exp", "log", "sqrt", "rsqrt", "neg", "relu", "relu_mask")[op], functions[op](a))

    def matmul(self, a, b, ta=False, tb=False):
        a, b = a.tensor, b.tensor
        return self.record("matmul", (a.T if ta else a) @ (b.T if tb else b))

    def reduce(self, a, axis, maximum=False):
        out = a.tensor.amax(axis, keepdim=True) if maximum else a.tensor.sum(axis, keepdim=True)
        return self.record("max" if maximum else "sum", out)

    def slice(self, a, start, count):
        return self.wrap(a.tensor[:, start:start+count])

    def put(self, a, cols, start):
        out = torch.zeros(a.shape[0], cols, device="cuda", dtype=self.dtype)
        out[:, start:start+a.shape[1]] = a.tensor
        return self.wrap(out)

    def time(self, a, t):
        return self.wrap(a.tensor.reshape(-1, 9, 13)[:, t, :])

    def dropout(self, shape, seed):
        # Deliberately match the native reference Torch RNG/mask distribution,
        # not the customer's custom integer-runtime PRNG bit sequence.
        del seed
        # CUDA dropout consumes randomness differently for half/float inputs.
        # Generate in FP32 for every variant, then represent the exact 0/1.25
        # values in the runtime dtype. This matches native FP32 masks and RNG.
        mask = torch.nn.functional.dropout(torch.ones(shape, device="cuda", dtype=torch.float32), .2, True)
        return self.wrap(mask.to(self.dtype))


class XH2Runtime(FloatRuntime):
    dtype = torch.float16

    def __init__(self, bits=16, diagnostic_override=None, relu_path="nl"):
        super().__init__(bits)
        self.backend = _xh2_backend()
        impl = self.backend.implementation
        assert impl.IF_USE_TRITON and not impl.USE_FAST_TRITON_MATMUL
        self.luts = {n: self.backend.verbose_lut(self.backend.generate_lut_table(name=n, device=torch.device("cpu")))
                     for n in ("sigmoid", "tanh", "exp", "log")}
        self.luts["relu"] = self.backend.relu_table()
        # Only explicit diagnostic runs may substitute a category with FP32
        # arithmetic followed by FP16 representation. Main B/C never override.
        self.diagnostic_override = diagnostic_override
        self.relu_path = relu_path

    def binary(self, a, b, op):
        xf = self.backend.functions
        out = (xf.add_xh2a_default, xf.sub_xh2a_default,
               xf.mul_xh2a_default, xf.div_xh2a_default)[op](a.tensor, b.tensor, saturate=False)
        return self.record(("add", "sub", "mul", "div")[op], out)

    def unary(self, a, op):
        xf, vp = self.backend.functions, self.backend.arithmetic
        x = a.tensor
        if op < 4:
            names = ("sigmoid", "tanh", "exp", "log")
            functions = (xf.sigmoid_xh2a_default, xf.tanh_xh2a_default,
                         xf.exp_xh2a_default, xf.log_xh2a_default)
            y = functions[op](x.contiguous(), *self.luts[names[op]])
        elif op == 4:
            y = vp.sqrt_xh2a_default(x)
        elif op == 5:
            y = vp.rsqrt_xh2a_default(x.contiguous(), type="quick")
        elif op == 6:
            y = vp.neg_xh2a_default(x)
        elif op == 7:
            y = (xf.relu_xh2a_nl_default(x.contiguous(), *self.luts["relu"])
                 if self.relu_path == "nl" else xf.relu_xh2a_vp_default(x))
        elif op == 8:
            # Comparison has no differentiable arithmetic. Use the SDK mask.
            from xhquant.backend.xh2a.functions.vp.compare import greater_xh2a_default
            y = greater_xh2a_default(x, torch.zeros_like(x)).to(torch.float16)
        else:
            raise ValueError(op)
        return self.record(("sigmoid", "tanh", "exp", "log", "sqrt", "rsqrt", "neg", "relu", "relu_mask")[op], y)

    def matmul(self, a, b, ta=False, tb=False):
        a, b = a.tensor, b.tensor
        a, b = (a.T if ta else a), (b.T if tb else b)
        if self.diagnostic_override == self.phase + "_matmul_fp32":
            y = (a.float() @ b.float()).half()
        else:
            y = self.backend.matmul(a.contiguous(), b.contiguous(),
                    "sefp", 5, self.bits, 64, "RNE", True,
                    5, self.bits, 64, "RNE", True, "fp24", "trunc", "float16")
        return self.record("matmul", y)

    def reduce(self, a, axis, maximum=False):
        # Preserve customer Node.mean = SUM / N. Do not silently replace it
        # by the SDK's early-scaled dedicated MEAN, a different numeric graph.
        xf = self.backend.functions
        fn = xf.reducemax_xh2a_default if maximum else xf.reducesum_xh2a_default
        return self.record("max" if maximum else "sum", fn(a.tensor.contiguous(), axis, keepdim=True))


def make_model(rt, reference, seed):
    _, graph = load_customer_modules()
    return graph.Model(rt, reference.initial_arrays(), reference.minimum.cpu().numpy(),
                 reference.span.cpu().numpy(), seed, False)


def loss_parts(model, lp, labels):
    """The exact expression/order of customer Model.loss, with named monitors."""
    _, graph = load_customer_modules()
    Node = graph.Node
    p = model.nodes
    ce = -(Node(labels)*lp).sum(1).mean(0)
    l2k = .01*(p[0]*p[0]).sum(0).sum(1)
    l2d = .001*(p[3]*p[3]).sum(0).sum(1)
    return ce+l2k+l2d, (ce, l2k, l2d)


def adam_probe_step(w, g, m, v, c1, c2):
    """Exact customer order via XH2 FP16 primitives, for scalar diagnostics."""
    backend = _xh2_backend()
    xf, vp = backend.functions, backend.arithmetic
    scalar = lambda value: w.new_tensor(value)
    add, sub = xf.add_xh2a_default, xf.sub_xh2a_default
    mul, div = xf.mul_xh2a_default, xf.div_xh2a_default
    sqrt = vp.sqrt_xh2a_default
    c1 = add(c1, mul(sub(scalar(1), c1), scalar(.1)))
    c2 = add(c2, mul(sub(scalar(1), c2), scalar(.001)))
    m = add(m, mul(sub(g, m), scalar(.1)))
    square = mul(g, g)
    v = add(v, mul(sub(square, v), scalar(.001)))
    rate = div(mul(scalar(1e-4), sqrt(c2)), c1)
    numerator = mul(rate, m)
    denominator = add(sqrt(v), scalar(1e-7))
    update = div(numerator, denominator)
    return sub(w, update), m, v, c1, c2, square, rate, numerator, denominator, update
