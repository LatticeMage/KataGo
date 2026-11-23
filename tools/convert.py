#!/usr/bin/env python3
"""Convert a KataGo model (.bin.gz or .txt.gz, model version 8..14) into a
dynamic-board-size ONNX graph (NCHW, no padding mask: every position fills the board).

Supports ordinary / gpool / nested-bottleneck blocks, ReLU and Mish.

Usage: convert.py model.{bin,txt}.gz out.onnx
"""
import gzip
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto as TP

WS = b" \n\r\t"


class Tok:
    def __init__(self, data, binary):
        self.d, self.i, self.binary = data, 0, binary

    def s(self):
        d, i = self.d, self.i
        while d[i] in WS:
            i += 1
        j = i
        while j < len(d) and d[j] not in WS:
            j += 1
        self.i = j
        return d[i:j].decode()

    def int(self):
        return int(self.s())

    def floats(self, n):
        if self.binary:
            i = self.i
            while self.d[i] != 0x40:  # '@'
                i += 1
            assert self.d[i:i + 5] == b"@BIN@", "bad binary float marker"
            i += 5
            a = np.frombuffer(self.d, dtype="<f4", count=n, offset=i).copy()
            self.i = i + 4 * n
            return a
        return np.array([float(self.s()) for _ in range(n)], np.float32)

    def done(self):
        return self.d[self.i:].strip() == b""


def read_conv(tk):
    name = tk.s()
    ky, kx, ic, oc, dy, dx = (tk.int() for _ in range(6))
    assert dy == 1 and dx == 1, "dilation unsupported"
    w = tk.floats(ky * kx * ic * oc).reshape(ky, kx, ic, oc)  # y,x,ic,oc
    w = np.ascontiguousarray(w.transpose(3, 2, 0, 1))          # oc,ic,y,x
    return dict(name=name, w=w, k=(ky, kx), ic=ic, oc=oc)


def read_bn(tk):
    name = tk.s()
    c = tk.int()
    eps = float(tk.s())
    has_scale = tk.int()
    has_bias = tk.int()
    mean = tk.floats(c)
    var = tk.floats(c)
    scale = tk.floats(c) if has_scale else np.ones(c, np.float32)
    bias = tk.floats(c) if has_bias else np.zeros(c, np.float32)
    return dict(name=name, c=c, eps=eps, mean=mean, var=var, scale=scale, bias=bias)


def read_act(tk, version):
    tk.s()
    kind = tk.s() if version >= 11 else "ACTIVATION_RELU"
    assert kind in ("ACTIVATION_RELU", "ACTIVATION_MISH", "ACTIVATION_IDENTITY"), kind
    return {"ACTIVATION_RELU": "Relu", "ACTIVATION_MISH": "Mish", "ACTIVATION_IDENTITY": None}[kind]


def read_matmul(tk):
    name = tk.s()
    ic = tk.int()
    oc = tk.int()
    return dict(name=name, w=tk.floats(ic * oc).reshape(ic, oc), ic=ic, oc=oc)


def read_bias(tk):
    name = tk.s()
    c = tk.int()
    return dict(name=name, b=tk.floats(c), c=c)


def read_stack(tk, version, n):
    blocks = []
    for _ in range(n):
        kind = tk.s()
        b = dict(kind=kind, name=tk.s())
        if kind == "ordinary_block":
            b["preBN"] = read_bn(tk); b["preAct"] = read_act(tk, version)
            b["reg"] = read_conv(tk)
            b["midBN"] = read_bn(tk); b["midAct"] = read_act(tk, version)
            b["fin"] = read_conv(tk)
        elif kind == "gpool_block":
            b["preBN"] = read_bn(tk); b["preAct"] = read_act(tk, version)
            b["reg"] = read_conv(tk)
            b["gconv"] = read_conv(tk)
            b["gBN"] = read_bn(tk); b["gAct"] = read_act(tk, version)
            b["g2b"] = read_matmul(tk)
            b["midBN"] = read_bn(tk); b["midAct"] = read_act(tk, version)
            b["fin"] = read_conv(tk)
        elif kind == "nested_bottleneck_block":
            m = tk.int()
            b["preBN"] = read_bn(tk); b["preAct"] = read_act(tk, version)
            b["preConv"] = read_conv(tk)
            b["inner"] = read_stack(tk, version, m)
            b["postBN"] = read_bn(tk); b["postAct"] = read_act(tk, version)
            b["postConv"] = read_conv(tk)
        else:
            raise SystemExit(f"unsupported block kind {kind}")
        blocks.append(b)
    return blocks


class G:
    def __init__(self):
        self.nodes, self.inits = [], []
        self.n = 0

    def uid(self, p):
        self.n += 1
        return f"{p}_{self.n}"

    def const(self, arr, p="c"):
        name = self.uid(p)
        self.inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def op(self, typ, ins, nout=1, **attrs):
        outs = [self.uid(typ.lower()) for _ in range(nout)]
        self.nodes.append(helper.make_node(typ, ins, outs, **attrs))
        return outs[0] if nout == 1 else outs

    def conv(self, x, c):
        ky, kx = c["k"]
        return self.op("Conv", [x, self.const(c["w"], "w")], kernel_shape=[ky, kx],
                       pads=[ky // 2, kx // 2, ky // 2, kx // 2])

    def bn_act(self, x, bn, act):
        y = self.op("BatchNormalization", [
            x, self.const(bn["scale"]), self.const(bn["bias"]),
            self.const(bn["mean"]), self.const(bn["var"])], epsilon=bn["eps"])
        return self.op(act, [y]) if act else y

    def matmul(self, x, m):
        return self.op("MatMul", [x, self.const(m["w"], "mm")])

    def add_bias(self, x, b):
        return self.op("Add", [x, self.const(b["b"], "bias")])

    def chan_bias(self, x, v):  # x [N,C,H,W] + v [N,C]
        v4 = self.op("Unsqueeze", [v, self.const(np.array([2, 3], np.int64))])
        return self.op("Add", [x, v4])


def main():
    src, dst = sys.argv[1], sys.argv[2]
    tk = Tok(gzip.open(src, "rb").read(), binary=src.endswith(".bin.gz"))
    name = tk.s()
    version = tk.int()
    assert 8 <= version <= 14, f"model version {version} not supported by this converter"
    n_in = tk.int()
    n_glob = tk.int()
    post = [float(tk.s()) for _ in range(7)] if version >= 13 else [20.0] * 4 + [30.0, 0.25, 150.0]

    tk.s()  # "trunk"
    n_blocks = tk.int()
    trunk_c = tk.int()
    for _ in range(4):
        tk.int()  # mid, regular, dilated, gpool channels
    init_conv = read_conv(tk)
    init_mm = read_matmul(tk)
    blocks = read_stack(tk, version, n_blocks)
    tip_bn = read_bn(tk)
    tip_act = read_act(tk, version)

    tk.s()  # policy head
    p1 = read_conv(tk); g1 = read_conv(tk)
    g1bn = read_bn(tk); g1act = read_act(tk, version)
    p_g2b = read_matmul(tk)
    p1bn = read_bn(tk); p1act = read_act(tk, version)
    p2 = read_conv(tk)
    p_pass = read_matmul(tk)

    tk.s()  # value head
    v1 = read_conv(tk)
    v1bn = read_bn(tk); v1act = read_act(tk, version)
    v2m = read_matmul(tk); v2b = read_bias(tk); v2act = read_act(tk, version)
    v3m = read_matmul(tk); v3b = read_bias(tk)
    sv3m = read_matmul(tk); sv3b = read_bias(tk)
    vown = read_conv(tk)
    assert tk.done(), "unparsed data left at end of model"

    g = G()
    ax_hw = g.const(np.array([2, 3], np.int64), "axes")
    sp, glob = "InputSpatial", "InputGlobal"
    ch0 = g.op("Slice", [sp, g.const(np.array([0], np.int64)), g.const(np.array([1], np.int64)),
                         g.const(np.array([1], np.int64))])
    n_pts = g.op("ReduceSum", [ch0, ax_hw], keepdims=0)  # [N,1]
    s14 = g.op("Sub", [g.op("Sqrt", [n_pts]), g.const(np.array(14.0, np.float32))])
    f1 = g.op("Mul", [s14, g.const(np.array(0.1, np.float32))])
    f2 = g.op("Sub", [g.op("Mul", [g.op("Mul", [s14, s14]), g.const(np.array(0.01, np.float32))]),
                      g.const(np.array(0.1, np.float32))])

    def gpool(x):
        mean = g.op("ReduceMean", [x, ax_hw], keepdims=0)
        mx = g.op("ReduceMax", [x, ax_hw], keepdims=0)
        return g.op("Concat", [mean, g.op("Mul", [mean, f1]), mx], axis=1)

    def vpool(x):
        mean = g.op("ReduceMean", [x, ax_hw], keepdims=0)
        return g.op("Concat", [mean, g.op("Mul", [mean, f1]), g.op("Mul", [mean, f2])], axis=1)

    def stack(x, blks):
        for b in blks:
            if b["kind"] == "nested_bottleneck_block":
                t = g.bn_act(x, b["preBN"], b["preAct"])
                t = g.conv(t, b["preConv"])
                t = stack(t, b["inner"])
                t = g.bn_act(t, b["postBN"], b["postAct"])
                x = g.op("Add", [x, g.conv(t, b["postConv"])])
                continue
            t = g.bn_act(x, b["preBN"], b["preAct"])
            r = g.conv(t, b["reg"])
            if b["kind"] == "gpool_block":
                gp = g.bn_act(g.conv(t, b["gconv"]), b["gBN"], b["gAct"])
                r = g.chan_bias(r, g.matmul(gpool(gp), b["g2b"]))
            r = g.bn_act(r, b["midBN"], b["midAct"])
            x = g.op("Add", [x, g.conv(r, b["fin"])])
        return x

    x = g.chan_bias(g.conv(sp, init_conv), g.op("MatMul", [glob, g.const(init_mm["w"], "mm")]))
    x = stack(x, blocks)
    x = g.bn_act(x, tip_bn, tip_act)

    p = g.conv(x, p1)
    gpv = gpool(g.bn_act(g.conv(x, g1), g1bn, g1act))
    p = g.bn_act(g.chan_bias(p, g.matmul(gpv, p_g2b)), p1bn, p1act)
    policy = g.conv(p, p2)
    policy_pass = g.matmul(gpv, p_pass)

    v = g.bn_act(g.conv(x, v1), v1bn, v1act)
    v2 = g.add_bias(g.matmul(vpool(v), v2m), v2b)
    if v2act:
        v2 = g.op(v2act, [v2])
    value = g.add_bias(g.matmul(v2, v3m), v3b)
    score = g.add_bias(g.matmul(v2, sv3m), sv3b)
    own = g.conv(v, vown)

    for src_name, dst_name in [(policy, "OutputPolicy"), (policy_pass, "OutputPolicyPass"), (value, "OutputValue"),
                               (score, "OutputScoreValue"), (own, "OutputOwnership")]:
        g.nodes.append(helper.make_node("Identity", [src_name], [dst_name]))

    def vi(nm, shape):
        return helper.make_tensor_value_info(nm, TP.FLOAT, shape)

    graph = helper.make_graph(
        g.nodes, name,
        [vi("InputSpatial", ["N", n_in, "H", "W"]), vi("InputGlobal", ["N", n_glob])],
        [vi("OutputPolicy", ["N", p2["oc"], "H", "W"]), vi("OutputPolicyPass", ["N", p_pass["oc"]]),
         vi("OutputValue", ["N", 3]), vi("OutputScoreValue", ["N", sv3m["oc"]]),
         vi("OutputOwnership", ["N", 1, "H", "W"])],
        initializer=g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    # post-processing multipliers travel with the model
    for k, val in zip(["td", "scoreMean", "scoreStdev", "lead", "varTime", "stValue", "stScore"], post):
        model.metadata_props.add(key=f"katago.{k}", value=str(val))
    model.metadata_props.add(key="katago.version", value=str(version))
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"{name}: v{version}, trunk {trunk_c}, {len(g.nodes)} nodes, policy ch {p2['oc']}, "
          f"score ch {sv3m['oc']} -> {dst}")


if __name__ == "__main__":
    main()
