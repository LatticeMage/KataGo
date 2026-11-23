#!/usr/bin/env python3
"""Static int8 quantization (QOperator: QLinearConv / QLinearMatMul) of a converted KataGo ONNX model,
calibrated on real positions (models/calib_<n>.bin). Then reports fp32-vs-int8 agreement.

Usage: quantize.py in.onnx out.onnx [--per-tensor] [--method minmax|percentile]
"""
import sys, numpy as np, onnx, onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod

src, dst = sys.argv[1], sys.argv[2]
method = CalibrationMethod.Percentile if "percentile" in sys.argv else CalibrationMethod.MinMax


def load(n):
    raw = np.fromfile(f"models/calib_{n}.bin", dtype=np.float32)
    per = 22 * n * n + 19
    N = raw.size // per
    sp = raw[: N * 22 * n * n].reshape(N, 22, n, n)
    gl = raw[N * 22 * n * n:].reshape(N, 19)
    return sp, gl


data = {n: load(n) for n in (9, 13, 19)}


class Reader(CalibrationDataReader):
    def __init__(self, which):
        self.items = []
        for n, (sp, gl) in data.items():
            idx = range(0, len(sp), 2 if n == 19 else 3)  # subsample, calibration is slow
            self.items += [{"InputSpatial": sp[i:i + 1], "InputGlobal": gl[i:i + 1]} for i in idx]
        np.random.RandomState(0).shuffle(self.items)
        self.it = iter(self.items)

    def get_next(self):
        return next(self.it, None)


# The pooled heads / BN statistics stay fp32; only the heavy Conv and MatMul run in int8.
quantize_static(
    src, dst, Reader(None),
    quant_format=QuantFormat.QOperator,
    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
    per_channel="--per-tensor" not in sys.argv,
    op_types_to_quantize=["Conv"],
    calibrate_method=method,
    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
)
print("saved", dst)

# agreement check
a, b = ort.InferenceSession(src), ort.InferenceSession(dst)
klp, dv, ds, top1, cnt = [], [], [], 0, 0
for n, (sp, gl) in data.items():
    for i in range(0, len(sp), 4):
        f = {"InputSpatial": sp[i:i + 1], "InputGlobal": gl[i:i + 1]}
        oa, ob = a.run(None, f), b.run(None, f)
        pa = oa[0][0, 0].ravel(); pb = ob[0][0, 0].ravel()
        pa = np.exp(pa - pa.max()); pa /= pa.sum(); pb = np.exp(pb - pb.max()); pb /= pb.sum()
        klp.append(float(np.sum(pa * (np.log(pa + 1e-9) - np.log(pb + 1e-9)))))
        top1 += int(pa.argmax() == pb.argmax()); cnt += 1
        wa = 1 / (1 + np.exp(oa[2][0][1] - oa[2][0][0])); wb = 1 / (1 + np.exp(ob[2][0][1] - ob[2][0][0]))
        dv.append(abs(wa - wb)); ds.append(abs(oa[3][0][0] - ob[3][0][0]) * 20)
print(f"samples {cnt}: policy KL mean {np.mean(klp):.4f} max {np.max(klp):.3f}; top1 agree {top1/cnt:.1%}; "
      f"winrate |diff| mean {np.mean(dv):.3f} max {np.max(dv):.3f}; score |diff| mean {np.mean(ds):.2f}")
