"""Stage 7 - model optimisation: ONNX export, FP16 and INT8 quantisation, CPU benchmarking.

Latency is measured with ONNX Runtime limited to 2 CPU threads to mimic the
free hosting tier the live demo runs on.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def export_onnx(model, path: Path, img_size: int) -> Path:
    import torch
    model = model.eval().cpu()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=["logits", "features"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "features": {0: "batch"}},
                      opset_version=17, dynamo=False)
    return path


def to_fp16(src: Path, dst: Path) -> Path:
    import onnx
    from onnxconverter_common import float16
    # the head Gemm reads the float32 'features' graph output -> keep that node in FP32
    m = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True, node_block_list=["/fc/Gemm"])
    onnx.save(m, str(dst))
    return dst


def to_int8_dynamic(src: Path, dst: Path) -> Path:
    """Weight-only INT8 (dynamic activations) - no calibration data needed."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def to_int8_static(src: Path, dst: Path, calib_batches: list[np.ndarray]) -> Path:
    """Full INT8 (weights + activations), calibrated on TRAIN images only."""
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"image": b} for b in calib_batches])

        def get_next(self):
            return next(self.it, None)

    pre = dst.with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre))
    quantize_static(str(pre), str(dst), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    pre.unlink(missing_ok=True)
    return dst


def session(path: Path, threads: int = 2):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def benchmark(sess, x: np.ndarray, runs: int = 30, warmup: int = 5) -> dict:
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: x})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t) * 1000)
    return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90))}


def size_mb(path: Path) -> float:
    return Path(path).stat().st_size / 1e6
