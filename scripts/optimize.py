"""Optimise the classifier for CPU deployment and pick the variant to ship (validation only).

PyTorch FP32 -> ONNX FP32 -> ONNX FP16 -> ONNX INT8 (dynamic) -> ONNX INT8 (static, calibrated
on TRAIN crops). Accuracy is measured on the VAL split (ground-truth crops), latency with
2 CPU threads (the free hosting tier), size on disk. The shipped variant is the fastest one
whose val top-1 is within 0.5 points of ONNX FP32.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import CropSet  # noqa: E402
from vericar import optimization as opt  # noqa: E402
from vericar.classifier import build_model  # noqa: E402
from vericar.config import CLS_IMG_SIZE, MODELS_DIR, RESULTS_DIR, SEED  # noqa: E402
from vericar.data import load_split  # noqa: E402
from vericar.evaluation import topk_accuracy  # noqa: E402

MAX_DROP = 0.005


def batches(ds, n=None, bs=16):
    idx = range(len(ds) if n is None else n)
    xs, ys = [], []
    for i in idx:
        x, _, y, _, _ = ds[i]
        xs.append(x)
        ys.append(y)
        if len(xs) == bs:
            yield np.stack(xs), np.array(ys)
            xs, ys = [], []
    if xs:
        yield np.stack(xs), np.array(ys)


def onnx_eval(path, ds):
    s = opt.session(path, threads=0)
    name = s.get_inputs()[0].name
    L, Y = [], []
    for x, y in batches(ds):
        L.append(s.run(None, {name: x})[0])
        Y.append(y)
    return topk_accuracy(np.concatenate(L), np.concatenate(Y))


def main():
    ck = torch.load(MODELS_DIR / "classifier.pt", map_location="cpu", weights_only=False)
    model = build_model(len(ck["class_names"]), pretrained=False)
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    model.eval()

    val = load_split("val")
    ds_val = CropSet(val, mode="gt", enhance=False)
    rng = np.random.default_rng(SEED)
    calib_recs = [load_split("train")[i] for i in rng.choice(6922, 128, replace=False)]
    calib = [x for x, _ in batches(CropSet(calib_recs, mode="gt", enhance=False), bs=8)]
    one = calib[0][:1]

    paths = {"onnx_fp32": MODELS_DIR / "classifier_fp32.onnx", "onnx_fp16": MODELS_DIR / "classifier_fp16.onnx",
             "onnx_int8_dynamic": MODELS_DIR / "classifier_int8_dynamic.onnx",
             "onnx_int8_static": MODELS_DIR / "classifier_int8.onnx"}
    opt.export_onnx(model, paths["onnx_fp32"], CLS_IMG_SIZE)
    opt.to_fp16(paths["onnx_fp32"], paths["onnx_fp16"])
    opt.to_int8_dynamic(paths["onnx_fp32"], paths["onnx_int8_dynamic"])
    opt.to_int8_static(paths["onnx_fp32"], paths["onnx_int8_static"], calib)

    rows = {}
    # PyTorch FP32 baseline (2 threads)
    torch.set_num_threads(2)
    xt = torch.from_numpy(one)
    with torch.inference_mode():
        for _ in range(3):
            model(xt)
        ts = []
        for _ in range(15):
            t = time.perf_counter()
            model(xt)
            ts.append((time.perf_counter() - t) * 1000)
    torch.save(model.state_dict(), MODELS_DIR / "_tmp_fp32.pt")
    rows["pytorch_fp32"] = {"size_mb": opt.size_mb(MODELS_DIR / "_tmp_fp32.pt"), "median_ms": float(np.median(ts))}
    (MODELS_DIR / "_tmp_fp32.pt").unlink()
    torch.set_num_threads(8)

    for k, p in paths.items():
        r = {"size_mb": opt.size_mb(p), **opt.benchmark(opt.session(p, threads=2), one)}
        r.update(onnx_eval(p, ds_val))
        rows[k] = r
        print(k, json.dumps({a: round(b, 4) for a, b in r.items()}), flush=True)

    ref = rows["onnx_fp32"]["top1"]
    ok = [k for k in paths if rows[k]["top1"] >= ref - MAX_DROP]
    chosen = min(ok, key=lambda k: rows[k]["median_ms"])
    meta_p = MODELS_DIR / "hybrid.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        meta["onnx_file"] = paths[chosen].name
        meta_p.write_text(json.dumps(meta))
    (RESULTS_DIR / "optimization_val.json").write_text(json.dumps(
        {"rows": rows, "chosen": chosen, "rule": f"fastest variant with val top-1 >= ONNX FP32 - {MAX_DROP:.3f}",
         "latency_threads": 2, "accuracy_split": "val (ground-truth crops)"}, indent=1))
    print("CHOSEN", chosen, flush=True)


if __name__ == "__main__":
    main()
