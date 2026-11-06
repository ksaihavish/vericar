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

import os  # noqa: E402

from vericar import optimization as opt  # noqa: E402
from vericar.classifier import CarCrops, build_model  # noqa: E402
from vericar.config import CLS_IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR, RESULTS_DIR, SEED  # noqa: E402
from vericar.data import load_split  # noqa: E402
from vericar.evaluation import topk_accuracy  # noqa: E402

MAX_DROP = 0.005


MEAN = np.asarray(IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1)
STD = np.asarray(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)


def load_crops(recs, workers=12):
    """All crops decoded ONCE, in parallel, as uint8 NCHW (same crop/resize as the app)."""
    from torch.utils.data import DataLoader
    xs, ys = [], []
    for x, y in DataLoader(CarCrops(recs, train=False), batch_size=64, num_workers=workers):
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def norm(x_u8):
    return (x_u8.astype(np.float32) / 255.0 - MEAN) / STD


def onnx_eval(path, X, Y, threads, bs=32, budget_s=180.0, subset=400):
    """Accuracy on val crops. If the full pass would exceed budget_s (e.g. dynamic INT8, whose
    integer kernels are slow on this CPU), use a fixed seeded subset and record its size."""
    s = opt.session(path, threads=threads)
    name = s.get_inputs()[0].name
    t = time.perf_counter()
    first = s.run(None, {name: norm(X[:bs])})[0]
    per_batch = time.perf_counter() - t
    idx = np.arange(len(X))
    if per_batch * len(X) / bs > budget_s:
        idx = np.sort(np.random.default_rng(SEED).choice(len(X), subset, replace=False))
        first = None
    Xs, Ys = X[idx], Y[idx]
    L = [first] if first is not None else []
    for i in range(bs if first is not None else 0, len(Xs), bs):
        L.append(s.run(None, {name: norm(Xs[i:i + bs])})[0])
    return {**topk_accuracy(np.concatenate(L).astype(np.float32), Ys), "acc_n": int(len(Ys))}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-export", action="store_true", help="reuse existing ONNX files")
    ap.add_argument("--acc-threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    ck = torch.load(MODELS_DIR / "classifier.pt", map_location="cpu", weights_only=False)
    model = build_model(len(ck["class_names"]), pretrained=False)
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    model.eval()

    Xv, Yv = load_crops(load_split("val"))
    rng = np.random.default_rng(SEED)
    calib_recs = [load_split("train")[i] for i in rng.choice(6922, 128, replace=False)]
    Xc, _ = load_crops(calib_recs)
    calib = [norm(Xc[i:i + 8]) for i in range(0, len(Xc), 8)]
    one = calib[0][:1]

    paths = {"onnx_fp32": MODELS_DIR / "classifier_fp32.onnx", "onnx_fp16": MODELS_DIR / "classifier_fp16.onnx",
             "onnx_int8_dynamic": MODELS_DIR / "classifier_int8_dynamic.onnx",
             "onnx_int8_static": MODELS_DIR / "classifier_int8.onnx"}
    if not (args.skip_export and all(p.exists() for p in paths.values())):
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

    for k, p in paths.items():   # latency first (2 threads), before any heavy accuracy pass
        rows[k] = {"size_mb": opt.size_mb(p), **opt.benchmark(opt.session(p, threads=2), one, runs=15)}
    for k, p in paths.items():
        rows[k].update(onnx_eval(p, Xv, Yv, args.acc_threads))
        print(k, json.dumps({a: round(b, 4) for a, b in rows[k].items()}), flush=True)

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
