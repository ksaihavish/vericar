"""Fit every hybrid / verification parameter on the VALIDATION split only.

Runs the real pipeline on val images (YOLO box -> crop -> quality gate ->
adaptive enhancement -> CNN, plus classical features), then fits:
  1. temperature T (min val NLL)
  2. classical fusion weight w (min val NLL, grid 0..0.5)
  3. unknown-car threshold: 95th percentile of val kNN distances to train embeddings
  4. strict verification thresholds: the lowest MATCH threshold whose hard-clone
     MATCH rate <= 5% on val; MISMATCH threshold keeping legit MISMATCH <= 1%
  5. identify-mode threshold: lowest confidence whose selective val accuracy >= 95%
Writes models/hybrid.json, models/ood_bank.npy and results/val_fitting.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from common import CropSet, detect_all, load_torch_classifier, primary_box, run_set  # noqa: E402
from vericar.config import FEAT_DIR, MODELS_DIR, RESULTS_DIR, SEED  # noqa: E402
from vericar.data import class_names, hierarchy, load_split  # noqa: E402
from vericar.evaluation import claim_sets, cleared_rate, clone_study, ece, topk_accuracy  # noqa: E402
from vericar.hybrid import KnnOOD, Thresholds, fit_temperature, fuse, softmax  # noqa: E402

OOD_K = 10
HARD_CLONE_MAX_MATCH = 0.02   # strict: at most 2% of hard clones may be cleared (validation)
LEGIT_MAX_MISMATCH = 0.01
IDENTIFY_TARGET_ACC = 0.95
MODEL_VERIFIED = 0.80


def nll(p, y):
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def main():
    names, hier = class_names(), hierarchy()
    val = load_split("val")
    y = np.array([r["label"] for r in val])

    dets = detect_all(MODELS_DIR / "detector.pt", val)
    boxes = [primary_box(d) for d in dets]
    model = load_torch_classifier()
    svm = joblib.load(MODELS_DIR / "classical_svm.joblib")
    out = run_set(CropSet(val, boxes, "det", enhance=True, classical_feats=True), model, svm)
    logits, feats, p_cls, flagged = out["logits"], out["feats"], out["p_classical"], out["flagged"]
    print(f"val: {out['enhanced'].mean():.1%} enhanced, {flagged.mean():.1%} flagged by quality gate", flush=True)

    # 1-2. calibration + fusion weight
    T = fit_temperature(logits, y)
    grid = []
    for w in np.round(np.arange(0, 0.51, 0.05), 2):
        p = fuse(logits, T, p_cls, w)
        grid.append({"w": float(w), "nll": nll(p, y), **topk_accuracy(p, y)})
    best_w = min(grid, key=lambda g: g["nll"])["w"]
    p = fuse(logits, T, p_cls, best_w)
    print(f"T={T:.2f}  w={best_w}  val top1 {topk_accuracy(p, y)}", flush=True)

    # 3. unknown-car detector
    train = np.load(FEAT_DIR / "cnn_train.npz")
    bank = train["feats"].astype(np.float16)
    ood = KnnOOD(bank, k=OOD_K)
    s_val = ood.score(feats)
    ood.threshold = float(np.percentile(s_val, 95))
    unknown = s_val > ood.threshold

    # hard-clone partner: nearest class prototype (train embeddings); easy: random different make
    f = train["feats"].astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    proto = np.stack([f[train["labels"] == c].mean(0) for c in range(len(names))])
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    sim = proto @ proto.T
    np.fill_diagonal(sim, -1)
    hard = sim.argmax(1)
    rng = np.random.default_rng(SEED)
    easy = np.array([rng.choice(np.flatnonzero(hier.make_of != hier.make_of[c])) for c in range(len(names))])

    # 4. strict verification thresholds
    claims = claim_sets(y, hard, easy)
    sweep = []
    for t in np.round(np.arange(0.20, 0.991, 0.01), 2):
        sweep.append({"t": float(t), **{k: cleared_rate(p, c, unknown, flagged, t) for k, c in claims.items()}})
    ok = [s for s in sweep if s["hard_clone"] <= HARD_CLONE_MAX_MATCH]
    t_match = ok[0]["t"] if ok else 0.99
    p_model_legit = np.array([p[i, hier.model_of == hier.model_of[c]].sum() for i, c in enumerate(y)])
    t_mis = max([t for t in np.round(np.arange(0.01, 0.31, 0.01), 2)
                 if (p_model_legit < t).mean() <= LEGIT_MAX_MISMATCH] or [0.01])

    # 5. identify threshold
    conf, correct = p.max(1), p.argmax(1) == y
    t_id = next((float(t) for t in np.round(np.arange(0.2, 0.96, 0.01), 2)
                 if correct[conf >= t].mean() >= IDENTIFY_TARGET_ACC), 0.9)

    # PARTIAL ("make and model verified, year not") needs a confident model family: fixed at 0.80
    thr = Thresholds(match=t_match, model=MODEL_VERIFIED, mismatch=float(t_mis), identify=t_id)
    study = clone_study(p, y, unknown, flagged, hier, thr, hard, easy)
    print("thresholds", thr.as_dict(), flush=True)
    print("val clone study", json.dumps({k: study[k] for k in ("false_alarm_rate", "catch_rate_hard", "catch_rate_easy")}), flush=True)

    meta = {"class_names": names, "temperature": T, "fusion_weight": best_w, "thresholds": thr.as_dict(),
            "ood_k": OOD_K, "ood_threshold": ood.threshold, "hard_partner": hard.tolist(),
            "easy_partner": easy.tolist(), "onnx_file": "classifier.onnx"}
    old = MODELS_DIR / "hybrid.json"
    if old.exists():  # keep the optimisation choice if optimize.py already ran
        meta["onnx_file"] = json.loads(old.read_text()).get("onnx_file", "classifier.onnx")
    (MODELS_DIR / "hybrid.json").write_text(json.dumps(meta))
    np.save(MODELS_DIR / "ood_bank.npy", bank)
    e_raw, _ = ece(softmax(logits), y)
    e_cal, rel = ece(p, y)
    (RESULTS_DIR / "val_fitting.json").write_text(json.dumps({
        "temperature": T, "fusion_grid": grid, "fusion_weight": best_w,
        "ece_raw": e_raw, "ece_calibrated": e_cal, "reliability_val": rel,
        "ood": {"k": OOD_K, "threshold": ood.threshold, "val_flag_rate": float(unknown.mean())},
        "verification_sweep": sweep, "thresholds": thr.as_dict(), "val_clone_study": study,
        "quality_gate_flag_rate": float(flagged.mean()), "enhanced_rate": float(out["enhanced"].mean()),
        "val_top1_deep_raw": topk_accuracy(softmax(logits), y), "val_top1_hybrid": topk_accuracy(p, y),
        "val_top1_classical_on_yolo_crop": topk_accuracy(p_cls, y),
    }, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
