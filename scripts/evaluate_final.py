"""THE single evaluation run on the official Stanford Cars test set (8,041 images).

Every model choice, threshold and weight was frozen on train/val before this runs.
The script refuses to run twice (results/test_run.json is the record); ``--force``
re-runs it and the re-run is recorded in that file.

Produces results/final_test.json with: detector comparison (IoU, mAP), classical vs
deep vs hybrid (Top-1/Top-5, per level), ablations, calibration and coverage,
clone-verification study, robustness, viewing-angle proxy, similar-class analysis,
unknown-car detection (probe set) and failure-case candidates.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from common import CropSet, PERTURBATIONS, detect_all, load_torch_classifier, primary_box, run_set  # noqa: E402
from vericar import classical as cl  # noqa: E402
from vericar.config import DATA_DIR, MODELS_DIR, REPO_ROOT, RESULTS_DIR, RUNS_DIR, SEED  # noqa: E402
from vericar.data import hierarchy, load_split  # noqa: E402
from vericar.evaluation import (confused_pairs, coverage_curve, detection_report, ece, iou, level_accuracy,  # noqa: E402
                                auroc, claim_sets, cleared_rate, clone_study, topk_accuracy,
                                year_given_model_accuracy)
from vericar.hybrid import KnnOOD, Thresholds, fuse, softmax  # noqa: E402
from vericar.preprocessing import load_rgb  # noqa: E402

MARKER = RESULTS_DIR / "test_run.json"
ROBUST_N = 1000


def method_metrics(p, y, hier) -> dict:
    e, _ = ece(p, y)
    return {**topk_accuracy(p, y), "make_acc": level_accuracy(p, y, hier.make_of),
            "model_acc": level_accuracy(p, y, hier.model_of),
            "year_given_model_acc": year_given_model_accuracy(p, y, hier.model_of), "ece": e}


def _classical_box(path):
    return cl.classical_roi(load_rgb(path))["box"]


def flat(dets):
    return [(i, s, b) for i, d in enumerate(dets) for b, s in d]


def main():
    force = "--force" in sys.argv
    if MARKER.exists() and not force:
        print(f"Test set already evaluated: {MARKER.read_text()}\nUse --force to re-run (it will be recorded).")
        return
    started = dt.datetime.now().isoformat(timespec="seconds")
    hier = hierarchy()
    meta = json.loads((MODELS_DIR / "hybrid.json").read_text())
    T, w, thr = meta["temperature"], meta["fusion_weight"], Thresholds(**meta["thresholds"])
    hard, easy = np.array(meta["hard_partner"]), np.array(meta["easy_partner"])
    ood = KnnOOD(np.load(MODELS_DIR / "ood_bank.npy"), k=meta["ood_k"], threshold=meta["ood_threshold"])
    test = load_split("test")
    y = np.array([r["label"] for r in test])
    R: dict = {"n_test": len(test), "frozen_params": {k: meta[k] for k in ("temperature", "fusion_weight",
                                                                          "thresholds", "ood_threshold")}}

    # ---------------------------------------------------------------- A. detection
    print("A. detection", flush=True)
    gts = {i: r["box"] for i, r in enumerate(test)}
    det_ft = detect_all(MODELS_DIR / "detector.pt", test)
    det_coco = detect_all("yolo11n.pt", test, coco=True)
    cboxes = Parallel(n_jobs=6, batch_size=64)(delayed(_classical_box)(r["path"]) for r in test)
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)   # free the idle workers' RAM (~0.6 GB each)
    R["detection"] = {
        "YOLO11n fine-tuned (ours)": detection_report(flat(det_ft), gts),
        "YOLO11n COCO-pretrained": detection_report(flat(det_coco), gts),
        "Classical edges+contours": detection_report([(i, 1.0, b) for i, b in enumerate(cboxes)], gts),
    }
    print(json.dumps(R["detection"], indent=1), flush=True)
    boxes = [primary_box(d) for d in det_ft]

    # ---------------------------------------------------------------- B. three pipelines + ablations
    print("B. pipelines", flush=True)
    model = load_torch_classifier()
    svm = joblib.load(MODELS_DIR / "classical_svm.joblib")
    need_cls = w > 0
    # CPU-only classical pipeline runs in a background thread while the GPU pipelines run
    pool = ThreadPoolExecutor(max_workers=1)
    fut_clas = pool.submit(run_set, CropSet(test, None, "classical", enhance=False, classical_feats=True), None, svm,
                           64, 3)
    hyb = run_set(CropSet(test, boxes, "det", enhance=True, classical_feats=need_cls), model, svm)
    raw = run_set(CropSet(test, boxes, "det", enhance=False, classical_feats=need_cls), model, svm)
    full = run_set(CropSet(test, None, "full", enhance=True, classical_feats=need_cls), model, svm)
    gtc = run_set(CropSet(test, None, "gt", enhance=False), model)
    clas = fut_clas.result()

    p_cls = clas["p_classical"]
    p_deep = softmax(raw["logits"])
    p_hyb = fuse(hyb["logits"], T, hyb.get("p_classical"), w)
    unk = ood.score(hyb["feats"]) > ood.threshold
    flg = hyb["flagged"]
    R["methods"] = {
        "Classical (contour ROI + HOG/LBP/GLCM/colour/geometry + SVM)": method_metrics(p_cls, y, hier),
        "Deep (YOLO crop + ConvNeXt-T)": method_metrics(p_deep, y, hier),
        "Hybrid (ours)": method_metrics(p_hyb, y, hier),
        "Reference: ConvNeXt-T on ground-truth crop": method_metrics(softmax(gtc["logits"]), y, hier),
    }
    R["hybrid_rates"] = {"enhanced": float(hyb["enhanced"].mean()), "quality_flagged": float(flg.mean()),
                         "unknown_flagged": float(unk.mean()),
                         "no_detection": float(np.mean([b is None for b in boxes]))}

    variants = {
        "Full hybrid": (p_hyb, unk, flg),
        "- adaptive preprocessing": (fuse(raw["logits"], T, raw.get("p_classical"), w),
                                     ood.score(raw["feats"]) > ood.threshold, raw["flagged"]),
        "- YOLO ROI crop (full frame)": (fuse(full["logits"], T, full.get("p_classical"), w),
                                         ood.score(full["feats"]) > ood.threshold, full["flagged"]),
        "- classical feature fusion (w=0)": (fuse(hyb["logits"], T), unk, flg),
        "- temperature calibration (T=1)": (fuse(hyb["logits"], 1.0, hyb.get("p_classical"), w), unk, flg),
        "- quality gate & unknown check": (p_hyb, np.zeros_like(unk), np.zeros_like(flg)),
    }
    R["ablation"] = {}
    for name, (p, u, f) in variants.items():
        cs = clone_study(p, y, u, f, hier, thr, hard, easy)
        R["ablation"][name] = {**method_metrics(p, y, hier), "false_alarm_rate": cs["false_alarm_rate"],
                               "catch_rate_hard": cs["catch_rate_hard"], "catch_rate_easy": cs["catch_rate_easy"]}
    print(json.dumps(R["methods"], indent=1), flush=True)

    # ---------------------------------------------------------------- C. confidence + verification
    conf, correct = p_hyb.max(1), p_hyb.argmax(1) == y
    R["confidence"] = {
        "coverage_hybrid": coverage_curve(conf, correct),
        "coverage_deep": coverage_curve(p_deep.max(1), p_deep.argmax(1) == y),
        "reliability_hybrid": ece(p_hyb, y)[1], "reliability_deep": ece(p_deep, y)[1],
        "identify_threshold": thr.identify,
        "at_identify_threshold": {"coverage": float((conf >= thr.identify).mean()),
                                  "accuracy": float(correct[conf >= thr.identify].mean())},
        "top3_acc_when_low_confidence": float((np.argsort(-p_hyb, 1)[:, :3] == y[:, None]).any(1)[conf < thr.identify].mean()),
    }
    R["verification"] = clone_study(p_hyb, y, unk, flg, hier, thr, hard, easy)
    R["verification"]["deep_only_same_thresholds"] = {
        k: v for k, v in clone_study(p_deep, y, np.zeros_like(unk), np.zeros_like(flg), hier, thr, hard, easy).items()
        if k.endswith("rate") or k.startswith("catch")}
    claims = claim_sets(y, hard, easy)
    R["verification"]["sweep"] = [{"t": float(t), **{k: cleared_rate(p_hyb, c, unk, flg, t) for k, c in claims.items()}}
                                  for t in np.round(np.arange(0.05, 0.996, 0.01), 3)]

    # ---------------------------------------------------------------- D. similar classes / angle proxy
    R["similar_classes"] = {"top_confusions": confused_pairs(p_hyb, y, hier.class_names, 15)}
    pred = p_hyb.argmax(1)
    wrong = pred != y
    R["similar_classes"]["error_breakdown"] = {
        "same_model_other_year": float((wrong & (hier.model_of[pred] == hier.model_of[y])).sum() / max(1, wrong.sum())),
        "same_make_other_model": float((wrong & (hier.make_of[pred] == hier.make_of[y]) &
                                        (hier.model_of[pred] != hier.model_of[y])).sum() / max(1, wrong.sum())),
        "other_make": float((wrong & (hier.make_of[pred] != hier.make_of[y])).sum() / max(1, wrong.sum())),
        "n_errors": int(wrong.sum())}
    ar = np.array([(r["box"][2] - r["box"][0]) / max(1, r["box"][3] - r["box"][1]) for r in test])
    bins = {"front/rear-ish (w/h < 1.4)": ar < 1.4, "three-quarter (1.4-1.9)": (ar >= 1.4) & (ar < 1.9),
            "side-ish (>= 1.9)": ar >= 1.9}
    R["viewpoint_proxy"] = {k: {"n": int(m.sum()), "hybrid_top1": float(correct[m].mean()),
                                "deep_top1": float((p_deep.argmax(1) == y)[m].mean()),
                                "classical_top1": float((p_cls.argmax(1) == y)[m].mean())} for k, m in bins.items()}

    # ---------------------------------------------------------------- E. unknown cars (probe set)
    probe_dir = next((d for d in (REPO_ROOT / "probe", DATA_DIR / "probe_modern") if d.exists()), None)
    s_test = ood.score(hyb["feats"])
    R["unknown_detection"] = {"test_flag_rate": float(unk.mean())}
    if probe_dir is not None:
        files = sorted([p for p in probe_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
        probe = [{"path": str(p), "box": [0, 0, 1, 1], "label": 0} for p in files]
        pdet = detect_all(MODELS_DIR / "detector.pt", probe)
        po = run_set(CropSet(probe, [primary_box(d) for d in pdet], "det", enhance=True), model, None, bs=16, workers=2)
        s_probe = ood.score(po["feats"])
        msp_t, msp_p = softmax(hyb["logits"], T).max(1), softmax(po["logits"], T).max(1)
        R["unknown_detection"].update({
            "probe_n": len(files), "probe_flag_rate": float((s_probe > ood.threshold).mean()),
            "auroc_knn": auroc(s_test, s_probe), "auroc_msp": auroc(-msp_t, -msp_p),
            "probe_files": [p.name for p in files], "probe_scores": s_probe.tolist(),
            "probe_pred": [hier.class_names[i] for i in softmax(po["logits"], T).argmax(1)]})

    # ---------------------------------------------------------------- F. robustness (fixed random subset)
    print("F. robustness", flush=True)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(test), ROBUST_N, replace=False))
    sub = [test[i] for i in idx]
    ys = y[idx]
    R["robustness"] = {}
    for name in PERTURBATIONS:
        fut_c = pool.submit(run_set, CropSet(sub, None, "classical", classical_feats=True, perturb=name), None, svm,
                            64, 3)
        d = detect_all(MODELS_DIR / "detector.pt", sub, perturb=name)
        b = [primary_box(x) for x in d]
        h = run_set(CropSet(sub, b, "det", enhance=True, classical_feats=need_cls, perturb=name), model, svm)
        r_ = run_set(CropSet(sub, b, "det", enhance=False, perturb=name), model)
        c_ = fut_c.result()
        ph = fuse(h["logits"], T, h.get("p_classical"), w)
        R["robustness"][name] = {
            "classical_top1": float((c_["p_classical"].argmax(1) == ys).mean()),
            "deep_top1": float((r_["logits"].argmax(1) == ys).mean()),
            "hybrid_top1": float((ph.argmax(1) == ys).mean()),
            "hybrid_retake_rate": float(h["flagged"].mean()), "enhanced_rate": float(h["enhanced"].mean()),
            "det_mean_iou": float(np.mean([iou(bb, r["box"]) if bb is not None else 0.0 for bb, r in zip(b, sub)])),
        }
        print(name, json.dumps(R["robustness"][name]), flush=True)

    # ---------------------------------------------------------------- G. failure-case candidates
    det_iou = np.array([iou(b, r["box"]) if b is not None else 0.0 for b, r in zip(boxes, test)])
    bright = np.array([np.asarray(load_rgb(test[i]["path"])).mean() if wrong[i] else 255 for i in range(len(test))])
    same_model = wrong & (hier.model_of[pred] == hier.model_of[y])
    cands = {
        "year_confusion": int(np.argmax(np.where(same_model, conf, -1))),
        "confident_wrong_make": int(np.argmax(np.where(wrong & (hier.make_of[pred] != hier.make_of[y]), conf, -1))),
        "bad_localisation": int(np.argmin(np.where(wrong, det_iou, 9))),
        "dark_image": int(np.argmin(bright)),
        "truncated_or_occluded": int(np.argmax(np.where(wrong, [(r["box"][0] <= 2) + (r["box"][2] >= r["width"] - 2)
                                                             for r in test], -1))),
    }
    R["failure_candidates"] = {k: {"index": i, "id": test[i]["id"], "true": hier.class_names[y[i]],
                                   "pred": hier.class_names[pred[i]], "conf": float(conf[i]), "det_iou": float(det_iou[i]),
                                   "top3": [hier.class_names[j] for j in np.argsort(-p_hyb[i])[:3]]}
                               for k, i in cands.items()}

    # ---------------------------------------------------------------- H. deployed pipeline latency (CPU)
    import os
    import shutil
    os.environ["VERICAR_CPU"] = "1"
    from vericar.pipeline import VeriCar
    eng = VeriCar(MODELS_DIR)
    eng.analyze(test[0]["path"])  # warm-up
    tims = [eng.analyze(test[i]["path"])["timings_ms"] for i in idx[:20]]
    R["latency"] = {k: float(np.median([t[k] for t in tims])) for k in tims[0]}
    shutil.copy2(RUNS_DIR / "classifier" / "history.json", RESULTS_DIR / "training_history.json")

    # ---------------------------------------------------------------- save
    np.savez_compressed(DATA_DIR / "test_outputs.npz", p_hyb=p_hyb.astype(np.float32), p_deep=p_deep.astype(np.float32),
                        p_cls=p_cls.astype(np.float32), y=y, unk=unk, flg=flg, det_iou=det_iou)
    (RESULTS_DIR / "final_test.json").write_text(json.dumps(R, indent=1))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    runs = json.loads(MARKER.read_text())["runs"] if MARKER.exists() else []
    runs.append({"started": started, "finished": dt.datetime.now().isoformat(timespec="seconds"),
                 "git_commit": commit, "forced": force})
    MARKER.write_text(json.dumps({"runs": runs}, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
