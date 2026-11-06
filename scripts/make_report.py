"""Write docs/eval_P5_<TEAM>.md from results/*.json, metrics in the order of the PS5 brief:
classification (Top-1/Top-5) -> localisation (IoU) -> detection (mAP) -> efficiency (size, time)
-> robustness -> confidence/threshold analysis, then ablation and failure cases.

    python scripts/make_report.py --team T01 --usns 1XX22CS001,1XX22CS002,1XX22CS003,1XX22CS004
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vericar.config import RESULTS_DIR  # noqa: E402


def j(name):
    p = RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


def pct(x):
    return f"{x * 100:.2f}%" if isinstance(x, (int, float)) else "n/a"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="TXX")
    ap.add_argument("--usns", default="USN1,USN2,USN3,USN4")
    a = ap.parse_args()
    F, O, V, C, H = j("final_test.json"), j("optimization_val.json"), j("val_fitting.json"), j("classical_val.json"), \
        j("training_history.json")
    fails = j("failures.json") or []
    m, d, r, conf, ver = F["methods"], F["detection"], F["robustness"], F["confidence"], F["verification"]
    L = [a.usns, "", f"# Evaluation results - P5 (Team {a.team}): VeriCar", "",
         f"Official Stanford Cars test set, **{F['n_test']:,} images, evaluated once** after all parameters were frozen "
         "on an 85/15 stratified train/validation split (seed 42) of the official training set. "
         "Robustness uses a fixed random 1,000-image test subset.", ""]

    L += ["## 1. Classification performance (Top-1 / Top-5)", "",
          table(["Method", "Top-1", "Top-5", "Make acc.", "Model acc.", "Year acc. given correct model"],
                [[k, pct(v["top1"]), pct(v["top5"]), pct(v["make_acc"]), pct(v["model_acc"]),
                  pct(v["year_given_model_acc"])] for k, v in m.items()]), ""]
    if C:
        L += ["Classical model selection and feature-family ablation (validation split):", "",
              table(["Variant", "Top-1", "Top-5"], [[k, pct(v["top1"]), pct(v["top5"])] for k, v in
                                                   {**C["model_selection_val"], **C["feature_family_ablation_val"]}.items()]), ""]

    L += ["## 2. Object localisation (IoU) and 3. Object detection (mAP)", "",
          table(["Detector", "Mean IoU", "IoU >= 0.5", "IoU >= 0.75", "mAP@0.5", "mAP@0.5:0.95", "Miss rate"],
                [[k, f"{v['mean_iou']:.3f}", pct(v["iou_ge_0.5"]), pct(v["iou_ge_0.75"]), f"{v['mAP50']:.3f}",
                  f"{v['mAP50_95']:.3f}", pct(v["miss_rate"])] for k, v in d.items()]), ""]

    if O:
        L += ["## 4. Model efficiency (size and inference time)", "",
              f"Classifier variants; latency = median of 30 single-image runs on CPU limited to 2 threads; accuracy on "
              f"validation ground-truth crops. Shipped variant: **{O['chosen']}** ({O['rule']}).", "",
              table(["Variant", "Size (MB)", "Latency (ms)", "Val Top-1"],
                    [[k, f"{v['size_mb']:.1f}", f"{v['median_ms']:.0f}", pct(v.get("top1"))] for k, v in O["rows"].items()]), ""]
    lat = j("latency_deployed.json")
    if lat:
        L += [f"End-to-end deployed pipeline (`VeriCar.analyze`, ONNX FP32 classifier, YOLO on CPU), median over "
              f"{lat['n']} test images, ms:", "",
              table(["Stage", "ms"], [[k, f"{v:.0f}"] for k, v in lat["median_ms"].items()]), ""]
    if O.get("notes"):
        L += ["Notes:", ""] + [f"- {n}" for n in O["notes"]] + [""]

    L += ["## 5. Robustness (performance under varying conditions)", "",
          table(["Condition", "Classical", "Deep", "Hybrid", "Hybrid RETAKE rate", "Box IoU"],
                [[k, pct(v["classical_top1"]), pct(v["deep_top1"]), pct(v["hybrid_top1"]), pct(v["hybrid_retake_rate"]),
                  f"{v['det_mean_iou']:.3f}"] for k, v in r.items()]), "",
          "Viewing-angle proxy (box width/height; Stanford Cars has no pose labels):", "",
          table(["View", "n", "Classical", "Deep", "Hybrid"],
                [[k, v["n"], pct(v["classical_top1"]), pct(v["deep_top1"]), pct(v["hybrid_top1"])]
                 for k, v in F["viewpoint_proxy"].items()]), ""]

    L += ["## 6. Confidence and threshold analysis", "",
          f"Temperature scaling T = {F['frozen_params']['temperature']:.2f} (validation). ECE: deep "
          f"{m['Deep (YOLO crop + ConvNeXt-T)']['ece']:.3f}, hybrid {m['Hybrid (ours)']['ece']:.3f}.", "",
          table(["Confidence threshold", "Coverage", "Accuracy of answered"],
                [[f"{x['threshold']:.2f}", pct(x["coverage"]), pct(x["accuracy"])] for x in conf["coverage_hybrid"][::2]]), "",
          f"At the default identify threshold {conf['identify_threshold']:.2f}: coverage "
          f"{pct(conf['at_identify_threshold']['coverage'])}, accuracy {pct(conf['at_identify_threshold']['accuracy'])}; "
          f"below it the app shows the top-3, which contain the true class {pct(conf['top3_acc_when_low_confidence'])} of the time.", "",
          "### Claim verification (our use case)", "",
          table(["Claim type", "MATCH (cleared)", "PARTIAL", "INSPECT", "MISMATCH", "UNKNOWN", "RETAKE"],
                [[k, *(pct(ver[k][s]) for s in ("MATCH", "PARTIAL", "INSPECT", "MISMATCH", "UNKNOWN", "RETAKE"))]
                 for k in ("legit", "hard_clone", "easy_clone")]), "",
          f"Hard-clone catch rate **{pct(ver['catch_rate_hard'])}**, easy-clone catch rate {pct(ver['catch_rate_easy'])}, "
          f"false-alarm rate on legitimate claims {pct(ver['false_alarm_rate'])} (strict policy, thresholds fixed on validation)."]
    ud = F.get("unknown_detection", {})
    if "probe_n" in ud:
        L += ["", f"Cars outside the dataset: {pct(ud['probe_flag_rate'])} of {ud['probe_n']} modern-car photos flagged UNKNOWN; "
                  f"AUROC kNN {ud['auroc_knn']:.3f} vs max-softmax {ud['auroc_msp']:.3f}; in-dataset test flag rate "
                  f"{pct(ud['test_flag_rate'])}."]

    L += ["", "## 7. Ablation study (test set)", "",
          table(["Variant", "Top-1", "ECE", "Hard-clone catch", "False alarms"],
                [[k, pct(v["top1"]), f"{v['ece']:.3f}", pct(v["catch_rate_hard"]), pct(v["false_alarm_rate"])]
                 for k, v in F["ablation"].items()]), ""]
    if H:
        L += ["Training log (anti-overfitting evidence):", "",
              table(["Stage", "Epoch", "Train top-1 (aug.)", "Train top-1 (clean)", "Val top-1", "Val loss", "Gap (pts)"],
                    [[h["stage"], h["epoch"], pct(h["train_top1_aug"]), pct(h["train_top1_clean"]), pct(h["val_top1"]),
                      f"{h['val_loss']:.3f}", f"{h['gap_points']:+.1f}"] for h in H]), ""]
    L += ["## 8. Failure cases", ""]
    for f in fails:
        L += [f"### {f['title']}", f"True: *{f['true']}*. Predicted: *{f['pred']}* ({f['conf']:.0%}).", "",
              f"![{f['title']}](../results/failures/{f['image']})", "", f["explanation"], ""]
    out = ROOT / "docs" / f"eval_P5_{a.team}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
