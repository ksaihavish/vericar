"""The single entry point: ``VeriCar().analyze(image, claim=None)``.

Runs the three pipelines on one image and returns everything the UI needs:

* classical : classical ROI -> handcrafted features -> SVM
* deep      : YOLO ROI -> ConvNeXt (raw softmax)
* hybrid    : YOLO ROI (classical fallback) -> quality gate -> adaptive enhancement
              -> ConvNeXt -> calibrated fusion with classical features -> kNN unknown check
              -> verdict (verify a claimed registration, or identify)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import classical as cl
from .classifier import Classifier, preprocess
from .config import CACHE_MAX_SIDE, CROP_MARGIN, MODELS_DIR
from .data import Hierarchy
from .detector import Detector
from .hybrid import KnnOOD, Thresholds, fuse, identify, softmax, top_candidates, verify
from .preprocessing import adaptive_enhance, crop_box, load_rgb, quality_report, resize_max

STATUS_COLOURS = {  # RGB
    "MATCH": (34, 197, 94), "CONFIDENT": (34, 197, 94), "PARTIAL": (245, 158, 11),
    "LOW_CONFIDENCE": (245, 158, 11), "INSPECT": (245, 158, 11), "MISMATCH": (239, 68, 68),
    "UNKNOWN": (139, 92, 246), "RETAKE": (148, 163, 184),
}


class VeriCar:
    def __init__(self, models_dir: Path = MODELS_DIR, onnx_file: str | None = None):
        models_dir = Path(models_dir)
        meta = json.loads((models_dir / "hybrid.json").read_text())
        self.meta = meta
        self.hier = Hierarchy.from_names(meta["class_names"])
        self.detector = Detector(models_dir / "detector.pt")
        onnx = models_dir / (onnx_file or meta.get("onnx_file", "classifier.onnx"))
        self.classifier = Classifier(onnx_path=onnx, torch_path=models_dir / "classifier.pt")
        import joblib
        self.svm = joblib.load(models_dir / "classical_svm.joblib")
        self.temperature = meta["temperature"]
        self.fusion_weight = meta["fusion_weight"]
        self.thr = Thresholds(**meta["thresholds"])
        self.ood = KnnOOD(np.load(models_dir / "ood_bank.npy"), k=meta["ood_k"], threshold=meta["ood_threshold"])

    # ------------------------------------------------------------------ helpers
    def _svm_probs(self, crop: np.ndarray) -> np.ndarray:
        return self.svm.predict_proba(cl.extract_features(crop)[None])[0]

    def _locate(self, img: np.ndarray) -> tuple[tuple, str, float]:
        dets = self.detector.detect(img, conf=0.25) or self.detector.detect(img, conf=0.05)
        if dets:
            return dets[0][0], "yolo", dets[0][1]
        return cl.classical_roi(img)["box"], "classical-fallback", 0.0

    # ------------------------------------------------------------------ main
    def analyze(self, image, claim: int | None = None, identify_threshold: float | None = None,
                gradcam: bool = False, intermediates: bool = True) -> dict:
        t = {}
        t0 = time.perf_counter()
        img, _ = resize_max(load_rgb(image), CACHE_MAX_SIDE)
        t["load"] = time.perf_counter() - t0

        # ---- detection
        t1 = time.perf_counter()
        box, box_src, det_score = self._locate(img)
        crop = crop_box(img, box, CROP_MARGIN)
        t["detect"] = time.perf_counter() - t1

        # ---- classical-only pipeline (no learning-based localisation)
        t1 = time.perf_counter()
        croi = cl.classical_roi(img)
        p_classical = self._svm_probs(crop_box(img, croi["box"], 0.0))
        t["classical"] = time.perf_counter() - t1

        # ---- deep-only pipeline
        t1 = time.perf_counter()
        x_raw = preprocess(crop)
        logits_raw, feats_raw = self.classifier.predict(x_raw)
        p_deep = softmax(logits_raw[0])
        t["deep"] = time.perf_counter() - t1

        # ---- hybrid pipeline
        t1 = time.perf_counter()
        q = quality_report(crop)
        issues = list(q.issues)
        if box_src != "yolo":
            issues.append("no vehicle detected")
        enhanced, steps = adaptive_enhance(crop, q)
        if steps:
            x_h = preprocess(enhanced)
            logits_h, feats_h = self.classifier.predict(x_h)
        else:
            x_h, logits_h, feats_h = x_raw, logits_raw, feats_raw
        p_svm_roi = self._svm_probs(enhanced) if self.fusion_weight > 0 else None
        p_hybrid = fuse(logits_h[0], self.temperature, p_svm_roi, self.fusion_weight)
        ood_score = float(self.ood.score(feats_h)[0])
        unknown = ood_score > self.ood.threshold
        thr = self.thr if identify_threshold is None else Thresholds(**{**self.thr.as_dict(),
                                                                         "identify": identify_threshold})
        if claim is None:
            verdict = identify(p_hybrid, self.hier, thr, unknown, issues)
        else:
            verdict = verify(p_hybrid, int(claim), self.hier, thr, unknown, issues)
        t["hybrid"] = time.perf_counter() - t1

        top = top_candidates(p_hybrid, self.hier, 5)
        out = {
            "box": [round(v, 1) for v in box], "box_source": box_src, "det_score": det_score,
            "image_size": list(img.shape[:2]),
            "prediction": top[0], "confidence": top[0]["prob"],
            "verdict": verdict.as_dict(),
            "quality": q.as_dict(), "enhancement": steps,
            "ood_score": ood_score, "ood_threshold": self.ood.threshold, "unknown": bool(unknown),
            "methods": {
                "classical": top_candidates(p_classical, self.hier, 5),
                "deep": top_candidates(p_deep, self.hier, 5),
                "hybrid": top,
            },
            "levels": {
                "make": _top_group(self.hier, p_hybrid, "make"),
                "model": _top_group(self.hier, p_hybrid, "model"),
            },
            "timings_ms": {k: round(v * 1000, 1) for k, v in t.items()},
        }
        out["timings_ms"]["total"] = round((time.perf_counter() - t0) * 1000, 1)
        if intermediates:
            vis = cl.visualize(img, box)
            out["images"] = {
                "input": img,
                "annotated": annotate(img, box, verdict.status, top[0]["label"], top[0]["prob"], croi["box"]),
                "edges": vis["edges"], "contours": vis["contours"], "hog": vis["hog"], "lbp": vis["lbp"],
                "crop": crop, "enhanced": enhanced,
                "classical_crop": crop_box(img, croi["box"], 0.0),
            }
            out["histograms"] = cl.hsv_histograms(crop)
        if gradcam:
            out.setdefault("images", {})["gradcam"] = overlay_cam(enhanced, self.classifier.gradcam(
                x_h, int(top[0]["class_idx"])))
        return out


def _top_group(hier: Hierarchy, probs: np.ndarray, level: str) -> dict:
    g = hier.group_probs(probs, level)
    names = hier.make_names if level == "make" else hier.model_names
    i = int(g.argmax())
    return {"name": names[i], "prob": float(g[i])}


def annotate(img, box, status, label, prob, classical_box=None) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    th = max(2, round(max(h, w) / 300))
    if classical_box is not None:
        x1, y1, x2, y2 = map(int, classical_box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)
    col = STATUS_COLOURS.get(status, (59, 130, 246))
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(out, (x1, y1), (x2, y2), col, th, cv2.LINE_AA)
    text = f"{label}  {prob:.0%}"
    fs = max(0.45, max(h, w) / 1100)
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, th - 1))
    ty = y1 - 6 if y1 - tht - 10 > 0 else y1 + tht + 10
    cv2.rectangle(out, (x1, ty - tht - 6), (min(w - 1, x1 + tw + 10), ty + base), col, -1)
    cv2.putText(out, text, (x1 + 5, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, th - 1),
                cv2.LINE_AA)
    return out


def overlay_cam(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return cv2.addWeighted(img, 0.55, heat, 0.45, 0)
