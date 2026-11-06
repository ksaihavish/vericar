"""Batch inference helpers shared by fit_hybrid.py and evaluate_final.py (GPU, offline only).

They reproduce exactly what ``VeriCar.analyze`` does per image (same crop margin,
quality gate, adaptive enhancement, ``classifier.preprocess``), just batched.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from vericar import classical as cl  # noqa: E402
from vericar.classifier import build_model, preprocess  # noqa: E402
from vericar.config import CROP_MARGIN, MODELS_DIR  # noqa: E402
from vericar.preprocessing import adaptive_enhance, crop_box, load_rgb, quality_report  # noqa: E402


# ------------------------------------------------------------------ perturbations (robustness study)
def _occlude(img, box, frac, rng):
    x1, y1, x2, y2 = map(int, box)
    bw, bh = max(2, x2 - x1), max(2, y2 - y1)
    area = frac * bw * bh
    ar = rng.uniform(0.5, 2.0)
    w = int(min(bw, np.sqrt(area * ar)))
    h = int(min(bh, area / max(w, 1)))
    px, py = rng.integers(x1, max(x1 + 1, x2 - w)), rng.integers(y1, max(y1 + 1, y2 - h))
    out = img.copy()
    out[py:py + h, px:px + w] = 127
    return out


def _lowres(img, f):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


PERTURBATIONS = {
    "clean": lambda im, b, r: im,
    "dark (x0.35)": lambda im, b, r: (im * 0.35).astype(np.uint8),
    "bright (x1.6)": lambda im, b, r: np.clip(im * 1.6, 0, 255).astype(np.uint8),
    "low contrast (x0.4)": lambda im, b, r: np.clip((im - im.mean()) * 0.4 + im.mean(), 0, 255).astype(np.uint8),
    "occlusion 10%": lambda im, b, r: _occlude(im, b, 0.10, r),
    "occlusion 25%": lambda im, b, r: _occlude(im, b, 0.25, r),
    "occlusion 40%": lambda im, b, r: _occlude(im, b, 0.40, r),
    "low-res 1/4": lambda im, b, r: _lowres(im, 4),
    "low-res 1/8": lambda im, b, r: _lowres(im, 8),
    "blur sigma 2": lambda im, b, r: cv2.GaussianBlur(im, (0, 0), 2),
    "blur sigma 4": lambda im, b, r: cv2.GaussianBlur(im, (0, 0), 4),
    "noise sigma 20": lambda im, b, r: np.clip(im + r.normal(0, 20, im.shape), 0, 255).astype(np.uint8),
}


def load_perturbed(rec: dict, name: str, seed: int) -> np.ndarray:
    img = load_rgb(rec["path"])
    return PERTURBATIONS[name](img, rec["box"], np.random.default_rng(seed))


# ------------------------------------------------------------------ detection
def detect_all(weights, recs, perturb="clean", coco=False, conf=0.001, bs=32):
    """All detections per record as [(box, score), ...]. Perturbed images are passed as arrays."""
    from vericar.detector import COCO_VEHICLES
    from ultralytics import YOLO
    model = YOLO(str(weights), task="detect")
    out = []
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]
        if perturb == "clean":
            src = [r["path"] for r in chunk]
        else:
            src = [load_perturbed(r, perturb, 1000 + i + j)[..., ::-1] for j, r in enumerate(chunk)]
        for res in model.predict(src, imgsz=640, conf=conf, verbose=False, device=0, max_det=20):
            dets = []
            for b, s, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                if coco and int(c) not in COCO_VEHICLES:
                    continue
                dets.append((tuple(float(v) for v in b), float(s)))
            out.append(sorted(dets, key=lambda d: -d[1]))
    return out


def primary_box(dets):
    """Same rule as VeriCar._locate: best box at conf>=0.25, else >=0.05, else None."""
    for t in (0.25, 0.05):
        ok = [d for d in dets if d[1] >= t]
        if ok:
            return ok[0][0]
    return None


# ------------------------------------------------------------------ crops -> CNN / classical
class CropSet(Dataset):
    """mode: 'det' (given box, classical fallback), 'gt', 'full', 'classical' (classical ROI)."""

    def __init__(self, recs, boxes=None, mode="det", enhance=True, classical_feats=False, perturb="clean"):
        self.recs, self.boxes, self.mode = recs, boxes, mode
        self.enhance, self.classical_feats, self.perturb = enhance, classical_feats, perturb

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = load_rgb(r["path"]) if self.perturb == "clean" else load_perturbed(r, self.perturb, 1000 + i)
        fallback = 0
        if self.mode == "gt":
            box = r["box"]
        elif self.mode == "full":
            box = (0, 0, img.shape[1], img.shape[0])
        elif self.mode == "classical":
            box = cl.classical_roi(img)["box"]
        else:
            box = self.boxes[i]
            if box is None:
                box, fallback = cl.classical_roi(img)["box"], 1
        crop = crop_box(img, box, 0.0 if self.mode == "classical" else CROP_MARGIN)
        q = quality_report(crop)
        steps = []
        if self.enhance:
            crop, steps = adaptive_enhance(crop, q)
        x = preprocess(crop)[0] if self.mode != "classical" else np.zeros((3, 8, 8), np.float32)
        f = cl.extract_features(crop) if self.classical_feats else np.zeros(1, np.float32)
        return x, f, int(r["label"]), int(bool(q.issues) or fallback), int(bool(steps))


def load_torch_classifier(dev="cuda"):
    ck = torch.load(MODELS_DIR / "classifier.pt", map_location="cpu", weights_only=False)
    m = build_model(len(ck["class_names"]), pretrained=False)
    m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    return m.eval().to(dev).to(memory_format=torch.channels_last)


@torch.no_grad()
def run_set(ds: CropSet, model=None, svm=None, bs=64, workers=8) -> dict:
    """Returns logits/feats (if model), classical probs (if svm & features), labels, flags."""
    L, F, P, Y, Q, E = [], [], [], [], [], []
    for x, f, y, qi, en in DataLoader(ds, batch_size=bs, num_workers=workers, persistent_workers=False):
        if model is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo, fe = model(x.cuda().to(memory_format=torch.channels_last))
            L.append(lo.float().cpu().numpy())
            F.append(fe.float().cpu().numpy())
        if svm is not None and ds.classical_feats:
            P.append(svm.predict_proba(f.numpy()))
        Y.append(y.numpy())
        Q.append(qi.numpy())
        E.append(en.numpy())
    out = {"labels": np.concatenate(Y), "flagged": np.concatenate(Q).astype(bool), "enhanced": np.concatenate(E).astype(bool)}
    if L:
        out["logits"], out["feats"] = np.concatenate(L), np.concatenate(F)
    if P:
        out["p_classical"] = np.concatenate(P)
    return out
