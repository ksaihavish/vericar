"""Stage 6 - evaluation metrics (classification, localisation, calibration, verification)."""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ classification
def topk_accuracy(probs: np.ndarray, labels: np.ndarray, ks=(1, 5)) -> dict[str, float]:
    order = np.argsort(-probs, axis=1)
    return {f"top{k}": float((order[:, :k] == labels[:, None]).any(1).mean()) for k in ks}


def level_accuracy(probs: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """Accuracy after summing class probabilities into groups (make / model level)."""
    n_groups = groups.max() + 1
    gp = np.zeros((probs.shape[0], n_groups), np.float64)
    np.add.at(gp.T, groups, probs.T)
    return float((gp.argmax(1) == groups[labels]).mean())


def year_given_model_accuracy(probs, labels, model_of_class) -> float:
    """Of images whose model was predicted correctly, how often is the year (exact class) right?"""
    pred = probs.argmax(1)
    ok_model = model_of_class[pred] == model_of_class[labels]
    return float((pred[ok_model] == labels[ok_model]).mean()) if ok_model.any() else float("nan")


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    """Expected calibration error + per-bin data for a reliability diagram."""
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    total, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            total += m.mean() * gap
            bins.append({"lo": float(lo), "hi": float(hi), "conf": float(conf[m].mean()),
                         "acc": float(correct[m].mean()), "n": int(m.sum())})
    return float(total), bins


def coverage_curve(conf: np.ndarray, correct: np.ndarray, thresholds=None) -> list[dict]:
    """Accuracy of the predictions we keep vs how many we keep, per confidence threshold."""
    thresholds = np.linspace(0, 0.95, 20) if thresholds is None else thresholds
    out = []
    for t in thresholds:
        keep = conf >= t
        out.append({"threshold": float(t), "coverage": float(keep.mean()),
                    "accuracy": float(correct[keep].mean()) if keep.any() else float("nan")})
    return out


def confused_pairs(probs, labels, class_names, top: int = 15) -> list[dict]:
    pred = probs.argmax(1)
    pairs: dict[tuple[int, int], int] = {}
    for t, p in zip(labels, pred):
        if t != p:
            pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1])[:top]
    return [{"true": class_names[t], "pred": class_names[p], "count": c} for (t, p), c in ranked]


# ------------------------------------------------------------------ localisation
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def average_precision(dets: list[tuple[int, float, tuple]], gts: dict[int, tuple], thr: float) -> float:
    """Single-class AP (COCO 101-point) with one ground-truth box per image."""
    dets = sorted(dets, key=lambda d: -d[1])
    matched: set[int] = set()
    tp = np.zeros(len(dets))
    for i, (img_id, _, box) in enumerate(dets):
        if img_id in gts and img_id not in matched and iou(box, gts[img_id]) >= thr:
            tp[i] = 1
            matched.add(img_id)
    if not len(dets):
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / max(1, len(gts))
    precision = ctp / np.arange(1, len(dets) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # precision envelope
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, r, side="left")
        ap += precision[idx] if idx < len(precision) else 0.0
    return float(ap / 101)


def detection_report(dets, gts) -> dict:
    """mAP@0.5, mAP@0.5:0.95 and IoU stats of the best-scoring box per image."""
    best: dict[int, tuple] = {}
    for img_id, score, box in dets:
        if img_id not in best or score > best[img_id][0]:
            best[img_id] = (score, box)
    ious = np.array([iou(best[i][1], g) if i in best else 0.0 for i, g in gts.items()])
    maps = [average_precision(dets, gts, t) for t in np.arange(0.5, 0.96, 0.05)]
    return {"mAP50": maps[0], "mAP50_95": float(np.mean(maps)), "mean_iou": float(ious.mean()),
            "iou_ge_0.5": float((ious >= 0.5).mean()), "iou_ge_0.75": float((ious >= 0.75).mean()),
            "miss_rate": float(np.mean([i not in best for i in gts]))}


# ------------------------------------------------------------------ verification / OOD
def claim_sets(labels: np.ndarray, hard: np.ndarray, easy: np.ndarray) -> dict[str, np.ndarray]:
    """Legit claim = true class; hard clone = most similar other class; easy clone = random other make."""
    return {"legit": labels, "hard_clone": hard[labels], "easy_clone": easy[labels]}


def cleared_rate(probs, claims, unknown, flagged, t: float) -> float:
    """Fraction of claims that get MATCH (cleared) at threshold t (vectorised fast path)."""
    p = probs[np.arange(len(claims)), claims]
    return float(((p >= t) & ~unknown & ~flagged).mean())


def clone_study(probs, labels, unknown, flagged, hier, thr, hard, easy) -> dict:
    """Full verdict breakdown for legit / hard-clone / easy-clone claims."""
    from collections import Counter

    from .hybrid import verify
    out = {}
    for kind, claims in claim_sets(labels, hard, easy).items():
        st = Counter(verify(p, int(c), hier, thr, bool(u), ["flag"] if f else None).status
                     for p, c, u, f in zip(probs, claims, unknown, flagged))
        out[kind] = {s: st[s] / len(claims) for s in ("MATCH", "PARTIAL", "INSPECT", "MISMATCH", "UNKNOWN", "RETAKE")}
    out["false_alarm_rate"] = 1 - out["legit"]["MATCH"]
    out["catch_rate_hard"] = 1 - out["hard_clone"]["MATCH"]
    out["catch_rate_easy"] = 1 - out["easy_clone"]["MATCH"]
    return out


def auroc(scores_neg: np.ndarray, scores_pos: np.ndarray) -> float:
    """AUROC where higher score should mean 'positive' (e.g. out-of-distribution)."""
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(scores_neg)), np.ones(len(scores_pos))]
    return float(roc_auc_score(y, np.r_[scores_neg, scores_pos]))
