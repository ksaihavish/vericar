"""Stage 5 - hybrid fusion, unknown-car detection and claim verification.

* ``fuse``: temperature-calibrated CNN log-probs + weighted classical-SVM log-probs
  (weight and temperature are fitted on the validation split only).
* ``KnnOOD``: an image is "not in my knowledge" when its CNN embedding is far
  from every training embedding (cosine kNN distance). Threshold = 95th
  percentile of validation distances, so ~5% of known cars are sent to manual check.
* ``verify``: strict level-by-level check of a claimed registration
  (make -> model -> year) producing MATCH / PARTIAL / MISMATCH / INSPECT / UNKNOWN / RETAKE.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Hierarchy


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fuse(logits: np.ndarray, temperature: float = 1.0, p_classical: np.ndarray | None = None,
         weight: float = 0.0) -> np.ndarray:
    """Log-linear fusion of calibrated CNN probabilities with classical probabilities."""
    p = softmax(logits, temperature)
    if p_classical is None or weight <= 0:
        return p
    logp = (1 - weight) * np.log(p + 1e-9) + weight * np.log(p_classical + 1e-6)
    return softmax(logp)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017) by grid search on validation NLL."""
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.5, 3.0, 51):
        p = softmax(logits, t)
        nll = -np.log(p[np.arange(len(labels)), labels] + 1e-12).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


class KnnOOD:
    """Cosine k-nearest-neighbour distance to the training embeddings."""

    def __init__(self, bank: np.ndarray, k: int = 10, threshold: float | None = None):
        b = bank.astype(np.float32)
        self.bank = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        self.k, self.threshold = k, threshold

    def score(self, feats: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        f = feats.astype(np.float32)
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        out = np.empty(len(f), np.float32)
        for i in range(0, len(f), 512):
            sims = f[i:i + 512] @ self.bank.T
            k = self.k + (1 if exclude_self else 0)
            top = -np.partition(-sims, k - 1, axis=1)[:, :k]
            top = np.sort(top, axis=1)[:, ::-1][:, (1 if exclude_self else 0):]
            out[i:i + 512] = 1.0 - top.mean(1)
        return out

    def is_unknown(self, feats: np.ndarray) -> np.ndarray:
        return self.score(feats) > self.threshold


# ------------------------------------------------------------------ verification
@dataclass
class Thresholds:
    match: float = 0.80        # P(exact claimed class) needed to clear a claim
    model: float = 0.80        # P(claimed model family) needed for PARTIAL (year unverified)
    mismatch: float = 0.05     # P(claimed model family) below this = MISMATCH
    identify: float = 0.60     # identify mode: below this show "low confidence" + top-3

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class Verdict:
    status: str
    headline: str
    detail: str
    levels: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)

    def as_dict(self):
        return dict(self.__dict__)


def top_candidates(probs: np.ndarray, hier: Hierarchy, k: int = 3) -> list[dict]:
    idx = np.argsort(-probs)[:k]
    return [{**hier.describe(int(i)), "class_idx": int(i), "prob": float(probs[i])} for i in idx]


def educated_guess(probs: np.ndarray, hier: Hierarchy) -> dict:
    """Coarse guess for cars outside the 196 classes: make + body + nearest known class."""
    pm = hier.group_probs(probs, "make")
    pb = hier.group_probs(probs, "body")
    return {"make": hier.make_names[int(pm.argmax())], "make_prob": float(pm.max()),
            "body": hier.body_names[int(pb.argmax())], "body_prob": float(pb.max()),
            "closest": hier.class_names[int(probs.argmax())]}


def verify(probs: np.ndarray, claim: int, hier: Hierarchy, thr: Thresholds,
           unknown: bool = False, quality_issues: list[str] | None = None) -> Verdict:
    """Strict claim verification. Anything that is not MATCH means 'manual check'."""
    claimed = hier.describe(claim)
    p_make = float(probs[hier.make_of == hier.make_of[claim]].sum())
    p_model = float(probs[hier.model_of == hier.model_of[claim]].sum())
    p_exact = float(probs[claim])
    # P(year | model) only means something once the model itself is plausible
    p_year = p_exact / p_model if p_model >= thr.mismatch else p_exact
    levels = [
        {"level": "Make", "claimed": claimed["make"], "prob": p_make},
        {"level": "Model", "claimed": claimed["model"].removeprefix(claimed["make"]).strip(), "prob": p_model},
        {"level": "Year", "claimed": str(claimed["year"]), "prob": p_year},
    ]
    cands = top_candidates(probs, hier, 3)
    for lv in levels:
        lv["state"] = "ok" if lv["prob"] >= thr.model else ("fail" if lv["prob"] < thr.mismatch else "unsure")

    if quality_issues:
        return Verdict("RETAKE", "Retake the photo", "Image quality is too low to verify: "
                       + ", ".join(quality_issues) + ".", levels, cands)
    if unknown:
        g = educated_guess(probs, hier)
        return Verdict("UNKNOWN", "Cannot verify - car not in my knowledge",
                       f"This car does not resemble any of the 196 models I was trained on. Educated guess: "
                       f"{g['make']} ({g['make_prob']:.0%}), body {g['body']}. Manual check required.", levels, cands)
    if p_exact >= thr.match and int(np.argmax(probs)) == claim:   # never clear a claim that is not the top guess
        return Verdict("MATCH", "Match - registration verified",
                       f"The photo matches {claimed['label']} ({p_exact:.0%}).", levels, cands)
    if p_model >= thr.model:
        alt = [c for c in cands if c["class_idx"] != claim and hier.model_of[c["class_idx"]] == hier.model_of[claim]]
        alt_txt = f" vs {alt[0]['year']} ({alt[0]['prob']:.0%})" if alt else ""
        return Verdict("PARTIAL", "Partial match - year not verifiable",
                       f"Make and model verified ({p_model:.0%}). The year cannot be separated from this photo: "
                       f"{claimed['year']} ({p_exact:.0%}){alt_txt}. Check documents for the year.", levels, cands)
    if p_model < thr.mismatch:
        top = cands[0]
        return Verdict("MISMATCH", "Mismatch - possible cloned plate",
                       f"The photo looks like {top['label']} ({top['prob']:.0%}), not the registered "
                       f"{claimed['label']} ({p_exact:.1%}).", levels, cands)
    return Verdict("INSPECT", "Inconclusive - manual check",
                   f"Evidence for {claimed['label']} is {p_exact:.0%}; not strong enough to clear or to flag.",
                   levels, cands)


def identify(probs: np.ndarray, hier: Hierarchy, thr: Thresholds, unknown: bool = False,
             quality_issues: list[str] | None = None) -> Verdict:
    cands = top_candidates(probs, hier, 3)
    top = cands[0]
    if quality_issues:
        return Verdict("RETAKE", "Retake the photo", "Image quality too low: " + ", ".join(quality_issues) + ".",
                       [], cands)
    if unknown:
        g = educated_guess(probs, hier)
        return Verdict("UNKNOWN", "Not in my knowledge",
                       f"Likely outside the 196 known models. Educated guess: {g['make']} "
                       f"({g['make_prob']:.0%}), {g['body']}; closest known: {g['closest']}.", [], cands)
    if top["prob"] >= thr.identify:
        return Verdict("CONFIDENT", top["label"], f"{top['prob']:.0%} confident.", [], cands)
    return Verdict("LOW_CONFIDENCE", "Low confidence - top candidates",
                   f"Best guess {top['label']} at only {top['prob']:.0%}; showing the top 3.", [], cands)
