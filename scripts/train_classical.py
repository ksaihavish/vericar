"""Classical baseline: handcrafted features (HOG, LBP, GLCM, HSV/Lab, geometry) + SVM / RF.

Features are extracted from ground-truth car crops of the TRAIN split; model
selection and the feature-group ablation use the VAL split only.
Usage:  python scripts/train_classical.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from vericar import classical as cl  # noqa: E402
from vericar.config import FEAT_DIR, MODELS_DIR, RESULTS_DIR, SEED  # noqa: E402
from vericar.data import load_split  # noqa: E402
from vericar.evaluation import topk_accuracy  # noqa: E402
from vericar.preprocessing import crop_box, load_rgb  # noqa: E402


def _feat(path, box):
    return cl.extract_features(crop_box(load_rgb(path), box, 0.0))


def features_for(split: str, jobs: int = 8) -> tuple[np.ndarray, np.ndarray]:
    cache = FEAT_DIR / f"classical_{split}_gt.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    recs = load_split(split)
    X = np.stack(Parallel(n_jobs=jobs, batch_size=64)(delayed(_feat)(r["path"], r["box"]) for r in recs))
    y = np.array([r["label"] for r in recs])
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def svm(C=10.0, n_pca=256, n_features=None):
    n_pca = min(n_pca, n_features - 1) if n_features else n_pca
    return make_pipeline(StandardScaler(), PCA(n_pca, whiten=True, random_state=SEED),
                         SVC(C=C, kernel="rbf", gamma="scale", probability=True, random_state=SEED))


def main():
    t0 = time.time()
    Xtr, ytr = features_for("train")
    Xva, yva = features_for("val")
    print(f"features: train {Xtr.shape} val {Xva.shape} ({time.time() - t0:.0f}s)", flush=True)

    candidates = {
        "SVM-RBF C=3": svm(3.0), "SVM-RBF C=10": svm(10.0),
        "RandomForest-500": make_pipeline(StandardScaler(), PCA(256, random_state=SEED),
                                          RandomForestClassifier(500, n_jobs=4, random_state=SEED)),
    }
    scores, fitted = {}, {}
    for name, model in candidates.items():
        t = time.time()
        model.fit(Xtr, ytr)
        acc = topk_accuracy(model.predict_proba(Xva), yva)
        scores[name] = {**acc, "fit_sec": round(time.time() - t, 1)}
        fitted[name] = model
        print(name, scores[name], flush=True)
    best_name = max(scores, key=lambda k: scores[k]["top1"])

    # feature-family ablation (val): drop one family at a time, same model type as the winner
    groups = cl.feature_group_slices()
    ablation = {}
    base = fitted[best_name]
    for g, sl in groups.items():
        keep = np.ones(Xtr.shape[1], bool)
        keep[sl] = False
        m = svm(base[-1].C, n_features=int(keep.sum())) if "SVM" in best_name else candidates[best_name]
        m.fit(Xtr[:, keep], ytr)
        ablation[f"without {g}"] = topk_accuracy(m.predict_proba(Xva[:, keep]), yva)
        print("ablation", g, ablation[f"without {g}"], flush=True)

    joblib.dump(fitted[best_name], MODELS_DIR / "classical_svm.joblib", compress=3)
    out = {"model_selection_val": scores, "selected": best_name, "feature_dims": {k: v.stop - v.start for k, v in groups.items()},
           "feature_family_ablation_val": ablation}
    (RESULTS_DIR / "classical_val.json").write_text(json.dumps(out, indent=1))
    print("DONE", best_name, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
