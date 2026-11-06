"""Download Stanford Cars, verify the test labels, cache resized images, build the split.

* Images: kagglehub "eduardo4jesus/stanford-cars-dataset" (train/test images + devkit).
* Test labels: that copy has none, so ``cars_test_annos_withlabels.mat`` comes from
  "abdelrahmant11/standford-cars-dataset-meta". We verify that every test file name
  and bounding box matches the original test annotation file before using it.
* Split: official train set -> 85% train / 15% val, stratified by class, seed 42.
  The official test set is cached but only ever read by scripts/evaluate_final.py.

Usage:  python scripts/prepare_data.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from vericar.config import CACHE_MAX_SIDE, DATA_DIR, MODELS_DIR, SEED, VAL_FRACTION  # noqa: E402
from vericar.preprocessing import quality_report  # noqa: E402


def annos(mat_path: Path) -> list[dict]:
    a = sio.loadmat(str(mat_path), squeeze_me=True)["annotations"]
    out = []
    for r in a:
        d = {"fname": str(r["fname"]),
             "box": [float(r["bbox_x1"]) - 1, float(r["bbox_y1"]) - 1, float(r["bbox_x2"]), float(r["bbox_y2"])]}
        if "class" in a.dtype.names:
            d["label"] = int(r["class"]) - 1
        out.append(d)
    return out


def cache_image(src: Path, dst: Path, box: list[float]) -> dict:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    s = min(1.0, CACHE_MAX_SIDE / max(w, h))
    if s < 1.0:
        img = img.resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, quality=95)
    W, H = img.size
    b = [max(0.0, box[0] * s), max(0.0, box[1] * s), min(W, box[2] * s), min(H, box[3] * s)]
    q = quality_report(np.asarray(img))
    return {"box": [round(v, 2) for v in b], "width": W, "height": H, "orig_size": [w, h],
            "brightness": q.brightness, "contrast": q.contrast, "sharpness": q.sharpness}


def main():
    import kagglehub
    root = Path(kagglehub.dataset_download("eduardo4jesus/stanford-cars-dataset"))
    meta = Path(kagglehub.dataset_download("abdelrahmant11/standford-cars-dataset-meta"))
    dev = root / "car_devkit" / "devkit"

    names = [str(n) for n in sio.loadmat(str(dev / "cars_meta.mat"), squeeze_me=True)["class_names"]]
    assert len(names) == 196
    train = annos(dev / "cars_train_annos.mat")
    test_nolabel = annos(dev / "cars_test_annos.mat")
    test = annos(next(meta.glob("cars_test_annos_withlabels*.mat")))

    # ---- integrity check of the external label file
    assert len(test) == len(test_nolabel) == 8041, (len(test), len(test_nolabel))
    for a, b in zip(test, test_nolabel):
        assert a["fname"] == b["fname"] and np.allclose(a["box"], b["box"]), (a, b)
    assert sorted({t["label"] for t in test}) == list(range(196))
    print(f"test labels verified: {len(test)} files and boxes match the original annotation file")

    jobs = []
    for split, recs, folder in (("train", train, root / "cars_train" / "cars_train"),
                                ("test", test, root / "cars_test" / "cars_test")):
        for r in recs:
            r["src"] = folder / r["fname"]
            r["rel"] = f"images/{split}/{r['fname']}"
            jobs.append(r)
    info = Parallel(n_jobs=12, batch_size=64)(delayed(cache_image)(r["src"], DATA_DIR / r["rel"], r["box"]) for r in jobs)
    for r, i in zip(jobs, info):
        r.update(i)

    def rec(r):
        return {"id": Path(r["fname"]).stem, "path": r["rel"], "box": r["box"], "label": r["label"],
                "width": r["width"], "height": r["height"], "orig_size": r["orig_size"]}

    labels = np.array([r["label"] for r in train])
    tr_idx, va_idx = train_test_split(np.arange(len(train)), test_size=VAL_FRACTION, stratify=labels,
                                      random_state=SEED)
    splits = {"train": [rec(train[i]) for i in sorted(tr_idx)], "val": [rec(train[i]) for i in sorted(va_idx)],
              "test": [rec(r) for r in test]}
    for k, v in splits.items():
        (DATA_DIR / f"{k}.json").write_text(json.dumps(v))
    (DATA_DIR / "classes.json").write_text(json.dumps(names, indent=0))
    shutil.copy2(DATA_DIR / "classes.json", MODELS_DIR / "classes.json")

    # ---- sanity statistics for the quality-gate thresholds (train split only)
    tr = [train[i] for i in tr_idx]
    stats = {k: np.percentile([r[k] for r in tr], [1, 5, 50]).round(1).tolist()
             for k in ("brightness", "contrast", "sharpness")}
    summary = {"n": {k: len(v) for k, v in splits.items()}, "classes": len(names),
               "per_class_val_min": int(np.bincount(labels[va_idx]).min()),
               "quality_percentiles_1_5_50_train": stats}
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
