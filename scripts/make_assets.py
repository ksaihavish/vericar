"""Build UI assets: one reference thumbnail per class + the demo gallery.

* assets/ref/NNN.jpg - the most typical TRAINING image of each class (closest to the
  class prototype in the CNN embedding space), cropped to its box.
* assets/demo/      - demo photos for the 3-act story (clone caught, year ambiguity,
  car outside the dataset) plus a few extra cases; chosen from the test outputs
  AFTER the final evaluation (no tuning happens on them).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vericar.config import ASSETS_DIR, DATA_DIR, FEAT_DIR, REPO_ROOT  # noqa: E402
from vericar.data import hierarchy, load_split  # noqa: E402
from vericar.preprocessing import crop_box, load_rgb  # noqa: E402


def save_crop(rec, dst: Path, width=360, margin=0.06):
    img = crop_box(load_rgb(rec["path"]), rec["box"], margin)
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])


def save_full(path, dst: Path, max_side=900, transform=None):
    img = load_rgb(path)
    if transform is not None:
        img = transform(img)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])


def references():
    hier = hierarchy()
    d = np.load(FEAT_DIR / "cnn_train.npz")
    f = d["feats"].astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    recs = load_split("train")
    for c in range(len(hier.class_names)):
        idx = np.flatnonzero(d["labels"] == c)
        proto = f[idx].mean(0)
        best = idx[int(np.argmax(f[idx] @ proto))]
        save_crop(recs[best], ASSETS_DIR / "ref" / f"{c:03d}.jpg")
    print("references done")


def demos():
    out_p = DATA_DIR / "test_outputs.npz"
    if not out_p.exists():
        print("no test outputs yet - skipping demo gallery")
        return
    hier = hierarchy()
    names = list(hier.class_names)
    o = np.load(out_p)
    p, y = o["p_hyb"], o["y"]
    test = load_split("test")
    ddir = ASSETS_DIR / "demo"
    items = []

    def best_of(cls_name, score):
        c = names.index(cls_name)
        idx = np.flatnonzero(y == c)
        return int(idx[np.argmax(score(idx, c))])

    # Act 1 - clone caught: photo of an S5 Coupe, registration claims S4 Sedan 2012
    i = best_of("Audi S5 Coupe 2012", lambda idx, c: p[idx, c])
    save_full(test[i]["path"], ddir / "1_clone_s5.jpg")
    items.append({"file": "1_clone_s5.jpg", "title": "Clone?", "claim": "Audi S4 Sedan 2012",
                  "note": "Registration says Audi S4 Sedan 2012. The photo is an Audi S5 Coupe."})
    # Act 2 - honest about the year: S4 2012 photo where 2007 and 2012 compete
    a, b = names.index("Audi S4 Sedan 2012"), names.index("Audi S4 Sedan 2007")
    i = best_of("Audi S4 Sedan 2012", lambda idx, c: np.minimum(p[idx, a], p[idx, b]))
    save_full(test[i]["path"], ddir / "2_year_s4.jpg")
    items.append({"file": "2_year_s4.jpg", "title": "Which year?", "claim": "Audi S4 Sedan 2012",
                  "note": "Registered as a 2012 S4. The 2007 and 2012 S4 look almost identical."})
    # Act 3 - outside the dataset (first probe photo, if provided)
    probe = next((d for d in (REPO_ROOT / "probe", DATA_DIR / "probe_modern") if d.exists()), None)
    if probe:
        files = sorted(x for x in probe.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        if files:
            save_full(files[0], ddir / "3_modern.jpg")
            items.append({"file": "3_modern.jpg", "title": "Modern car", "claim": None,
                          "note": "A car newer than anything in Stanford Cars (dataset ends at 2012)."})
    # Extra: clean match, dark photo, honest low-confidence case
    i = best_of("BMW M3 Coupe 2012", lambda idx, c: p[idx, c])
    save_full(test[i]["path"], ddir / "4_match_m3.jpg")
    items.append({"file": "4_match_m3.jpg", "title": "Clean match", "claim": "BMW M3 Coupe 2012",
                  "note": "Registration and photo agree."})
    save_full(test[i]["path"], ddir / "5_dark_m3.jpg", transform=lambda im: (im * 0.3).astype(np.uint8))
    items.append({"file": "5_dark_m3.jpg", "title": "Night shot", "claim": "BMW M3 Coupe 2012",
                  "note": "Same car, simulated night-time exposure (x0.3)."})
    conf = p.max(1)
    low = np.flatnonzero((conf < 0.45) & (p.argmax(1) == y))
    if len(low):
        i = int(low[np.argmax(conf[low])])
        save_full(test[i]["path"], ddir / "6_low_conf.jpg")
        items.append({"file": "6_low_conf.jpg", "title": "Hard case", "claim": None,
                      "note": f"True class: {names[y[i]]}. Watch the top-3 instead of a single guess."})
    (ddir / "demo.json").write_text(json.dumps(items, indent=1))
    print("demo gallery:", [x["file"] for x in items])


if __name__ == "__main__":
    if "--demos-only" not in sys.argv:
        references()
    demos()
