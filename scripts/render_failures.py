"""Render the failure-case panels chosen by evaluate_final.py into results/failures/.

Each panel: full image with ground-truth box (green) and our detection (red) | the crop
the classifier saw | Grad-CAM for the (wrong) prediction | reference image of the
predicted class | reference image of the true class. The written explanations live in
results/failures.json (existing explanations are kept when re-rendering).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vericar.config import ASSETS_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402
from vericar.data import hierarchy, load_split  # noqa: E402
from vericar.pipeline import VeriCar  # noqa: E402
from vericar.preprocessing import load_rgb  # noqa: E402

TITLES = {"year_confusion": "Same model, wrong year", "confident_wrong_make": "Confident, wrong make",
          "bad_localisation": "Detector boxed the wrong region", "dark_image": "Low light",
          "truncated_or_occluded": "Car cut off by the frame"}


def main():
    F = json.loads((RESULTS_DIR / "final_test.json").read_text())
    hier = hierarchy()
    names = list(hier.class_names)
    test = load_split("test")
    eng = VeriCar(MODELS_DIR)
    out_dir = RESULTS_DIR / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    old = {f["key"]: f for f in json.loads((RESULTS_DIR / "failures.json").read_text())} \
        if (RESULTS_DIR / "failures.json").exists() else {}
    items = []
    for key, c in F["failure_candidates"].items():
        rec = test[c["index"]]
        res = eng.analyze(rec["path"], gradcam=True)
        img = load_rgb(rec["path"]).copy()
        x1, y1, x2, y2 = map(int, rec["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 200, 40), 3)
        bx = list(map(int, res["box"]))
        cv2.rectangle(img, (bx[0], bx[1]), (bx[2], bx[3]), (230, 50, 50), 2)
        panels = [(img, "image: GT box (green), ours (red)"), (res["images"]["crop"], "what the classifier saw"),
                  (res["images"]["gradcam"], f"Grad-CAM for '{res['prediction']['label']}'")]
        for lab, cap in ((c["pred"], "reference: predicted"), (c["true"], "reference: true")):
            p = ASSETS_DIR / "ref" / f"{names.index(lab):03d}.jpg"
            if p.exists():
                panels.append((load_rgb(p), f"{cap}\n{lab}"))
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4), facecolor="#0A0F1C")
        for ax, (im, cap) in zip(axes, panels):
            ax.imshow(im)
            ax.set_title(cap, color="#E5E7EB", fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}.png", dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
        items.append({"key": key, "title": TITLES.get(key, key), "true": c["true"], "pred": c["pred"],
                      "conf": c["conf"], "det_iou": c["det_iou"], "top3": c["top3"], "image": f"{key}.png",
                      "quality": res["quality"], "explanation": old.get(key, {}).get("explanation", "TODO")})
    (RESULTS_DIR / "failures.json").write_text(json.dumps(items, indent=1))
    print("rendered", [i["key"] for i in items])


if __name__ == "__main__":
    main()
