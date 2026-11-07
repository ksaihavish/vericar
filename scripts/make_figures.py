"""Report figures (results/figures/): intermediate stages, verdict examples, robustness examples.

    python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vericar.config import ASSETS_DIR, RESULTS_DIR  # noqa: E402
from vericar.data import hierarchy  # noqa: E402
from vericar.pipeline import VeriCar  # noqa: E402

OUT = RESULTS_DIR / "figures"
BG, INK = "#F3F0E8", "#1B1814"


def panel_row(images, titles, path, h=3.2, suptitle=None):
    fig, axes = plt.subplots(1, len(images), figsize=(3.6 * len(images), h), facecolor=BG)
    for ax, im, t in zip(np.atleast_1d(axes), images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=9, color=INK)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eng = VeriCar()
    hier = hierarchy()
    demo = ASSETS_DIR / "demo"

    # 1. intermediate representations for one photo
    r = eng.analyze(demo / "2_year_s4.jpg", claim=hier.class_names.index("Audi S4 Sedan 2012"))
    im = r["images"]
    panel_row([im["annotated"], im["edges"], im["contours"], im["hog"], im["lbp"], im["gradcam"]],
              ["1 · detection (YOLO11n)", "2 · auto-Canny edges", "3 · contour, hull, classical ROI",
               "4 · HOG (shape)", "5 · LBP (texture)", "6 · class activation map"],
              OUT / "pipeline_stages.png", suptitle="Intermediate representations (claim: Audi S4 Sedan 2012)")

    # 2. verdict examples
    cases = [("1_clone_s5.jpg", "Audi S4 Sedan 2012"), ("2_year_s4.jpg", "Audi S4 Sedan 2012"),
             ("4_match_m3.jpg", "BMW M3 Coupe 2012"), ("7_modern_xuv700.jpg", None)]
    imgs, titles = [], []
    for f, claim in cases:
        r = eng.analyze(demo / f, claim=hier.class_names.index(claim) if claim else None)
        imgs.append(r["images"]["annotated"])
        titles.append(f"claim: {claim or 'none'}\n→ {r['verdict']['status']}: {r['verdict']['headline']}")
    panel_row(imgs, titles, OUT / "verdict_examples.png", h=3.6,
              suptitle="Verdicts: possible clone · year not separable · cleared · not in records")

    # 3. robustness: one car under the evaluated perturbations
    from common import PERTURBATIONS
    from vericar.preprocessing import load_rgb
    base = load_rgb(demo / "4_match_m3.jpg")
    rng = np.random.default_rng(0)
    h, w = base.shape[:2]
    box = (w * 0.05, h * 0.25, w * 0.95, h * 0.85)
    imgs, titles = [], []
    for name in ("clean", "dark (x0.35)", "occlusion 25%", "low-res 1/4", "blur sigma 2", "noise sigma 20"):
        x = PERTURBATIONS[name](base, box, rng)
        r = eng.analyze(x)
        imgs.append(r["images"]["annotated"])
        titles.append(f"{name}\n→ {r['verdict']['status']}: {r['prediction']['label']} ({r['confidence']:.0%})")
    panel_row(imgs, titles, OUT / "robustness_examples.png", h=3.4,
              suptitle="Robustness: the same car under the evaluated perturbations")
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
