"""Fine-tune YOLO11n (COCO-pretrained) as a single-class car detector on Stanford Cars boxes.

Uses the same train/val split as the classifier; the test split is NOT used here.
Usage:  python scripts/train_detector.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vericar.config import DATA_DIR, MODELS_DIR, RUNS_DIR, SEED  # noqa: E402
from vericar.data import load_split  # noqa: E402


def build_yolo_dataset() -> Path:
    root = DATA_DIR / "yolo"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for r in load_split(split):
            src = Path(r["path"])
            dst = root / "images" / split / src.name
            if not dst.exists():
                try:
                    os.link(src, dst)          # hard link: no extra disk space
                except OSError:
                    shutil.copy2(src, dst)
            x1, y1, x2, y2 = r["box"]
            w, h = r["width"], r["height"]
            (root / "labels" / split / (src.stem + ".txt")).write_text(
                f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
    yaml = root / "cars.yaml"
    yaml.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n")
    return yaml


def main():
    from ultralytics import YOLO
    yaml = build_yolo_dataset()
    model = YOLO("yolo11n.pt")
    # one large car per image, single class: converges quickly; 12 epochs keeps the laptop budget sane
    model.train(data=str(yaml), epochs=12, patience=4, imgsz=640, batch=32, seed=SEED, deterministic=False,
                project=str(RUNS_DIR), name="detector", exist_ok=True, workers=6, cos_lr=True,
                close_mosaic=3, plots=True, verbose=False)
    best = RUNS_DIR / "detector" / "weights" / "best.pt"
    shutil.copy2(best, MODELS_DIR / "detector.pt")
    print("saved", MODELS_DIR / "detector.pt", flush=True)


if __name__ == "__main__":
    main()
