"""Batch inference: every image in a folder -> annotated image + one JSON line per image.

    python scripts/run_inference.py --input_dir ./samples/ --output_dir ./results/inference/
    python scripts/run_inference.py --input_dir ./samples/ --output_dir ./out/ --claim "Audi S4 Sedan 2012"

Runs the same pipeline as the app (VeriCar.analyze): detection, classical + deep + hybrid
predictions, verdict, unknown-car checks. Models are downloaded on first use.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--claim", default=None, help="registered vehicle, exactly as one of the 196 class names")
    a = ap.parse_args()

    if runpy.run_path(str(ROOT / "scripts" / "download_models.py"))["main"]() != 0:
        return 1
    import cv2

    from vericar.pipeline import VeriCar
    eng = VeriCar()
    claim = None
    if a.claim:
        if a.claim not in eng.hier.class_names:
            print(f"Unknown class name: {a.claim!r}. See models/classes.json for the 196 valid names.")
            return 2
        claim = eng.hier.class_names.index(a.claim)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in a.input_dir.iterdir() if p.suffix.lower() in EXTS)
    with open(a.output_dir / "predictions.jsonl", "w", encoding="utf-8") as out:
        for p in files:
            try:
                r = eng.analyze(p, claim=claim)
            except Exception as e:  # unreadable or corrupt image: record and continue
                out.write(json.dumps({"file": p.name, "error": str(e)}) + "\n")
                print(f"{p.name}: error {e}")
                continue
            cv2.imwrite(str(a.output_dir / f"{p.stem}_annotated.jpg"), r["images"]["annotated"][..., ::-1])
            rec = {"file": p.name, "box": r["box"], "verdict": r["verdict"]["status"],
                   "headline": r["verdict"]["headline"], "make_model_year": r["prediction"]["label"],
                   "confidence": round(r["confidence"], 4), "unknown": r["unknown"],
                   "unknown_reasons": r["unknown_reasons"],
                   "methods_top1": {k: v[0]["label"] for k, v in r["methods"].items()},
                   "timings_ms": r["timings_ms"]}
            out.write(json.dumps(rec) + "\n")
            print(f"{p.name}: {rec['verdict']:14s} {rec['make_model_year']} ({rec['confidence']:.0%})")
    print(f"wrote {len(files)} results to {a.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
