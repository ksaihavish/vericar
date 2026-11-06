"""Stage 3 - vehicle detection (YOLO11, transfer-learned on Stanford Cars boxes).

The fine-tuned model has a single class ("car"). The COCO-pretrained model is
kept for the detector comparison (classes car / bus / truck count as vehicle).
If YOLO finds nothing, the hybrid pipeline falls back to the classical ROI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

COCO_VEHICLES = {2, 5, 7}  # car, bus, truck


class Detector:
    def __init__(self, weights: str | Path, coco: bool = False, imgsz: int = 640):
        from ultralytics import YOLO
        self.model = YOLO(str(weights), task="detect")
        self.coco, self.imgsz = coco, imgsz

    def detect(self, img: np.ndarray, conf: float = 0.25) -> list[tuple[tuple, float]]:
        """All vehicle boxes [(x1, y1, x2, y2), score] sorted by score (img is RGB)."""
        res = self.model.predict(img[..., ::-1], imgsz=self.imgsz, conf=conf, verbose=False, device="cpu"
                                 if not _cuda() else 0)[0]
        out = []
        for b, s, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            if self.coco and int(c) not in COCO_VEHICLES:
                continue
            out.append((tuple(float(v) for v in b), float(s)))
        return sorted(out, key=lambda d: -d[1])

    def detect_batch(self, paths: list[str], conf: float = 0.001) -> list[list[tuple[tuple, float]]]:
        """Batched detection from file paths (used for mAP evaluation)."""
        results = self.model.predict(paths, imgsz=self.imgsz, conf=conf, verbose=False, stream=True,
                                     device=0 if _cuda() else "cpu", max_det=20)
        out = []
        for res in results:
            dets = []
            for b, s, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                if self.coco and int(c) not in COCO_VEHICLES:
                    continue
                dets.append((tuple(float(v) for v in b), float(s)))
            out.append(sorted(dets, key=lambda d: -d[1]))
        return out


def _cuda() -> bool:
    import os
    if os.environ.get("VERICAR_CPU") == "1":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # pragma: no cover
        return False
