"""Stage 3 - vehicle detection (YOLO11n, transfer-learned on Stanford Cars boxes).

Two back-ends with the same interface:
* ``.onnx`` - used by the app: ONNX Runtime + our own letterbox / decode / NMS, so the deployed
  app does not need PyTorch or Ultralytics in memory (fits a free 1 GB host).
* ``.pt``   - Ultralytics, used by the training / evaluation scripts (and for the COCO baseline).
The fine-tuned model has one class ("car"). If nothing is found, the hybrid pipeline falls back
to the classical ROI.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

COCO_VEHICLES = {2, 5, 7}  # car, bus, truck


class Detector:
    def __init__(self, weights: str | Path, coco: bool = False, imgsz: int = 640):
        self.coco, self.imgsz, self.weights = coco, imgsz, Path(weights)
        if self.weights.suffix == ".onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.enable_cpu_mem_arena = False
            so.intra_op_num_threads = 2
            self.sess = ort.InferenceSession(str(self.weights), so, providers=["CPUExecutionProvider"])
            self.model = None
        else:
            from ultralytics import YOLO
            self.model, self.sess = YOLO(str(weights), task="detect"), None

    # ---------------------------------------------------------------- ONNX path
    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = round(h * r), round(w * r)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, left, top

    def _detect_onnx(self, img: np.ndarray, conf: float, iou: float = 0.45, max_det: int = 20):
        canvas, r, left, top = self._letterbox(img)
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0          # RGB, [0, 1]
        pred = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0][0]    # (4 + n_classes, N)
        scores = pred[4:].max(0)
        keep = scores >= conf
        if not keep.any():
            return []
        cx, cy, bw, bh = pred[0, keep], pred[1, keep], pred[2, keep], pred[3, keep]
        s = scores[keep]
        x1, y1 = (cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r
        x2, y2 = (cx + bw / 2 - left) / r, (cy + bh / 2 - top) / r
        H, W = img.shape[:2]
        x1, x2 = np.clip(x1, 0, W), np.clip(x2, 0, W)
        y1, y2 = np.clip(y1, 0, H), np.clip(y2, 0, H)
        idx = cv2.dnn.NMSBoxes(np.stack([x1, y1, x2 - x1, y2 - y1], 1).tolist(), s.tolist(), conf, iou)
        idx = np.array(idx).reshape(-1)[:max_det]
        out = [((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])), float(s[i])) for i in idx]
        return sorted(out, key=lambda d: -d[1])

    # ---------------------------------------------------------------- public
    def detect(self, img: np.ndarray, conf: float = 0.25) -> list[tuple[tuple, float]]:
        """All vehicle boxes [(x1, y1, x2, y2), score] sorted by score (img is RGB)."""
        if self.sess is not None:
            return self._detect_onnx(img, conf)
        res = self.model.predict(img[..., ::-1], imgsz=self.imgsz, conf=conf, verbose=False,
                                 device=0 if _cuda() else "cpu")[0]
        out = []
        for b, s, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            if self.coco and int(c) not in COCO_VEHICLES:
                continue
            out.append((tuple(float(v) for v in b), float(s)))
        return sorted(out, key=lambda d: -d[1])


def _cuda() -> bool:
    import os
    if os.environ.get("VERICAR_CPU") == "1":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # pragma: no cover
        return False
