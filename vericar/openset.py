"""Open-set fallback: zero-shot OpenCLIP ViT-B/32 (LAION-2B) over an open vocabulary.

The trained ConvNeXt only knows 196 Stanford Cars classes (up to 2012). This independent,
pre-trained model compares the car crop with text descriptions of those 196 classes AND of
~300 post-2012 models. If most of the probability goes to the post-2012 vocabulary, the car
is flagged 'not in records' and the best post-2012 names are offered as an UNVERIFIED hint.
It never changes a verdict to MATCH; it can only send a car to manual check.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32).reshape(3, 1, 1)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32).reshape(3, 1, 1)


class OpenSet:
    def __init__(self, models_dir: Path):
        import onnxruntime as ort
        models_dir = Path(models_dir)
        self.meta = json.loads((models_dir / "openset.json").read_text())
        self.labels = self.meta["labels"]
        self.n_known = self.meta["n_known"]
        self.text = np.load(models_dir / "openset_text.npy").astype(np.float32)
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(models_dir / "openset_clip_int8.onnx"), so,
                                         providers=["CPUExecutionProvider"])

    @staticmethod
    def preprocess(crop: np.ndarray) -> np.ndarray:
        """Letterbox to a square (keeps the whole car, unlike centre-crop), 224 px, CLIP normalisation."""
        h, w = crop.shape[:2]
        s = max(h, w)
        canvas = np.full((s, s, 3), (123, 117, 104), np.uint8)
        canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
        x = cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        return ((x.transpose(2, 0, 1) - MEAN) / STD)[None]

    def raw(self, crop: np.ndarray) -> dict:
        emb = self.sess.run(None, {"image": self.preprocess(crop)})[0][0]
        z = self.meta["logit_scale"] * (self.text @ emb)
        p = np.exp(z - z.max())
        p /= p.sum()
        return {"p": p, "p_modern": float(p[self.n_known:].sum())}

    def __call__(self, crop: np.ndarray, k: int = 3) -> dict:
        r = self.raw(crop)
        pm = r["p"][self.n_known:]
        idx = np.argsort(-pm)[:k]
        top = [{"label": self.labels[self.n_known + i], "prob": float(pm[i] / max(r["p_modern"], 1e-9))}
               for i in idx]
        return {"p_modern": r["p_modern"], "flag": r["p_modern"] > self.meta["tau"], "top": top}
