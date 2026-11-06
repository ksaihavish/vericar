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
        """CLIP's own recipe: shortest side -> 224 (bicubic), centre crop 224, CLIP normalisation.
        (Letterboxing the whole car was tried first: 62% vs 82% zero-shot top-1 on validation.)"""
        h, w = crop.shape[:2]
        s = 224 / min(h, w)
        img = cv2.resize(crop, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_CUBIC)
        y0, x0 = (img.shape[0] - 224) // 2, (img.shape[1] - 224) // 2
        x = img[y0:y0 + 224, x0:x0 + 224].astype(np.float32) / 255.0
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
