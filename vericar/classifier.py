"""Stage 4 - fine-grained CNN classifier (ConvNeXt-Tiny, transfer learning).

Training uses a two-stage schedule (head only -> top blocks only, discriminative
LRs) driven by ``scripts/train_classifier.py``. At inference time the app uses
the ONNX export (see optimization.py) and falls back to PyTorch; Grad-CAM always
uses PyTorch because it needs gradients.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .config import CLS_IMG_SIZE, CROP_MARGIN, IMAGENET_MEAN, IMAGENET_STD
from .preprocessing import crop_box, normalize_for_model

import threading

MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"
_CAM_LOCK = threading.Lock()
TOP_STAGES = ("stages.2", "stages.3", "head")   # unfrozen in stage 2 (head = final norm)


def build_model(n_classes: int = 196, drop: float = 0.4, pretrained: bool = True):
    import timm
    import torch.nn as nn

    class CarClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0,
                                              drop_path_rate=0.1)
            self.dropout = nn.Dropout(drop)
            self.fc = nn.Linear(self.backbone.num_features, n_classes)

        def forward(self, x):
            feats = self.backbone(x)
            return self.fc(self.dropout(feats)), feats

    return CarClassifier()


def set_trainable(model, stage: str) -> None:
    """'head': only the linear head trains. 'top': head + last two ConvNeXt stages."""
    for p in model.backbone.parameters():
        p.requires_grad = False
    if stage == "top":
        for name in TOP_STAGES:
            for p in model.backbone.get_submodule(name).parameters():
                p.requires_grad = True


def param_groups(model, lr_head: float, lr_top: float) -> list[dict]:
    """Discriminative learning rates: head > last stage > second-to-last stage."""
    g = [{"params": list(model.fc.parameters()), "lr": lr_head}]
    last = list(model.backbone.get_submodule("stages.3").parameters()) + \
        list(model.backbone.get_submodule("head").parameters())
    g.append({"params": [p for p in last if p.requires_grad], "lr": lr_top})
    g.append({"params": [p for p in model.backbone.get_submodule("stages.2").parameters() if p.requires_grad],
              "lr": lr_top * 0.4})
    return [x for x in g if x["params"]]


# ------------------------------------------------------------------ data pipeline
class CarCrops:
    """Torch dataset of car crops returning uint8 CHW tensors (cheap CPU work only).

    CPU side (workers): decode, crop (with box-margin / position jitter when training),
    horizontal flip, resize - exactly the OpenCV resize the app uses at inference.
    GPU side (``gpu_augment``): mild colour jitter, normalisation, small random erasing.
    Augmentations keep make/model/year intact. 10% of training samples use the full
    frame so the model tolerates a bad / missing crop.
    """

    def __init__(self, records: list[dict], train: bool):
        self.records, self.train = records, train

    def __len__(self):
        return len(self.records)

    def _crop(self, img: np.ndarray, box) -> np.ndarray:
        if not self.train:
            return crop_box(img, box, CROP_MARGIN)
        if random.random() < 0.10:
            return img
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        jit = lambda s: random.uniform(-0.05, 0.05) * s  # noqa: E731
        m = random.uniform(0.0, 0.18)
        return crop_box(img, (x1 + jit(w), y1 + jit(h), x2 + jit(w), y2 + jit(h)), m)

    def __getitem__(self, i):
        import cv2
        import torch
        r = self.records[i]
        img = cv2.cvtColor(cv2.imread(r["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        crop = self._crop(img, r["box"])
        if self.train and random.random() < 0.5:
            crop = crop[:, ::-1]
        interp = cv2.INTER_AREA if crop.shape[0] > CLS_IMG_SIZE else cv2.INTER_LINEAR
        crop = cv2.resize(np.ascontiguousarray(crop), (CLS_IMG_SIZE, CLS_IMG_SIZE), interpolation=interp)
        return torch.from_numpy(crop).permute(2, 0, 1).contiguous(), r["label"]


def gpu_augment(x, train: bool):
    """uint8 [B,3,H,W] on GPU -> normalised float. Training adds colour jitter + random erasing."""
    import torch
    x = x.float() / 255.0
    if train:
        b = x.shape[0]
        r = lambda lo, hi: torch.empty(b, 1, 1, 1, device=x.device).uniform_(lo, hi)  # noqa: E731
        x = x * r(0.7, 1.3)                                                    # brightness
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - m) * r(0.7, 1.3) + m                                          # contrast
        g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        x = (g + (x - g) * r(0.75, 1.25)).clamp_(0, 1)                         # saturation
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    if train:                                                                  # random erasing, p=0.25
        b, _, h, w = x.shape
        area = torch.empty(b, device=x.device).uniform_(0.02, 0.12) * h * w
        ar = torch.exp(torch.empty(b, device=x.device).uniform_(-1.0, 1.0))
        eh = (area * ar).sqrt().clamp(max=h).long()
        ew = (area / ar).sqrt().clamp(max=w).long()
        y0 = (torch.rand(b, device=x.device) * (h - eh + 1)).long()
        x0 = (torch.rand(b, device=x.device) * (w - ew + 1)).long()
        ys = torch.arange(h, device=x.device).view(1, h, 1)
        xs = torch.arange(w, device=x.device).view(1, 1, w)
        mask = (ys >= y0.view(-1, 1, 1)) & (ys < (y0 + eh).view(-1, 1, 1)) & \
               (xs >= x0.view(-1, 1, 1)) & (xs < (x0 + ew).view(-1, 1, 1))
        mask &= (torch.rand(b, device=x.device) < 0.25).view(-1, 1, 1)
        x = x.masked_fill(mask.unsqueeze(1), 0.0)
    return x


# ------------------------------------------------------------------ inference
def preprocess(crop: np.ndarray) -> np.ndarray:
    return normalize_for_model(crop, CLS_IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD)


class Classifier:
    """Unified inference wrapper: ONNX Runtime if available, else PyTorch.

    ``predict(batch)`` -> (logits [N,196], features [N,768]) as float32 numpy.
    """

    def __init__(self, onnx_path: Path | None = None, torch_path: Path | None = None):
        self.onnx = self.torch_model = None
        if onnx_path is not None and Path(onnx_path).exists():
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.enable_cpu_mem_arena = False          # lower resident memory on small hosts
            self.onnx = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
            self.input_name = self.onnx.get_inputs()[0].name
        self.torch_path = torch_path
        if self.onnx is None:
            self._load_torch()

    def _load_torch(self):
        if self.torch_model is None:
            import torch
            ckpt = torch.load(self.torch_path, map_location="cpu", weights_only=False)
            m = build_model(len(ckpt["class_names"]), pretrained=False)
            m.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
            self.torch_model = m.eval()
        return self.torch_model

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.onnx is not None:
            logits, feats = self.onnx.run(None, {self.input_name: x.astype(np.float32)})
            return logits.astype(np.float32), feats.astype(np.float32)
        import torch
        with torch.inference_mode():
            logits, feats = self._load_torch()(torch.from_numpy(x))
        return logits.numpy(), feats.numpy()

    def gradcam(self, x: np.ndarray, class_idx: int) -> np.ndarray:
        """Grad-CAM heat-map (H x W in [0,1]) from the last ConvNeXt stage (one at a time: ~0.5 GB peak)."""
        with _CAM_LOCK:
            return self._gradcam(x, class_idx)

    def _gradcam(self, x: np.ndarray, class_idx: int) -> np.ndarray:
        import gc

        import torch
        import torch.nn.functional as F
        torch.set_num_threads(2)
        model = self._load_torch()
        store = {}

        def fwd_hook(_m, _i, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        h = model.backbone.stages[-1].register_forward_hook(fwd_hook)
        try:
            xt = torch.from_numpy(x).requires_grad_(True)
            logits, _ = model(xt)
            model.zero_grad()
            logits[0, class_idx].backward()
        finally:
            h.remove()
        act, grad = store["act"][0], store["grad"][0]
        cam = F.relu((grad.mean(dim=(1, 2), keepdim=True) * act).sum(0))
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().numpy()
        del store, logits, xt
        gc.collect()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
