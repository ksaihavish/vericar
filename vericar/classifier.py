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

MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"
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
    """Torch dataset of car crops. ``train=True`` adds moderate augmentation.

    Augmentations keep make/model/year intact: horizontal flip, box-margin and
    scale jitter, mild colour jitter, small random erasing. 10% of training
    samples use the full frame so the model tolerates a bad / missing crop.
    """

    def __init__(self, records: list[dict], train: bool):
        import torch
        from torchvision.transforms import v2 as T

        self.records, self.train = records, train
        norm = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        if train:
            self.tf = T.Compose([
                T.Resize((CLS_IMG_SIZE, CLS_IMG_SIZE), antialias=True),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.3, 0.3, 0.25, 0.02),
                *norm,
                T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
            ])
        else:
            self.tf = T.Compose([T.Resize((CLS_IMG_SIZE, CLS_IMG_SIZE), antialias=True), *norm])

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
        from PIL import Image
        r = self.records[i]
        img = np.asarray(Image.open(r["path"]).convert("RGB"))
        return self.tf(Image.fromarray(self._crop(img, r["box"]))), r["label"]


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
            return logits, feats
        import torch
        with torch.inference_mode():
            logits, feats = self._load_torch()(torch.from_numpy(x))
        return logits.numpy(), feats.numpy()

    def gradcam(self, x: np.ndarray, class_idx: int) -> np.ndarray:
        """Grad-CAM heat-map (H x W in [0,1]) from the last ConvNeXt stage."""
        import torch
        import torch.nn.functional as F
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
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
