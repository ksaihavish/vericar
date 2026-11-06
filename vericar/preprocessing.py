"""Stage 1 - classical preprocessing, normalisation and image-quality checks.

Everything here is plain OpenCV/NumPy so it runs fast on CPU. The quality
report drives the hybrid "RETAKE" state and the adaptive enhancement decides
whether a dark / flat image gets CLAHE + gamma before the CNN sees it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

# Thresholds come from train-split statistics (1st percentiles: brightness 58,
# contrast 41, sharpness 147 - see data/summary.json) and sit well below them, so
# the gate only fires on clearly bad input, not on normal photos.
DARK_V = 35.0          # mean V (0-255) below this = too dark to verify (gate)
ENHANCE_V = 60.0       # below this the image is enhanced (gamma + CLAHE) before the CNN
FLAT_STD = 28.0        # std of L channel below this = low contrast -> CLAHE
BLUR_VAR = 35.0        # variance of Laplacian (512px wide gray) below this = blurry
MIN_ROI_SIDE = 48      # car crop smaller than this (px) is too small to classify


def load_rgb(src) -> np.ndarray:
    """Load path / bytes / PIL / ndarray into an RGB uint8 array (EXIF-rotated)."""
    if isinstance(src, np.ndarray):
        arr = src
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return np.ascontiguousarray(arr.astype(np.uint8))
    img = src if isinstance(src, Image.Image) else Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.asarray(img, dtype=np.uint8).copy()


def resize_max(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is <= max_side. Returns (image, scale)."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


@dataclass
class QualityReport:
    brightness: float
    contrast: float
    sharpness: float
    min_side: int
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict:
        return {"brightness": round(self.brightness, 1), "contrast": round(self.contrast, 1),
                "sharpness": round(self.sharpness, 1), "min_side": self.min_side,
                "issues": list(self.issues)}


def quality_report(img: np.ndarray) -> QualityReport:
    """Brightness, contrast, sharpness and resolution of an RGB image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    g = cv2.resize(gray, (512, max(1, round(512 * h / w))), interpolation=cv2.INTER_AREA)
    rep = QualityReport(
        brightness=float(hsv[..., 2].mean()),
        contrast=float(lab[..., 0].std()),
        sharpness=float(cv2.Laplacian(g, cv2.CV_64F).var()),
        min_side=int(min(h, w)),
    )
    if rep.brightness < DARK_V:
        rep.issues.append("too dark")
    if rep.sharpness < BLUR_VAR:
        rep.issues.append("blurry")
    if rep.min_side < MIN_ROI_SIDE:
        rep.issues.append("too small")
    return rep


def gray_world(img: np.ndarray) -> np.ndarray:
    """Gray-world colour constancy: equalise channel means (removes colour casts)."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0) + 1e-6
    f *= means.mean() / means
    return np.clip(f, 0, 255).astype(np.uint8)


def clahe_lab(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the L channel only."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def gamma_correct(img: np.ndarray, target_mean: float = 110.0) -> np.ndarray:
    """Gamma curve that moves the mean intensity towards target_mean."""
    mean = max(img.mean(), 1.0)
    gamma = np.clip(np.log(target_mean / 255.0) / np.log(mean / 255.0), 0.4, 2.5)
    lut = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def adaptive_enhance(img: np.ndarray, rep: QualityReport | None = None) -> tuple[np.ndarray, list[str]]:
    """Enhance only when needed. Returns (image, list of applied steps).

    Well-exposed images pass through untouched so the CNN sees the same
    distribution it was trained on; dark / flat images get gamma + CLAHE.
    """
    rep = rep or quality_report(img)
    steps: list[str] = []
    out = img
    if rep.brightness < ENHANCE_V:
        out = gamma_correct(out)
        steps.append("gamma")
    if rep.contrast < FLAT_STD or steps:
        out = clahe_lab(out)
        steps.append("CLAHE")
    return out, steps


def crop_box(img: np.ndarray, box, margin: float = 0.0) -> np.ndarray:
    """Crop (x1, y1, x2, y2) with a relative context margin, clipped to the image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return img
    return img[y1:y2, x1:x2]


def normalize_for_model(img: np.ndarray, size: int, mean, std) -> np.ndarray:
    """Squash-resize to size x size, scale to [0,1], ImageNet-normalise -> NCHW float32."""
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR)
    x = (x.astype(np.float32) / 255.0 - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return x.transpose(2, 0, 1)[None]
