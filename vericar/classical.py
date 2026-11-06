"""Stage 2 - classical computer vision: edges, contours, ROI, colour, texture, geometry.

``extract_features`` turns a car crop into one fixed-length vector
(HOG + LBP + GLCM + HSV/Lab colour + geometric shape descriptors) that the
classical SVM is trained on. ``classical_roi`` localises the car without any
learning (edge map -> morphological closing -> largest contour).
``visualize`` returns the intermediate images shown in the app.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from .config import CLASSICAL_SIZE

HSV_BINS = (16, 8, 8)
LBP_SETTINGS = ((8, 1), (16, 2))
GRID = 4


# ---------------------------------------------------------------- edges / contours
def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with thresholds derived from the median intensity (no manual tuning)."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    v = float(np.median(g))
    lo, hi = int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v))
    return cv2.Canny(g, lo, max(hi, lo + 1))


def classical_roi(img: np.ndarray) -> dict:
    """Locate the dominant object using only edges + morphology.

    Returns the box (x1, y1, x2, y2), the chosen contour and the closed edge map.
    Falls back to the full frame when no meaningful contour exists.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = auto_canny(gray)
    k = max(3, int(0.03 * max(h, w)) | 1)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    closed = cv2.dilate(closed, None, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 0.01 * h * w]
    if not contours:
        return {"box": (0, 0, w, h), "contour": None, "edges": edges, "closed": closed}
    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    return {"box": (x, y, x + bw, y + bh), "contour": c, "edges": edges, "closed": closed}


def geometric_features(img: np.ndarray, roi: dict | None = None) -> np.ndarray:
    """Shape descriptors of the ROI contour + edge-density grid (19 + 16 values)."""
    h, w = img.shape[:2]
    roi = roi or classical_roi(img)
    x1, y1, x2, y2 = roi["box"]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    c = roi["contour"]
    if c is not None:
        area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c)) + 1e-6
        perim = cv2.arcLength(c, True) + 1e-6
        hu = cv2.HuMoments(cv2.moments(c)).ravel()
    else:
        area, hull_area, perim, hu = bw * bh, bw * bh, 2 * (bw + bh), np.zeros(7)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    hu = np.clip(hu, -30, 30) / 30.0
    base = np.array([
        bw / bh,                         # aspect ratio (side view vs front view)
        area / (bw * bh),                # extent
        area / hull_area,                # solidity
        4 * np.pi * area / perim ** 2,   # circularity
        (bw * bh) / (w * h),             # relative size
        (x1 + x2) / (2 * w), (y1 + y2) / (2 * h),  # centre
        bw / w, bh / h,
        perim / (2 * (bw + bh)),         # contour complexity
        len(cv2.approxPolyDP(c, 0.01 * perim, True)) / 100.0 if c is not None else 0.0,
        0.0,
    ], dtype=np.float32)
    # edge density on a GRID x GRID layout of the ROI (where are the lines: grille, windows...)
    e = roi["edges"][y1:y2, x1:x2]
    e = cv2.resize(e, (GRID * 16, GRID * 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    dens = e.reshape(GRID, 16, GRID, 16).mean(axis=(1, 3)).ravel()
    return np.concatenate([base, hu.astype(np.float32), dens.astype(np.float32)])


# ---------------------------------------------------------------- colour / texture
def colour_features(crop: np.ndarray) -> np.ndarray:
    """HSV histogram (16+8+8) + Lab mean/std (6)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    feats = []
    for ch, bins, rng in zip(range(3), HSV_BINS, (180, 256, 256)):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0, rng]).ravel()
        feats.append(hist / (hist.sum() + 1e-6))
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32) / 255.0
    feats.append(np.concatenate([lab.mean(0), lab.std(0)]))
    return np.concatenate(feats).astype(np.float32)


def texture_features(gray: np.ndarray) -> np.ndarray:
    """Multi-scale uniform LBP histograms + GLCM statistics."""
    feats = []
    for p, r in LBP_SETTINGS:
        lbp = local_binary_pattern(gray, p, r, method="uniform")
        hist, _ = np.histogram(lbp, bins=p + 2, range=(0, p + 2))
        feats.append(hist / (hist.sum() + 1e-6))
    q = (gray // 32).astype(np.uint8)  # 8 grey levels keeps GLCM cheap
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=8, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        v = graycoprops(glcm, prop).ravel()
        feats.append(np.nan_to_num(v / (10.0 if prop == "contrast" else 1.0)))
    return np.concatenate(feats).astype(np.float32)


def hog_features(gray: np.ndarray) -> np.ndarray:
    return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def extract_features(crop: np.ndarray) -> np.ndarray:
    """Full classical descriptor for one car crop (RGB uint8)."""
    crop = cv2.resize(crop, CLASSICAL_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)  # illumination normalisation
    return np.concatenate([
        hog_features(gray),
        texture_features(gray),
        colour_features(crop),
        geometric_features(crop),
    ])


FEATURE_GROUPS = None  # filled lazily by feature_group_slices()


def feature_group_slices() -> dict[str, slice]:
    """Index ranges of each feature family inside the vector (for ablations)."""
    global FEATURE_GROUPS
    if FEATURE_GROUPS is None:
        dummy = np.full((CLASSICAL_SIZE[1], CLASSICAL_SIZE[0], 3), 127, np.uint8)
        dummy[10:50, 20:100] = 200
        gray = cv2.cvtColor(dummy, cv2.COLOR_RGB2GRAY)
        n = [len(hog_features(gray)), len(texture_features(gray)), len(colour_features(dummy)),
             len(geometric_features(dummy))]
        names, start, out = ["hog", "texture", "colour", "geometry"], 0, {}
        for name, k in zip(names, n):
            out[name] = slice(start, start + k)
            start += k
        FEATURE_GROUPS = out
    return FEATURE_GROUPS


# ---------------------------------------------------------------- visualisation
def visualize(img: np.ndarray, box=None) -> dict[str, np.ndarray]:
    """Intermediate images for the app's pipeline tab (all RGB uint8)."""
    roi = classical_roi(img)
    edges_rgb = cv2.cvtColor(roi["edges"], cv2.COLOR_GRAY2RGB)
    cont = img.copy()
    if roi["contour"] is not None:
        cv2.drawContours(cont, [roi["contour"]], -1, (0, 220, 255), 2)
        hull = cv2.convexHull(roi["contour"])
        cv2.drawContours(cont, [hull], -1, (255, 160, 0), 1)
    x1, y1, x2, y2 = roi["box"]
    cv2.rectangle(cont, (x1, y1), (x2, y2), (255, 60, 60), 2)
    crop_box = box if box is not None else roi["box"]
    bx1, by1, bx2, by2 = map(int, crop_box)
    crop = img[by1:by2, bx1:bx2] if (bx2 - bx1 > 4 and by2 - by1 > 4) else img
    small = cv2.resize(crop, CLASSICAL_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    _, hog_img = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_img = cv2.normalize(hog_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return {
        "edges": edges_rgb,
        "contours": cont,
        "classical_box": np.array(roi["box"]),
        "hog": cv2.applyColorMap(hog_img, cv2.COLORMAP_INFERNO)[..., ::-1],
        "lbp": cv2.cvtColor(lbp, cv2.COLOR_GRAY2RGB),
        "crop": crop,
    }


def hsv_histograms(crop: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    return {name: cv2.calcHist([hsv], [i], None, [b], [0, r]).ravel()
            for i, (name, b, r) in enumerate(zip(("hue", "saturation", "value"), (36, 32, 32), (180, 256, 256)))}
