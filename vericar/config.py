"""Central configuration: paths, seeds and fixed hyper-parameters.

Heavy artefacts (dataset cache, training runs) live OUTSIDE the repo in
VERICAR_WORK so they never get committed or synced. Deployable model files
live in ``models/`` and are downloaded by ``scripts/download_models.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = Path(os.environ.get("VERICAR_WORK", Path.home() / "vericar_work"))
DATA_DIR = WORK_DIR / "data"          # resized image cache + split files
RUNS_DIR = WORK_DIR / "runs"          # training logs / checkpoints
FEAT_DIR = WORK_DIR / "features"      # cached classical features / embeddings
MODELS_DIR = Path(os.environ.get("VERICAR_MODELS", REPO_ROOT / "models"))
RESULTS_DIR = REPO_ROOT / "results"
ASSETS_DIR = REPO_ROOT / "assets"

SEED = 42
VAL_FRACTION = 0.15                   # stratified hold-out from the official train set

CACHE_MAX_SIDE = 640                  # images are cached with this longest side
CLS_IMG_SIZE = 384                    # classifier input (square)
CROP_MARGIN = 0.08                    # context margin around a box at inference
CLASSICAL_SIZE = (128, 64)            # (w, h) ROI size for classical features

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

for _d in (WORK_DIR, DATA_DIR, RUNS_DIR, FEAT_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
