# VeriCar: fine-grained vehicle recognition and plate-clone verification

**Computer Vision Hackathon 2026, Problem Statement 5 (P5)**, Stanford Cars, 196 make/model/year classes.

VeriCar is built for a parking or fleet officer who needs to answer one question: *"Does this car match what its registration says?"*
Cloned plates are usually put on a car that looks like the registered one, so the system:

- identifies the car with a hybrid classical + deep pipeline (YOLO detection → ConvNeXt classifier, with classical preprocessing, quality checks and features fused in),
- **verifies a claimed registration level by level** (make → model → year) and returns **MATCH, PARTIAL, MISMATCH, INSPECT, UNKNOWN or RETAKE**,
- is tuned **strict**: clearing a cloned car costs more than a manual check, so thresholds are set for a high clone-catch rate, and the resulting false-alarm rate is reported,
- shows **why**: competing look-alike classes side by side, Grad-CAM, and every classical intermediate step,
- says **"not in my knowledge"** for cars outside the 196 training classes, then gives an educated guess at make and body type.

## Results (official test set, 8,041 images, evaluated once)

| | Top-1 | Top-5 | Make acc. | ECE |
|---|---|---|---|---|
| Classical (contour ROI + HOG/LBP/GLCM/colour/geometry + SVM) | 5.7% | 15.3% | 14.9% | 0.004 |
| Deep (YOLO11n crop → ConvNeXt-Tiny) | 93.4% | 99.2% | 97.8% | 0.145 |
| **Hybrid (ours)** | **93.4%** | **99.2%** | **97.8%** | **0.012** |

| Detector | mAP@0.5 | mAP@0.5:0.95 | Mean IoU |
|---|---|---|---|
| **YOLO11n fine-tuned (ours)** | **0.990** | **0.933** | **0.953** |
| YOLO11n COCO-pretrained | 0.855 | 0.782 | 0.871 |
| Classical edges + contours | 0.640 | 0.226 | 0.654 |

**Claim verification (strict):** 98.3% of hard clones (the most similar other model) and 100% of easy clones are caught; the cost is 17.1% of legitimate cars sent to a manual check.

**Honest notes**
- The hybrid's top-1 equals deep-only. Its gains are a **12× better calibration** (ECE 0.145 → 0.012), which the strict verifier depends on (without it, false alarms rise to 42.6%), plus the quality gate and the unknown-car check.
- The validation-fitted classical fusion weight came out **0**, so classical features don't change the hybrid's label. Classical CV contributes the ROI fallback, the quality gate and the adaptive enhancement. Enhancement *hurts* on very dark photos (90.4% vs 93.5%), and that is reported in the robustness table.
- **Cars newer than 2012** (outside Stanford Cars) are caught by two checks: kNN distance to the training embeddings, plus a zero-shot **OpenCLIP ViT-B/32** open-set fallback that was added after the test run. On 25 modern-car photos, 64% are flagged by kNN alone and **92% by both**. CLIP's suggested names are unverified leads only (top-3 correct 52%).
- The classical-only pipeline falls to 5.7% on test because its own contour ROI is imprecise (21.3% on clean ground-truth crops in validation).

Full tables (robustness, ablation, optimisation, confidence, failure cases) are in [`docs/eval_P5_<TeamID>.md`](docs/) and the app's Results tab. Design: [`docs/design.md`](docs/design.md) · architecture: [`docs/architecture.md`](docs/architecture.md) · viva prep: [`docs/viva_prep.md`](docs/viva_prep.md).

## Run it (one command)

Windows: double-click **`start.bat`**, or:

```bash
powershell -ExecutionPolicy Bypass -File run.ps1
```

Linux / macOS:

```bash
sh run.sh
```

Docker:

```bash
docker build -t vericar . && docker run -p 7860:7860 vericar
```

The launcher creates a Python 3.12 virtual environment, installs `requirements.txt`, downloads the trained models from this repo's GitHub release (SHA-256 verified) and starts the Streamlit app.
`GEMINI_API_KEY` in a `.env` file is **optional** and only enables the officer-note text and the AI second opinion for unknown cars.

## Reproduce training and evaluation

All steps in order with logging (`VERICAR_WORK/runs/*.log`):

```bash
python scripts/run_all.py
```

Or one by one:

```bash
pip install -r requirements-train.txt          # plus a CUDA build of torch
python scripts/prepare_data.py                 # download, verify test labels, cache, 85/15 split (seed 42)
python scripts/train_detector.py               # YOLO11n fine-tune (train/val only)
python scripts/train_classifier.py             # ConvNeXt-Tiny, staged fine-tune with overfitting guards
python scripts/train_classical.py              # handcrafted features + SVM
python scripts/fit_hybrid.py                   # temperature, fusion weight, thresholds (validation only)
python scripts/optimize.py                     # ONNX FP32 / FP16 / INT8
python scripts/evaluate_final.py               # the ONE run on the official test set
```

## Repository layout

| Path | Stage |
|---|---|
| `vericar/preprocessing.py` | normalisation, quality checks, adaptive CLAHE/gamma |
| `vericar/classical.py` | edges, contours, classical ROI, colour/texture/geometric features |
| `vericar/detector.py` | YOLO11 vehicle detector |
| `vericar/classifier.py` | ConvNeXt-Tiny fine-grained classifier, Grad-CAM |
| `vericar/hybrid.py` | fusion, calibration, unknown-car check, claim verification |
| `vericar/evaluation.py` | Top-k, IoU, mAP, ECE, coverage, AUROC |
| `vericar/optimization.py` | ONNX export, FP16, INT8, CPU benchmark |
| `vericar/pipeline.py` | `VeriCar().analyze(image, claim=None)`: the single entry point |
| `app/streamlit_app.py` | the UI (a thin layer over `analyze`) |
