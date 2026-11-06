# VeriCar: fine-grained vehicle recognition and plate-clone verification

**Computer Vision Hackathon 2026, Problem Statement 5 (P5)**, Stanford Cars, 196 make/model/year classes.

VeriCar is built for a parking or fleet officer who needs to answer one question: *"Does this car match what its registration says?"*
Cloned plates are usually put on a car that looks like the registered one, so the system:

- identifies the car with a hybrid classical + deep pipeline (YOLO detection → ConvNeXt classifier, with classical preprocessing, quality checks and features fused in),
- **verifies a claimed registration level by level** (make → model → year) and returns **MATCH, PARTIAL, MISMATCH, INSPECT, UNKNOWN or RETAKE**,
- is tuned **strict**: clearing a cloned car costs more than a manual check, so thresholds are set for a high clone-catch rate, and the resulting false-alarm rate is reported,
- shows **why**: competing look-alike classes side by side, Grad-CAM, and every classical intermediate step,
- says **"not in my knowledge"** for cars outside the 196 training classes, then gives an educated guess at make and body type.

> Status: under active development during the hackathon. Results, design doc and architecture diagram will land in `results/` and `docs/`.

## Run it (one command)

Windows:

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
