# VeriCar: design document

**Problem statement:** P5, Intelligent Fine-Grained Vehicle Recognition and Analysis System (Computer Vision Hackathon 2026)
**Dataset:** Stanford Cars: 16,185 images, 196 make/model/year classes, one bounding box per image
**Live app:** see README · **Code:** this repository · **Results:** [`eval_P5_<TeamID>.md`](.) and the app's Results tab

---

## 1. Problem and user

**User.** A parking or fleet officer who has looked up a plate and now has a *claimed* vehicle (make, model, year) on screen.
**Decision.** "Does the car in front of me match its registration, or should I inspect it?"
**Why it matters.** Cloned plates are deliberately put on cars that *resemble* the registered one: same make, similar body. Telling them apart is exactly the fine-grained problem that P5 asks for.

**Cost asymmetry.** Clearing a cloned car (missed clone) is worse than sending a legitimate car to a two-minute manual check (false alarm). The system is therefore tuned **strict**. We report the catch rate on hard clones *and* the false-alarm rate we pay for it.

## 2. Requirements → where they are met

| PS5 requirement | Where |
|---|---|
| Preprocessing & normalisation | `preprocessing.py`: resize, gray-world, CLAHE, gamma, quality report, ImageNet normalisation |
| Edge detection & contour analysis | `classical.py`: auto-Canny, morphological closing, largest contour, convex hull |
| Colour & texture features | `classical.py`: HSV histograms, Lab moments, multi-scale LBP, GLCM, HOG |
| ROI extraction | classical contour ROI (`classical_roi`) and YOLO ROI (`detector.py`) |
| Geometric features | aspect ratio, extent, solidity, circularity, Hu moments, edge-density grid |
| YOLO / R-CNN detection | YOLO11n, COCO-pretrained, fine-tuned on Stanford Cars boxes |
| CNN/ViT fine-grained classifier, transfer learning, 196 classes | ConvNeXt-Tiny (ImageNet-22k → 1k), two-stage fine-tune |
| Hybrid integration, detected ROI → classifier | `pipeline.py` (YOLO crop → classifier), `hybrid.py` (fusion, gate, verification) |
| Boxes + labels + confidence | annotated image, verdict card, confidence gauge |
| Low-confidence handling | identify mode: threshold slider → top-3; verify mode: PARTIAL / INSPECT / UNKNOWN / RETAKE |
| Similar classes | make → model → year hierarchy, PARTIAL state, confusion analysis, hard-clone study |
| Illumination / angle / occlusion / resolution | adaptive enhancement + quality gate; robustness study (12 conditions); viewpoint proxy |
| Classical vs DL vs hybrid | three pipelines run on every image (Compare tab) and on the whole test set |
| Ablation | 5 hybrid components removed one at a time + classical feature families |
| Optimisation | ONNX FP32 / FP16 / INT8 dynamic / INT8 static, 2-thread CPU latency |
| Unseen test images, ≥ 3 failure cases | official test split used once; 5 analysed failures |
| Metrics: Top-1/5, IoU, mAP, size, time, robustness, confidence | `evaluation.py`, `scripts/evaluate_final.py` |

## 3. Architecture

See [`architecture.md`](architecture.md). The diagram names the module and function behind every box.
The single entry point is `VeriCar.analyze(image, claim=None)`. It returns the box, the make/model/year prediction and its confidence, the verdict, every intermediate image, and the per-method (classical / deep / hybrid) predictions. The Streamlit UI is a thin layer over it.

## 4. Data protocol (anti-leakage)

- The Kaggle copy has no test labels. The official `cars_test_annos_withlabels.mat` was obtained separately and **verified**: all 8,041 file names and boxes match the original test annotation file (`prepare_data.py` asserts it).
- The official train set is split **85 / 15, stratified by class, seed 42**: 6,922 train / 1,222 val.
- **Every** choice is made on validation: model selection, the classical SVM's C, early stopping, temperature, fusion weight, unknown-car threshold, verification thresholds, and the optimised variant to ship.
- The test set is read by exactly one script, `evaluate_final.py`. It refuses to run twice, and `results/test_run.json` records each run.

## 5. Methods

### 5.1 Classical pipeline
Classical ROI (auto-Canny with median-based thresholds → closing with a kernel at 3% of the image size → largest external contour) → crop → 128×64 → CLAHE → **3,913-d descriptor**: HOG (3,780), LBP 8/1 + 16/2 and GLCM (60), HSV histograms + Lab moments (38), geometry (35). Then StandardScaler → PCA-256 (whitened) → RBF-SVM with Platt probabilities. The SVM (C=3) beat a 500-tree random forest on validation (21.3% vs 9.3% top-1).

### 5.2 Detector
YOLO11n pretrained on COCO, fine-tuned as a single-class "car" detector (640 px, ≤ 25 epochs, early stopping patience 5, AdamW, cosine LR). If no box is found at confidence 0.25, the threshold drops to 0.05; if there is still nothing, the classical ROI is used and the verdict becomes RETAKE ("no vehicle detected").

### 5.3 Fine-grained classifier
ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`), 384×384 input, dropout 0.4 head, drop-path 0.1.
- **Stage 1** (3 epochs): backbone frozen, linear head only.
- **Stage 2** (≤ 18 epochs): last two ConvNeXt stages + head, **discriminative LRs** (head 5e-4, stage 3 1.2e-4, stage 2 4.8e-5), AdamW (wd 0.05), 1-epoch warm-up + cosine.
- **Regularisation:** label smoothing 0.1; moderate augmentation that never changes make/model/year (flip, box-margin/position jitter, 10% full-frame samples, brightness/contrast/saturation jitter, small random erasing).
- **Guards:** hard epoch cap; early stopping on val loss (patience 4); best checkpoint by val top-1; the training stops if *epoch train accuracy − val accuracy* exceeds 10 points. The stricter *clean-train* gap (non-augmented 1,222-image train subset) is logged every epoch and reported too.

### 5.4 Hybrid integration
1. YOLO ROI (classical fallback).
2. **Quality gate** (classical): brightness < 35, Laplacian variance < 35, or crop < 48 px → RETAKE. The thresholds sit well below the 1st percentile of the training images.
3. **Adaptive enhancement** (classical): gamma + CLAHE only when the crop is dark (V < 60) or flat (L-std < 28), so normal photos are untouched.
4. ConvNeXt logits → **temperature scaling** (T fitted on validation).
5. **Log-linear fusion** with the classical SVM probabilities (weight w fitted on validation NLL; w = 0 is allowed and would be reported honestly).
6. **Unknown-car check:** cosine kNN distance (k = 10) of the 768-d embedding to the training embeddings; threshold = 95th percentile on validation.
7. **Verification:** probabilities summed into make, model family and year. MATCH needs the claimed class to be the **top-1 prediction** *and* calibrated P(claimed class) ≥ τ, where τ is the lowest threshold with ≤ 2% hard clones cleared on validation. PARTIAL = claimed model family ≥ 0.80 but the year not verified. MISMATCH = P(claimed model family) below a threshold that keeps legitimate MISMATCHes ≤ 1% on validation. Anything else is INSPECT.

### 5.4b Open-set fallback (added after the main test run)
Stanford Cars ends in 2012, so most cars on today's roads are outside the classifier's 196 classes. After the frozen test evaluation, a **second, independent, pre-trained model** was added. It is used only to *refuse*, never to clear a claim: **OpenCLIP ViT-B/32 (LAION-2B), zero-shot, no training**. Its image encoder is exported to ONNX INT8 (`vericar/openset.py`, built by `scripts/build_openset.py`). It compares the car crop with text descriptions of the 196 known classes plus 343 post-2012 models (`vericar/openset_vocab.py`, including Indian-market cars). If P(post-2012 vocabulary) exceeds a threshold set on the **Stanford validation split** (97th percentile, so 3% of known cars are flagged by this check alone), the car is flagged NOT IN RECORDS. Its best post-2012 names are shown as an *unverified lead*. Post-hoc check on the 25 modern-car probe photos: the kNN check alone flagged 64%, **kNN + open-set flagged 92%**, and the top-1 name was right 16% of the time (top-3 52%), so names are never presented as an identification. Its effect on the frozen test-set numbers was not re-measured; on validation it adds at most 3 points of extra manual checks for known cars. Full numbers: `results/openset_eval.json`.

### 5.5 Explainability
Grad-CAM from the last ConvNeXt stage; look-alike candidates shown next to the most typical training image of each class (nearest to the class prototype in embedding space); every classical intermediate (edges, contours, ROI, HOG, LBP, colour histograms); and an optional Gemini officer note generated **only** from the structured verdict.

### 5.6 Optimisation
PyTorch → ONNX FP32 → FP16 → INT8 dynamic → INT8 static (QDQ, per-channel, calibrated on 128 *training* crops). The app ships the fastest variant whose validation top-1 is within 0.5 points of ONNX FP32.

## 6. Evaluation protocol

| Metric (PS5 order) | How |
|---|---|
| Top-1 / Top-5 | 196-way, test set; also per level: make, model family, year given the right model |
| IoU | best-scoring box vs the ground-truth box, per image |
| mAP | single-class AP, 101-point interpolation, @0.5 and @0.5:0.95; three detectors compared |
| Size, inference time | file size; median CPU latency (2 threads) per classifier variant; end-to-end `analyze()` stage timings |
| Robustness | fixed 1,000-image test subset × 12 conditions (dark, bright, low contrast, occlusion 10/25/40% of the car box, ×¼ and ×⅛ resolution, blur σ 2/4, noise σ 20), detection re-run on every perturbed image |
| Confidence & threshold | reliability diagram + ECE, accuracy-coverage curve, top-3 accuracy below threshold |
| Verification (ours) | every test image × {true claim, hardest look-alike, random other make}; MATCH rate curves over τ |
| Unknown cars | flag rate and AUROC on a probe set of post-2012 cars vs in-dataset test images |

## 7. Results

The full numbers are in `eval_P5_<TeamID>.md` (generated by `scripts/make_report.py` from `results/*.json`) and on the app's Results tab. The headline figures are summarised in the README.

## 8. Failure analysis

Five failure cases are selected automatically from the test errors (year confusion, confident wrong make, wrong localisation, low light, car cut off by the frame). Each is rendered with the detection, the classifier's crop, Grad-CAM and reference images (`scripts/render_failures.py`), and explained by hand in `results/failures.json`.

## 9. Limitations and ethics

See the About tab ([`about.md`](about.md)). In short: the dataset ends at 2012 and covers US-market cars only. The year is the weakest level. Strictness costs more manual checks. The tool flags and never accuses, and no faces or plates are processed. The optional Gemini features are opt-in, labelled, and never decide anything.

## 10. Deployment

Streamlit Community Cloud, deployed from this GitHub repo (Hugging Face now requires a paid plan for Docker Spaces). The app downloads the models from the GitHub release on first start and verifies them with SHA-256; Linux hosts get the CPU-only PyTorch wheel. A `Dockerfile` is provided for any other host. One-command local run: `run.ps1` / `run.sh`. The Gemini key is a host secret and is never committed.

## 11. Team and contributions

| Member | Contribution |
|---|---|
| Team lead | System design and full implementation (data, detector, classifier, hybrid verification, evaluation, app, deployment) |
| Member 2 | UI design |
| Member 3 | ML model training |
| Member 4 | Classical computer-vision methods |
