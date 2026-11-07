# Viva preparation

The rubric gives 25 points to the demo and viva: *"every member can explain their own part, the model and classical choices, the metrics and a failure case."* Each section below belongs to one member. Read your own section and the **shared core** at the end. Numbers are in `docs/eval_P5_T70.md`.

---

## 1. Kukkadapu Sai Havish (1RUA24CSE0212): system, hybrid verification, evaluation, deployment

**What it does.** `VeriCar.analyze(image, claim)` runs three pipelines on every photo and turns the hybrid's probabilities into a verdict: MATCH / PARTIAL / INSPECT / MISMATCH / UNKNOWN / RETAKE.

**Why claim verification and not just classification?** The user (a parking/fleet officer) already *has* a claimed identity from the plate record. The real question is "does this car match?", and cloned plates are put on look-alike cars, which is exactly the fine-grained problem.

**Likely questions**
- *Why is it strict?* A missed clone costs more than a two-minute manual check. τ_match is the lowest threshold that clears ≤ 2% of *hard* clones on validation. The cost of that choice is the false-alarm rate, which we report.
- *What is a hard clone?* For each class, the most similar other class: nearest class prototype in the CNN embedding space, computed from training images only.
- *How do you know the car is outside the dataset?* Cosine kNN distance (k = 10) from the image's 768-d embedding to all training embeddings. The threshold is the 95th percentile of validation distances, so about 5% of known cars are also sent to a manual check. AUROC against modern-car photos is in the report.
- *How did you avoid tuning on the test set?* Every parameter was frozen on validation. `evaluate_final.py` refuses to run twice, and `results/test_run.json` is the record.
- *What happens with a 2024 car?* Two independent unknown checks: kNN distance to our training embeddings, and an open-source OpenCLIP ViT-B/32 model (zero-shot, not trained by us) that compares the photo with text names of our 196 classes plus 343 post-2012 models. On 25 modern-car photos, kNN alone flagged 64% and both together flagged 100% (25 of 25). CLIP's names are shown as unverified leads (top-3 right 52% of the time), never as an identification, and CLIP can only *send a car to manual check*, never clear one. We added it after the test run and say so.
- *What does Gemini do?* Only wording: it writes the officer note from the structured verdict (text only), and gives a labelled "unverified" second opinion for UNKNOWN cars. It never changes a verdict, and the app works without it.

## 2. Harshitha R (1RUA24CSE0159): ML model training

**What it does.** ConvNeXt-Tiny, ImageNet-22k pre-trained, fine-tuned on car crops at 384 px for 196 classes.

**Why ConvNeXt-Tiny?** A strong transfer backbone for fine-grained tasks that still fits a 6 GB laptop GPU with mixed precision and runs on a free CPU host in under a second (ONNX).

**Training recipe (anti-overfitting)**
- 85/15 stratified split, seed 42; the test set is never seen.
- Stage 1: backbone frozen, head only (3 epochs). Stage 2: only the last two ConvNeXt stages + head, with discriminative LRs (head 5e-4 > stage 3 1.2e-4 > stage 2 4.8e-5).
- Label smoothing 0.1, dropout 0.4, drop-path 0.1, weight decay 0.05, gradient clipping.
- Augmentation that never changes make/model/year: flip, box-margin/position jitter, 10% full-frame samples, brightness/contrast/saturation jitter, small random erasing (for occlusion). Colour jitter runs on the GPU to keep the laptop's CPU free.
- Early stopping on val loss (patience 4), best checkpoint by val top-1, and a stop if the train−val accuracy gap exceeds 10 points.

**Likely questions**
- *Is it overfitting?* Show the training-curve chart: epoch train accuracy vs validation, plus the clean-train curve. With about 35 images per class, clean-train accuracy is always higher than validation. What matters is that validation loss stopped improving and that's where we stopped.
- *Why not unfreeze everything?* Early layers hold generic edges and textures that 35 images per class can't improve; freezing them reduces overfitting and memory.
- *What is temperature scaling?* One scalar T divides the logits so confidences match observed accuracy (lower ECE). Fitted on validation; it does not change the top-1 label.
- *Optimisation?* ONNX FP32 → FP16 → INT8 (dynamic and static with calibration on training crops). We ship the fastest variant within 0.5 points of FP32 accuracy on validation.

## 3. Dhyey Abhay Ghoda (1RUA24CSE0134): classical computer vision

**What it does.** A full non-learning-based perception path, plus classical components inside the hybrid.

**Pieces**
- **Preprocessing:** gray-world colour constancy, CLAHE on the L channel (illumination normalisation), gamma correction for dark images.
- **Edges and contours:** Canny with thresholds from the median intensity (no hand tuning), morphological closing to join the car's outline, largest external contour → classical ROI box (IoU vs ground truth in the report).
- **Features (3,913-d):** HOG (shape), LBP at two scales + GLCM (texture), HSV histograms + Lab moments (colour), aspect ratio, extent, solidity, circularity, Hu moments and an edge-density grid (geometry).
- **Classifier:** StandardScaler → PCA 256 → RBF-SVM (C=3 chosen on validation; beat a random forest 21.3% vs 9.3%).
- **Inside the hybrid:** the quality gate (brightness, Laplacian sharpness, size → RETAKE), adaptive enhancement, the fallback ROI when YOLO finds nothing, and late fusion of the SVM probabilities.

**Likely questions**
- *Why is classical accuracy so low?* 196 classes that differ by grille and lamp details. Global descriptors capture shape and texture but not those part-level cues. That's why it's honest to show it, and why it's used for quality, ROI and robustness rather than as the main classifier.
- *Which classical features matter?* The feature-family ablation: removing HOG drops validation top-1 from 21.3% to 8.3%, while removing colour barely matters (a model's paint colour varies, so colour says little about the model).
- *Where does classical help the hybrid?* Be honest, because the numbers are honest. The validation-fitted fusion weight came out 0, so the classical SVM doesn't change the label. Classical CV does the ROI fallback, the quality gate (it turns very blurry or tiny inputs into RETAKE instead of confident mistakes) and the contour detector (mAP@0.5 0.64 with no learning). Adaptive gamma + CLAHE actually *hurt* on very dark photos (90.4% vs 93.5%), because the CNN's brightness augmentation already copes with darkness. That's a real finding, and we report it.

## 4. Harshitha Narasimha (1RUA24CSE0158): UI design

**What it does.** A Streamlit app built for a phone-holding officer: verdict first, then evidence.

**Design decisions**
- Verdict card with an icon + label + colour (never colour alone), headline and one-sentence reason.
- Level-by-level bars (make → model → year) so the officer sees *which* level failed.
- Look-alikes side by side with reference images: the core of the "why".
- Tabs: Verify · Pipeline (every intermediate image) · Classical vs Deep vs Hybrid · Results · About.
- Colours from a colour-blind-validated palette (method colours checked with a CVD validator); dark theme for outdoor/night use; works at phone width; loading spinners and error messages for bad files.
- Charts follow one rule set: one axis, legends, hover tooltips, a table next to each chart.

**Likely questions**
- *What happens with a bad upload?* An unreadable file gets an error message; a dark, blurry or tiny car gets a RETAKE verdict; a photo with no car gets "no vehicle detected".
- *Why is the threshold slider only in Identify mode?* In Verify mode the thresholds are a policy (strictness) fixed on validation; letting a user drag it would make the catch rate meaningless.

---

## Shared core (everyone)

- **Dataset:** Stanford Cars, 16,185 images, 196 classes, years up to 2012, one box per image. Official split 8,144 / 8,041; we split train into 6,922 train / 1,222 val.
- **Metrics:** Top-1/Top-5 · IoU · mAP@0.5 and @0.5:0.95 · size and CPU latency · robustness under 12 conditions · calibration (ECE) and accuracy-coverage · verification catch / false-alarm rates.
- **A failure case to explain:** pick one from the Results → Failures tab (e.g. same model, wrong year). Say *what* went wrong (prediction vs truth), *why* (the classes look identical from that angle; Grad-CAM focuses on body shape, not the year-specific lamps) and *what the system does about it* (PARTIAL instead of MATCH; manual document check).
- **Limitations:** 2012 cut-off and US-market classes; year is the weakest level; strictness costs extra checks; clean web photos differ from street photos.
