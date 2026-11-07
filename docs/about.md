### Why VeriCar exists

A **cloned plate** is a real registration number copied onto a different car, usually one that *looks like* the registered vehicle. A parking or fleet officer can look up a plate in seconds; what they cannot easily check is whether the car in front of them is really the make, model and year on that record. VeriCar answers exactly that question from one photo.

### How it decides

1. **Find the car.** YOLO11n (fine-tuned on Stanford Cars boxes) draws the box. If it finds nothing, a purely classical edge-and-contour detector takes over and the result is marked *retake*.
2. **Check the photo.** Brightness, contrast, sharpness and crop size are measured classically. Dark or flat crops get gamma + CLAHE; unusable ones become **RETAKE** instead of a guess.
3. **Recognise.** ConvNeXt-Tiny (ImageNet-22k pre-trained, fine-tuned in two stages) scores all 196 classes; probabilities are temperature-calibrated on the validation split. A classical HOG/LBP/colour/geometry SVM is fused in with a validation-fitted weight; that weight came out as 0 (the classical vote didn't improve validation likelihood), and we report that rather than force it.
4. **Know what it doesn't know.** Two independent checks. (a) The car's embedding is far from every training car (k-nearest-neighbour distance). (b) An open-source, zero-shot OpenCLIP model finds the photo closer to a post-2012 model than to any of the 196 known classes. Either one gives **NOT IN RECORDS**: an honest "not in my 196 models", an educated guess at make and body type, and OpenCLIP's possible post-2012 matches, clearly labelled *unverified* (it names the exact model correctly only about half the time within its top 3).
5. **Verify the claim, level by level.** Class probabilities are summed into make → model → year. Only a claim that is the model's top prediction *and* clears the validation-fitted confidence bar is **MATCH**; a verified model with an unseparable year is **PARTIAL**; strong evidence against the claim is **MISMATCH**; everything else is **INSPECT**.

**Strict by design.** Letting a cloned car drive away costs more than a two-minute manual check, so the MATCH threshold was chosen on the validation split so that at most 2% of *hard* clones (the most similar-looking other model) get cleared. The price is a higher false-alarm rate, which is reported next to the catch rate on the Results tab.

### Limitations

- **Dataset era and coverage.** Stanford Cars contains 196 US-market models up to **2012**. Most cars on the road today are not in it; the system is built to say so (UNKNOWN) rather than force a label, but it cannot verify those cars.
- **Year is the weakest level.** Several classes are the same body across years (e.g. Audi S4 2007 vs 2012). The app reports PARTIAL instead of pretending.
- **Photos, not people.** Accuracy drops with heavy occlusion, very low resolution and extreme darkness (see robustness results). Unusual viewpoints (pure rear or top views) are rare in the training data.
- **Test set ≠ the street.** Stanford Cars images are mostly clean web photos. Numbers on phone photos in bad conditions will be lower.

### Ethics and responsible use

- VeriCar **flags, it never accuses**: every non-MATCH verdict means "a human checks", and the officer sees the evidence (candidates, Grad-CAM, confidence) behind it.
- The strict policy means **more legitimate drivers are checked**; deployers must weigh that inconvenience and must not use a MISMATCH alone as grounds for action.
- **No people, faces or plates are processed.** Only the vehicle crop is classified.
- **Optional AI text (Gemini).** The officer note is generated from the structured verdict only (no image). The second opinion for unknown cars *does* send the car crop to Google's API; it is off when no API key is configured, clearly labelled *unverified*, and never changes the verdict.
- Dataset images belong to their original owners and are used for research and education only.

### Team

| Member | Contribution |
|---|---|
| Kukkadapu Sai Havish (1RUA24CSE0212) | System design and full implementation: data pipeline, detector, classifier, hybrid verification, evaluation, app and deployment |
| *(member)* | UI design |
| *(member)* | ML model training |
| *(member)* | Classical computer-vision methods |
