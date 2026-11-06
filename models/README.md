# models/

Deployable model files are not stored in git. Fetch them with

```bash
python scripts/download_models.py
```

| File | What it is |
|---|---|
| `detector.pt` | YOLO11n fine-tuned on Stanford Cars boxes (single class "car") |
| `classifier.onnx` | ConvNeXt-Tiny classifier, optimised ONNX used for inference |
| `classifier.pt` | Same classifier in PyTorch (FP16 weights), used for Grad-CAM |
| `classical_svm.joblib` | Classical pipeline: scaler + PCA + SVM on handcrafted features |
| `ood_bank.npy` | Training-set embeddings for the "not in my knowledge" check |
| `hybrid.json` | Validation-fitted temperature, fusion weight and thresholds |
| `classes.json` | The 196 class names |
