# VeriCar - system architecture

Every box names the module (and function) that implements it, so the diagram can be checked against the code.

## Inference: `VeriCar.analyze(image, claim=None)` in `vericar/pipeline.py`

```mermaid
flowchart TB
    IN["Photo (upload / camera)<br/>+ optional claimed registration<br/>make → model → year"] --> LOAD["load_rgb + resize_max 640 px<br/><i>preprocessing.py</i>"]

    LOAD --> DET["YOLO11n detector, fine-tuned<br/><i>detector.py · Detector.detect</i>"]
    DET -->|no car found| CROI["Classical ROI fallback<br/>auto-Canny → closing → largest contour<br/><i>classical.py · classical_roi</i>"]
    DET -->|box| CROP["crop_box + 8% margin"]
    CROI --> CROP

    subgraph CLASSICAL["Classical-only pipeline"]
        direction TB
        C1["classical_roi on full frame"] --> C2["extract_features<br/>HOG · LBP · GLCM · HSV/Lab · Hu/shape · edge grid"]
        C2 --> C3["Scaler → PCA 256 → RBF-SVM<br/><i>models/classical_svm.joblib</i>"]
    end
    LOAD --> C1

    subgraph DEEP["Deep-only pipeline"]
        direction TB
        D1["classifier.preprocess<br/>384×384, ImageNet norm"] --> D2["ConvNeXt-Tiny (ONNX)<br/><i>classifier.py · Classifier.predict</i>"]
    end
    CROP --> D1

    subgraph HYBRID["Hybrid pipeline (ours)"]
        direction TB
        H1["quality_report<br/>brightness · contrast · sharpness · size"] --> H2["adaptive_enhance<br/>gamma + CLAHE only if dark / flat"]
        H2 --> H3["ConvNeXt-Tiny → logits + 768-d embedding"]
        H2 --> H4["classical features → SVM probs"]
        H3 --> H5["fuse: temperature T + weight w<br/><i>hybrid.py · fuse</i>"]
        H4 --> H5
        H3 --> H6["kNN distance to train embeddings<br/><i>hybrid.py · KnnOOD</i>"]
        H5 --> H7{"verdict<br/><i>hybrid.py · verify / identify</i>"}
        H6 --> H7
        H1 -->|issues| H7
    end
    CROP --> H1

    H7 --> OUT["MATCH · PARTIAL · INSPECT · MISMATCH · UNKNOWN · RETAKE<br/>(identify mode: CONFIDENT · LOW_CONFIDENCE + top-3)"]
    C3 --> CMP["per-method top-5 for the comparison tab"]
    D2 --> CMP
    H5 --> CMP
    OUT --> UI["Streamlit UI<br/><i>app/streamlit_app.py</i>"]
    CMP --> UI
    UI -. optional, text only .-> LLM["Gemini officer note<br/><i>llm.py · officer_note</i>"]
    UI -. optional, UNKNOWN only .-> LLM2["Gemini second opinion (unverified)<br/><i>llm.py · second_opinion</i>"]
```

## Offline: training, fitting and evaluation (`scripts/`)

```mermaid
flowchart LR
    K["Stanford Cars (Kaggle)<br/>+ verified test labels"] --> P["prepare_data.py<br/>cache 640 px · split 85/15 stratified, seed 42"]
    P --> TR["train split 6,922"]
    P --> VA["val split 1,222"]
    P --> TE["test split 8,041<br/><b>read only by evaluate_final.py</b>"]
    TR --> TD["train_detector.py<br/>YOLO11n fine-tune"]
    TR --> TC["train_classifier.py<br/>head → top stages, early stop, gap guard"]
    TR --> TS["train_classical.py<br/>features + SVM"]
    VA --> TD & TC & TS
    TD & TC & TS --> FH["fit_hybrid.py (val only)<br/>T, w, kNN threshold, strict thresholds"]
    TC --> OP["optimize.py (val only)<br/>ONNX FP32 / FP16 / INT8"]
    FH & OP --> EV["evaluate_final.py<br/>one run on test"]
    TE --> EV
    EV --> R["results/*.json → Results tab, eval_P5 report"]
```
