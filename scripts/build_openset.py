"""Build the zero-shot open-set fallback: OpenCLIP ViT-B/32 (LAION-2B), no training.

1. Text embeddings for 196 Stanford classes + ~300 post-2012 models (prompt ensemble).
2. Export CLIP's image encoder to ONNX, dynamic INT8 (runtime needs no torch for it).
3. Threshold on P(modern vocabulary): 97th percentile on the Stanford VALIDATION split,
   i.e. ~3% of known cars are flagged by this check on its own.
4. POST-HOC check on the modern-car probe set (added after the main test run; reported as such).

Usage: python scripts/build_openset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vericar.config import CROP_MARGIN, DATA_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402
from vericar.data import class_names, load_split  # noqa: E402
from vericar.openset import OpenSet  # noqa: E402
from vericar.openset_vocab import modern_labels  # noqa: E402
from vericar.preprocessing import crop_box, load_rgb  # noqa: E402

MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
TEMPLATES = ("a photo of a {}.", "a photo of a {}, a type of car.", "a close-up photo of the {}.")
PROBE_TRUTH = {"Audi_Q4_e_tron": "Audi Q4 e-tron", "Ford_Mustang_Mach_E": "Ford Mustang Mach-E",
               "Hyundai_Creta_SU2": "Hyundai Creta", "Hyundai_Ioniq_5": "Hyundai Ioniq 5", "Kia_EV6": "Kia EV6",
               "Mahindra_XUV700": "Mahindra XUV700", "Porsche_Taycan": "Porsche Taycan", "Tata_Nexon": "Tata Nexon",
               "Tesla_Model_3": "Tesla Model 3", "Tesla_Model_Y": "Tesla Model Y", "Toyota_GR86": "Toyota GR86",
               "Toyota_RAV4_XA50": "Toyota RAV4", "Volkswagen_ID_4": "Volkswagen ID.4"}


def main():
    import open_clip
    from onnxruntime.quantization import QuantType, quantize_dynamic
    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    tok = open_clip.get_tokenizer(MODEL)
    model.eval()
    known, modern = class_names(), modern_labels()
    vocab = known + modern
    with torch.no_grad():
        T = []
        for lab in vocab:
            e = model.encode_text(tok([t.format(lab) for t in TEMPLATES]))
            e = (e / e.norm(dim=-1, keepdim=True)).mean(0)
            T.append(e / e.norm())
        T = torch.stack(T).numpy().astype(np.float32)

    class Visual(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            f = self.m.encode_image(x)
            return f / f.norm(dim=-1, keepdim=True)

    fp32 = MODELS_DIR / "_openset_fp32.onnx"
    torch.onnx.export(Visual(model).eval(), torch.randn(1, 3, 224, 224), str(fp32), input_names=["image"],
                      output_names=["embedding"], dynamic_axes={"image": {0: "b"}, "embedding": {0: "b"}},
                      opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32), str(MODELS_DIR / "openset_clip_int8.onnx"), weight_type=QuantType.QInt8)
    fp32.unlink(missing_ok=True)
    np.save(MODELS_DIR / "openset_text.npy", T.astype(np.float16))
    meta = {"model": MODEL, "pretrained": PRETRAINED, "labels": vocab, "n_known": len(known),
            "logit_scale": float(model.logit_scale.exp().item()), "tau": 1.0}
    (MODELS_DIR / "openset.json").write_text(json.dumps(meta))

    # ---- calibrate on the Stanford VALIDATION split (known cars only)
    os_ = OpenSet(MODELS_DIR)
    val = load_split("val")
    pv, zs_ok = [], []
    for r in val:
        out = os_.raw(crop_box(load_rgb(r["path"]), r["box"], CROP_MARGIN))
        pv.append(out["p_modern"])
        zs_ok.append(int(np.argmax(out["p"][:len(known)])) == r["label"])
    pv = np.array(pv)
    tau = float(np.percentile(pv, 97))
    meta["tau"] = tau
    (MODELS_DIR / "openset.json").write_text(json.dumps(meta))
    os_ = OpenSet(MODELS_DIR)

    # ---- post-hoc probe check (modern cars), using the fine-tuned detector's crop like the app
    from common import detect_all, primary_box
    probe_dir = DATA_DIR / "probe_modern"
    files = sorted(p for p in probe_dir.glob("*.jpg"))
    recs = [{"path": str(p), "box": [0, 0, 1, 1], "label": 0} for p in files]
    boxes = [primary_box(d) for d in detect_all(MODELS_DIR / "detector.pt", recs)]
    rows = []
    for p, b in zip(files, boxes):
        img = load_rgb(p)
        crop = crop_box(img, b, CROP_MARGIN) if b is not None else img
        o = os_(crop)
        truth = PROBE_TRUTH[p.stem.rsplit("_", 1)[0]]
        top = o["top"][0]["label"] if o["top"] else ""
        rows.append({"file": p.name, "truth": truth, "p_modern": o["p_modern"], "flag": o["flag"],
                     "top3": [t["label"] for t in o["top"]], "top1_correct": top.startswith(truth),
                     "top3_correct": any(t["label"].startswith(truth) for t in o["top"])})
    knn = json.loads((RESULTS_DIR / "final_test.json").read_text())["unknown_detection"]
    knn_flags = {f: s > json.loads((MODELS_DIR / "hybrid.json").read_text())["ood_threshold"]
                 for f, s in zip(knn["probe_files"], knn["probe_scores"])}
    combined = np.mean([r["flag"] or knn_flags.get(r["file"], False) for r in rows])
    res = {"model": f"OpenCLIP {MODEL} ({PRETRAINED}), zero-shot, image encoder ONNX INT8",
           "vocab": {"known": len(known), "modern": len(modern)}, "tau_p_modern": tau,
           "val_flag_rate": float((pv > tau).mean()), "val_zero_shot_top1_196": float(np.mean(zs_ok)),
           "posthoc_probe": {"n": len(rows), "flag_rate_clip": float(np.mean([r["flag"] for r in rows])),
                             "flag_rate_knn": float(np.mean(list(knn_flags.values()))),
                             "flag_rate_combined": float(combined),
                             "top1_modern_name_acc": float(np.mean([r["top1_correct"] for r in rows])),
                             "top3_modern_name_acc": float(np.mean([r["top3_correct"] for r in rows])),
                             "rows": rows}}
    (RESULTS_DIR / "openset_eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "posthoc_probe"}, indent=1))
    print(json.dumps({k: v for k, v in res["posthoc_probe"].items() if k != "rows"}, indent=1))
    for r in rows:
        print(f"{r['file']:28s} flag={r['flag']!s:5s} p_modern={r['p_modern']:.2f} top1={r['top3'][0] if r['top3'] else '-'}")


if __name__ == "__main__":
    main()
