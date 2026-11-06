"""Fine-tune ConvNeXt-Tiny on Stanford Cars crops - with explicit anti-overfitting guards.

Stage 1: backbone frozen, only the linear head trains (a few epochs).
Stage 2: last two ConvNeXt stages + head train with discriminative LRs.
Guards: hard epoch cap, early stopping on val loss, best checkpoint by val top-1,
label smoothing, dropout, drop-path, weight decay, moderate augmentation, and a
train-vs-val gap monitor that STOPS training if (epoch train top-1 - val top-1)
exceeds --max-gap points. The stricter clean-train gap (non-augmented train
subset vs val) is logged every epoch as well and reported.

Usage:  python scripts/train_classifier.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from vericar.classifier import CarCrops, build_model, gpu_augment, param_groups, set_trainable  # noqa: E402
from vericar.config import CLS_IMG_SIZE, FEAT_DIR, MODELS_DIR, RUNS_DIR, SEED  # noqa: E402
from vericar.data import class_names, load_split  # noqa: E402

CL = torch.channels_last


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def loader(records, train, bs, workers, persistent=True):
    return DataLoader(CarCrops(records, train), batch_size=bs, shuffle=train, num_workers=workers,
                      pin_memory=True, persistent_workers=persistent and workers > 0, drop_last=train)


@torch.no_grad()
def predict(model, dl, dev):
    """Per-sample logits/feats/labels without augmentation."""
    model.eval()
    L, Fe, Y = [], [], []
    for x, y in dl:
        x = gpu_augment(x.to(dev, non_blocking=True), False).to(memory_format=CL)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, feats = model(x)
        L.append(logits.float())
        Fe.append(feats.float())
        Y.append(y.to(dev))
    return torch.cat(L), torch.cat(Fe), torch.cat(Y)


def metrics(logits, y):
    return (F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item(),
            (logits.topk(5, 1).indices == y[:, None]).any(1).float().mean().item())


def cosine_warmup(opt, total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=18)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lr-top", type=float, default=1.2e-4)
    ap.add_argument("--max-gap", type=float, default=10.0, help="stop if epoch train top-1 minus val top-1 > this (points)")
    args = ap.parse_args()

    seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    out_dir = RUNS_DIR / "classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = class_names()
    train_recs, val_recs = load_split("train"), load_split("val")
    clean_train = random.Random(SEED).sample(train_recs, len(val_recs))  # same size as val, no augmentation
    nv = len(val_recs)

    dl_train = loader(train_recs, True, args.bs, args.workers)
    dl_eval = loader(val_recs + clean_train, False, 64, 2)   # one persistent loader: val then clean-train

    model = build_model(len(names)).to(dev).to(memory_format=CL)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    history, best = [], {"val_top1": -1.0}

    def run_stage(stage, epochs, groups, warmup_epochs):
        nonlocal best
        set_trainable(model, stage)
        opt = torch.optim.AdamW(groups, weight_decay=0.05)
        sched = cosine_warmup(opt, epochs * len(dl_train), warmup_epochs * len(dl_train))
        params = [p for g in groups for p in g["params"]]
        best_loss, bad = float("inf"), 0
        for ep in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            if stage == "head":
                model.backbone.eval()
            tl = tc = tn = 0
            for x, y in dl_train:
                x = gpu_augment(x.to(dev, non_blocking=True), True).to(memory_format=CL)
                y = y.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if stage == "head":
                        with torch.no_grad():
                            f = model.backbone(x)
                        logits = model.fc(model.dropout(f))
                    else:
                        logits, _ = model(x)
                loss = crit(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                tl += loss.item() * len(y)
                tc += (logits.argmax(1) == y).sum().item()
                tn += len(y)
            lo, _, yy = predict(model, dl_eval, dev)
            vl, v1, v5 = metrics(lo[:nv], yy[:nv])
            _, c1, _ = metrics(lo[nv:], yy[nv:])
            gap = (tc / tn - v1) * 100
            rec = {"stage": stage, "epoch": ep, "train_loss": tl / tn, "train_top1_aug": tc / tn,
                   "train_top1_clean": c1, "val_loss": vl, "val_top1": v1, "val_top5": v5,
                   "gap_points": gap, "gap_clean_points": (c1 - v1) * 100,
                   "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0}
            history.append(rec)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            (out_dir / "history.json").write_text(json.dumps(history, indent=1))
            if v1 > best["val_top1"]:
                best = {"val_top1": v1, "val_top5": v5, "val_loss": vl, "stage": stage, "epoch": ep}
                torch.save({"state_dict": {k: v.half() for k, v in model.state_dict().items()},
                            "class_names": names, "img_size": CLS_IMG_SIZE, "best": best}, out_dir / "best.pt")
            if gap > args.max_gap:
                print(f"STOP: train-val gap {gap:.1f} points > {args.max_gap}. Best checkpoint kept.", flush=True)
                return "gap"
            if vl < best_loss - 1e-3:
                best_loss, bad = vl, 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"Early stopping: val loss did not improve for {args.patience} epochs.", flush=True)
                    return "early_stop"
        return "max_epochs"

    r1 = run_stage("head", args.head_epochs, [{"params": list(model.fc.parameters()), "lr": 2e-3}], 0)
    set_trainable(model, "top")
    r2 = run_stage("top", args.max_epochs, param_groups(model, args.lr_head, args.lr_top), 1) if r1 != "gap" else "skipped"

    summary = {"best": best, "stage1_end": r1, "stage2_end": r2, "args": vars(args)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("SUMMARY", json.dumps(summary), flush=True)

    # reload best, export deployable checkpoint + logits/embeddings for hybrid fitting
    ck = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    (MODELS_DIR / "classifier.pt").write_bytes((out_dir / "best.pt").read_bytes())
    for split, recs in (("train", train_recs), ("val", val_recs)):
        lo, fe, yy = predict(model, loader(recs, False, 64, args.workers, persistent=False), dev)
        np.savez_compressed(FEAT_DIR / f"cnn_{split}.npz", logits=lo.cpu().numpy(),
                            feats=fe.cpu().numpy().astype(np.float16), labels=yy.cpu().numpy())
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
