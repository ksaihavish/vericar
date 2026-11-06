"""Publish trained models (GitHub release) and the app (Hugging Face Docker Space).

    python scripts/publish.py --release                      # models -> GitHub release + manifest
    python scripts/publish.py --space <hf-user>/vericar      # upload the app to a Docker Space

The Space needs `hf auth login` (a write token) once on this machine. The optional
GEMINI_API_KEY is added as a Space *secret* in the Space settings, never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vericar.config import MODELS_DIR  # noqa: E402

REPO = "ksaihavish/vericar"
TAG = "models-v1"
MANIFEST = ROOT / "scripts" / "models_manifest.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def release():
    meta = json.loads((MODELS_DIR / "hybrid.json").read_text())
    files = ["detector.pt", "classifier.pt", meta["onnx_file"], "classical_svm.joblib", "ood_bank.npy",
             "hybrid.json", "classes.json"]
    manifest = {f: sha256(MODELS_DIR / f) for f in files}
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    paths = [str(MODELS_DIR / f) for f in files]
    exists = subprocess.run(["gh", "release", "view", TAG, "-R", REPO], capture_output=True).returncode == 0
    if exists:
        subprocess.run(["gh", "release", "upload", TAG, *paths, "-R", REPO, "--clobber"], check=True)
    else:
        subprocess.run(["gh", "release", "create", TAG, *paths, "-R", REPO, "--title", "Trained models v1",
                        "--notes", "Model files for VeriCar; downloaded and SHA-256-verified by scripts/download_models.py."],
                       check=True)
    print("release ready:", files)


SPACE_HEADER = """---
title: VeriCar
emoji: 🚘
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Fine-grained car recognition and plate-clone verification
---

"""


def space(space_id: str):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        folder_path=str(ROOT), repo_id=space_id, repo_type="space",
        ignore_patterns=[".git/*", ".venv/*", "models/*.pt", "models/*.onnx", "models/*.joblib", "models/*.npy",
                         "models/*.json", "**/__pycache__/*", ".env", "results/failures/*.png.tmp", "probe/*"],
        commit_message="Deploy VeriCar")
    readme = SPACE_HEADER + (ROOT / "README.md").read_text(encoding="utf-8")
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", repo_id=space_id,
                    repo_type="space", commit_message="Space card")
    print(f"https://huggingface.co/spaces/{space_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--space", default=None)
    a = ap.parse_args()
    if a.release:
        release()
    if a.space:
        space(a.space)


if __name__ == "__main__":
    main()
