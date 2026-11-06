"""Download the trained model files from the GitHub release and verify their SHA-256.

Skips files that are already present with the right hash, so it is safe to run
on every start.  Usage:  python scripts/download_models.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vericar.config import MODELS_DIR  # noqa: E402

REPO = os.environ.get("VERICAR_REPO", "ksaihavish/vericar")
TAG = os.environ.get("VERICAR_MODELS_TAG", "models-v1")
MANIFEST = ROOT / "scripts" / "models_manifest.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, digest in manifest.items():
        dst = MODELS_DIR / name
        if dst.exists() and sha256(dst) == digest:
            continue
        url = f"https://github.com/{REPO}/releases/download/{TAG}/{name}"
        print(f"downloading {name} ...", flush=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        if sha256(tmp) != digest:
            tmp.unlink(missing_ok=True)
            print(f"checksum mismatch for {name}", file=sys.stderr)
            return 1
        tmp.replace(dst)
    print(f"models ready in {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
