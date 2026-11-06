"""Build the 'cars outside the dataset' probe set from Wikimedia Commons (free licences).

Stanford Cars stops at 2012; these categories are post-2015 models, so none of them is
one of the 196 classes. Images go to VERICAR_WORK/data/probe_modern/ with a
sources.json recording each file's Commons page, author and licence (for attribution).
Used ONLY to evaluate the unknown-car check; never for training or tuning.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vericar.config import DATA_DIR  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "VeriCar-hackathon-probe/1.0 (educational computer-vision project)"}
CATEGORIES = [
    "Tesla Model 3", "Tesla Model Y", "Hyundai Ioniq 5", "Kia EV6", "Hyundai Creta (SU2)", "Kia Seltos",
    "Tata Nexon", "Mahindra XUV700", "Toyota RAV4 (XA50)", "Toyota Corolla (E210)", "Honda Civic (eleventh generation)",
    "BMW 3 Series (G20)", "Mercedes-Benz C-Class (W206)", "Audi Q4 e-tron", "Volkswagen ID.4", "Ford Mustang Mach-E",
    "Ford Bronco (sixth generation)", "Chevrolet Corvette (C8)", "Porsche Taycan", "Toyota GR86",
]
PER_CATEGORY = 2


def _open(url: str, timeout: int):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 3:
                raise
            time.sleep(20 * (attempt + 1))   # polite back-off on rate limiting


def get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    with _open(url, 30) as r:
        return json.loads(r.read())


def main():
    out = DATA_DIR / "probe_modern"
    out.mkdir(parents=True, exist_ok=True)
    sp = out / "sources.json"
    sources = json.loads(sp.read_text()) if sp.exists() else []
    done = {s["model"] for s in sources}
    for cat in CATEGORIES:
        if cat in done:
            continue
        time.sleep(2)
        try:
            data = get({"action": "query", "generator": "categorymembers", "gcmtitle": f"Category:{cat}",
                        "gcmtype": "file", "gcmlimit": 30, "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime",
                        "iiurlwidth": 900})
        except Exception as e:  # noqa: BLE001
            print("skip", cat, e)
            continue
        pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p["title"])
        n = 0
        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            if ii.get("mime") != "image/jpeg" or ii.get("width", 0) < 800:
                continue
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            if not re.search(r"CC|Public domain|PD", lic, re.I):
                continue
            name = re.sub(r"[^A-Za-z0-9]+", "_", cat).strip("_") + f"_{n + 1}.jpg"
            with _open(ii["thumburl"], 60) as r, open(out / name, "wb") as f:
                f.write(r.read())
            author = re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
            sources.append({"file": name, "model": cat, "page": ii.get("descriptionurl"), "author": author,
                            "license": lic})
            n += 1
            time.sleep(2)
            if n >= PER_CATEGORY:
                break
        print(cat, n, flush=True)
        sp.write_text(json.dumps(sources, indent=1))
    print("probe images:", len(sources))


if __name__ == "__main__":
    main()
