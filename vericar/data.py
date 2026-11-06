"""Dataset records, the fixed train/val split and the make -> model -> year hierarchy.

Class names look like "Audi S4 Sedan 2012". We split them into
make ("Audi"), model family ("Audi S4 Sedan", i.e. the name without the year),
year (2012) and body type ("Sedan"). Summing class probabilities over these
groups gives the level-by-level verification used in the app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import DATA_DIR, MODELS_DIR

MULTIWORD_MAKES = ("AM General", "Aston Martin", "Land Rover")
BODY_TYPES = ("Sedan", "SUV", "Coupe", "Convertible", "Hatchback", "Wagon", "Minivan", "Van", "Cab",
              "SuperCab", "Pickup")


def parse_class_name(name: str) -> dict:
    words = name.split()
    year = int(words[-1]) if words[-1].isdigit() else -1
    make = next((m for m in MULTIWORD_MAKES if name.startswith(m + " ")), words[0])
    model = " ".join(words[:-1]) if year > 0 else name
    body = next((b for b in reversed(words[:-1]) if b in BODY_TYPES), "Other")
    if body in ("Cab", "SuperCab", "Pickup"):
        body = "Pickup"
    return {"make": make, "model": model, "year": year, "body": body}


@dataclass(frozen=True)
class Hierarchy:
    class_names: tuple[str, ...]
    make_names: tuple[str, ...]
    model_names: tuple[str, ...]
    body_names: tuple[str, ...]
    make_of: np.ndarray    # class -> make index
    model_of: np.ndarray   # class -> model-family index
    body_of: np.ndarray    # class -> body index
    year_of: np.ndarray    # class -> year

    @classmethod
    def from_names(cls, names: list[str]) -> "Hierarchy":
        parsed = [parse_class_name(n) for n in names]
        makes = sorted({p["make"] for p in parsed})
        models = sorted({p["model"] for p in parsed})
        bodies = sorted({p["body"] for p in parsed})
        return cls(tuple(names), tuple(makes), tuple(models), tuple(bodies),
                   np.array([makes.index(p["make"]) for p in parsed]),
                   np.array([models.index(p["model"]) for p in parsed]),
                   np.array([bodies.index(p["body"]) for p in parsed]),
                   np.array([p["year"] for p in parsed]))

    def group_probs(self, probs: np.ndarray, level: str) -> np.ndarray:
        """Sum class probabilities [..., 196] into make / model / body groups."""
        idx = {"make": self.make_of, "model": self.model_of, "body": self.body_of}[level]
        n = {"make": len(self.make_names), "model": len(self.model_names), "body": len(self.body_names)}[level]
        out = np.zeros(probs.shape[:-1] + (n,), np.float64)
        for c in range(probs.shape[-1]):
            out[..., idx[c]] += probs[..., c]
        return out

    def describe(self, c: int) -> dict:
        return {"label": self.class_names[c], "make": self.make_names[self.make_of[c]],
                "model": self.model_names[self.model_of[c]], "year": int(self.year_of[c]),
                "body": self.body_names[self.body_of[c]]}


def class_names() -> list[str]:
    for p in (MODELS_DIR / "classes.json", DATA_DIR / "classes.json"):
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError("classes.json not found - run scripts/prepare_data.py or scripts/download_models.py")


@lru_cache(maxsize=1)
def hierarchy() -> Hierarchy:
    return Hierarchy.from_names(class_names())


def load_split(split: str) -> list[dict]:
    """Records {path, box (x1,y1,x2,y2 in cached-image pixels), label, id} for train/val/test."""
    recs = json.loads((DATA_DIR / f"{split}.json").read_text())
    for r in recs:
        r["path"] = str(DATA_DIR / r["path"])
    return recs


def split_path(split: str) -> Path:
    return DATA_DIR / f"{split}.json"
