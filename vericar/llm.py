"""Optional Gemini helpers. The pipeline makes every decision; the LLM only
(1) rewrites the structured verdict as a short officer note (text only, no image), and
(2) gives a clearly labelled, unverified second opinion when the car is outside the
    196 known classes (the only case where the image is sent).
Everything degrades gracefully: no key / no network -> functions return None.
"""
from __future__ import annotations

import json
import os

DEFAULT_MODELS = ("gemini-flash-latest", "gemini-2.5-flash")


def _key() -> str | None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            key = None
    return key or None


def available() -> bool:
    return _key() is not None


def _generate(contents, max_tokens: int = 220) -> str | None:
    key = _key()
    if not key:
        return None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        models = [os.environ["GEMINI_MODEL"]] if os.environ.get("GEMINI_MODEL") else list(DEFAULT_MODELS)
        for m in models:
            try:
                r = client.models.generate_content(
                    model=m, contents=contents,
                    config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=max_tokens))
                if r.text:
                    return r.text.strip()
            except Exception:
                continue
    except Exception:
        return None
    return None


def officer_note(result: dict, claim_label: str | None) -> str | None:
    """2-3 sentence note for the officer, built ONLY from the pipeline's structured output."""
    v = result["verdict"]
    facts = {
        "verdict": v["status"], "headline": v["headline"], "detail": v["detail"],
        "claimed_registration": claim_label,
        "level_checks": [{k: (round(x, 3) if isinstance(x, float) else x) for k, x in lv.items()} for lv in v["levels"]],
        "top_candidates": [{"label": c["label"], "prob": round(c["prob"], 3)} for c in v["candidates"]],
        "image_quality_issues": result["quality"]["issues"], "outside_known_models": result["unknown"],
    }
    prompt = ("You write a short note for a parking/fleet officer checking whether a car matches its "
              "registration. Use ONLY the facts in the JSON. Do not name any car model that is not in the "
              "JSON. Say what to do next (clear, check documents, or inspect). 2-3 plain sentences, no "
              "markdown.\n\n" + json.dumps(facts))
    return _generate(prompt, 160)


def second_opinion(jpeg_bytes: bytes, guess: dict) -> str | None:
    """Unverified opinion on a car outside the 196 training classes (image is sent to Gemini)."""
    try:
        from google.genai import types
    except Exception:
        return None
    prompt = ("Identify this car's make, model and approximate model-year range from visible design cues "
              "(grille, headlights, body shape). Give your confidence (low/medium/high) and name 1-2 cues. "
              f"A separate classifier guessed make '{guess.get('make')}' and body '{guess.get('body')}'; "
              "agree or disagree. Max 3 short sentences, no markdown.")
    return _generate([types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"), prompt], 160)
