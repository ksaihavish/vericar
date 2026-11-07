"""VeriCar Streamlit app - a thin UI over ``VeriCar().analyze``.

Layout follows Streamlit's bundled design references (theme in config.toml, native elements):
input card (registered vehicle · photo · demo cases) -> verdict card (left, the answer) next to the
photo card (right, the evidence) -> look-alikes.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

import components as ui  # noqa: E402
from results_view import hsv_figure, render_results  # noqa: E402
from vericar import llm  # noqa: E402
from vericar.config import ASSETS_DIR  # noqa: E402
from vericar.hybrid import Thresholds, identify  # noqa: E402

st.set_page_config(page_title="VeriCar · Vehicle identity check", page_icon=":material/directions_car:",
                   layout="wide", initial_sidebar_state="collapsed")
NO_CLAIM = "None, just identify the car"


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading models (the first start downloads them, about a minute)…")
def engine():
    import runpy
    # hosts without a build step (e.g. Streamlit Community Cloud): fetch + SHA-256-verify the release files
    dl = runpy.run_path(str(ROOT / "scripts" / "download_models.py"))
    if dl["main"]() != 0:
        raise RuntimeError("model download failed")
    from vericar.pipeline import VeriCar
    return VeriCar()


@st.cache_data
def demo_items() -> list[dict]:
    p = ASSETS_DIR / "demo" / "demo.json"
    return json.loads(p.read_text()) if p.exists() else []


@st.cache_data(max_entries=8, show_spinner=False)
def analyze_cached(image_bytes: bytes, claim) -> dict:
    """Streamlit re-runs the script on every click; the models run once per (photo, claim)."""
    return engine().analyze(Image.open(io.BytesIO(image_bytes)), claim=claim)


@st.cache_data(max_entries=96, show_spinner=False)
def note_cached(key: str, _res: dict, claim_label):
    """At most one Gemini call per (photo, claim); errors return None and the panel stays hidden."""
    return llm.officer_note(_res, claim_label)


@st.cache_data(max_entries=96, show_spinner=False)
def opinion_cached(key: str, _jpeg: bytes, _guess: dict):
    return llm.second_opinion(_jpeg, _guess)


def to_jpeg(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def ref_image(idx: int):
    for folder in ("ref_thumbs", "ref"):
        p = ASSETS_DIR / folder / f"{idx:03d}.jpg"
        if p.exists():
            return str(p)
    return None


def on_demo():
    title = st.session_state.get("demo_pick")
    d = next((x for x in demo_items() if x["title"] == title), None)
    st.session_state.demo = d
    if d is not None:
        st.session_state.claim_select = d.get("claim") or NO_CLAIM
        st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1   # clear an uploaded file


# ------------------------------------------------------------------ header
head, meta = st.columns([1.6, 1], vertical_alignment="bottom")
with head:
    st.title("VeriCar")
    st.caption("Vehicle identity check for spotting cloned plates: does the car in the photo match its registration?")
with meta, st.container(horizontal=True, horizontal_alignment="right"):
    st.badge("YOLO11n detector", icon=":material/center_focus_strong:", color="gray")
    st.badge("ConvNeXt-Tiny classifier", icon=":material/neurology:", color="gray")
    st.badge("196 models · Stanford Cars ≤ 2012", icon=":material/database:", color="gray")

try:
    eng = engine()
except Exception as e:  # pragma: no cover - shown to the user
    st.error(f"The models could not be loaded ({e}). Reload the page; if it persists, run "
             "`python scripts/download_models.py` locally.", icon=":material/error:")
    st.stop()
hier = eng.hier
st.session_state.setdefault("demo", None)
st.session_state.setdefault("uploader_key", 0)

tab_check, tab_pipe, tab_cmp, tab_res, tab_about = st.tabs([
    ":material/directions_car: Check a car", ":material/account_tree: Pipeline steps",
    ":material/compare: Classical vs deep vs hybrid", ":material/analytics: Test results", ":material/info: About"])

# ================================================================== CHECK A CAR
with tab_check:
    with st.container(border=True):
        c1, c2 = st.columns(2, gap="medium")
        choice = c1.selectbox("Registered vehicle", [NO_CLAIM] + sorted(hier.class_names), key="claim_select",
                              help="The make, model and year from the plate lookup. Type to search.")
        f = c2.file_uploader("Photo of the car", type=["jpg", "jpeg", "png", "webp"],
                             key=f"up{st.session_state.uploader_key}", help="On a phone you can take a photo.")
        st.pills("Or try a demo case", [d["title"] for d in demo_items()], key="demo_pick", on_change=on_demo)
    claim = None if choice == NO_CLAIM else hier.class_names.index(choice)
    image_bytes = None
    if f is not None:
        image_bytes, st.session_state.demo = f.getvalue(), None
    elif st.session_state.demo is not None:
        image_bytes = (ASSETS_DIR / "demo" / st.session_state.demo["file"]).read_bytes()
        st.caption(st.session_state.demo.get("note", ""))

    if image_bytes is None:     # ------------------------------------------------ empty state
        cols = st.columns(3)
        steps = ((":material/badge:", "1. Choose the registered vehicle",
                  "The make, model and year from the plate lookup, or leave it on “None” to just identify the car."),
                 (":material/add_a_photo:", "2. Add a photo", "Upload or take a photo, or tap a demo case above."),
                 (":material/fact_check:", "3. Read the verdict",
                  "Cleared, check documents, manual check, possible clone, or not in records for cars newer than "
                  "2012."))
        for col, (icon, title, text) in zip(cols, steps):
            with col.container(border=True, height="stretch"):
                st.markdown(f"{icon} **{title}**")
                st.caption(text)
    else:
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except (UnidentifiedImageError, OSError):
            st.error("That file isn't a readable image. Upload a JPG, PNG or WEBP photo of the car.",
                     icon=":material/broken_image:")
            st.stop()
        try:
            with st.spinner("Detecting, classifying and verifying…"):
                res = analyze_cached(image_bytes, claim)
        except Exception as e:  # pragma: no cover
            st.error(f"Analysis failed ({e}). Try another photo or reload the page.", icon=":material/error:")
            st.stop()
        st.session_state.result = res

        if claim is not None:
            v = res["verdict"]
            conf = v["levels"][2]["prob"] * v["levels"][1]["prob"] if v["levels"] else res["confidence"]
            thr = eng.thr.match
        else:   # identify mode: the slider re-thresholds instantly from stored probabilities
            thr = st.session_state.get("id_thr", float(eng.thr.identify))
            v = identify(np.asarray(res["p_hybrid"]), hier, Thresholds(**{**eng.thr.as_dict(), "identify": thr}),
                         res["unknown"], res["issues"]).as_dict()
            conf = res["confidence"]
        kind, icon, label, colour = ui.STATUS.get(v["status"], ("info", ":material/info:", v["status"], "gray"))

        left, right = st.columns([1, 1.05], gap="medium")
        # ---------------------------------------------------------------- verdict card (the answer)
        with left.container(border=True, height="stretch"):
            if claim is not None:
                st.html(ui.plate(choice))
            if v["status"] == "UNKNOWN":
                getattr(st, kind)(f"**{label}** · not one of the 196 models this system was trained on", icon=icon)
            else:
                getattr(st, kind)(f"**{label}** · {v['headline']}", icon=icon)
            if res["unknown"] and res.get("open_set"):
                st.markdown("Closest post-2012 matches:")
                for t in res["open_set"]["top"]:
                    st.progress(min(1.0, t["prob"]), text=f"{t['label']} · {t['prob']:.0%}")
            else:
                st.markdown(v["detail"])
                if claim is not None and v["levels"]:
                    for lv in v["levels"]:
                        st.progress(max(0.0, min(1.0, lv["prob"])),
                                    text=f"{ui.LEVEL_ICON[lv['state']]} **{lv['level']}** · {lv['claimed']} · "
                                         f"{lv['prob']:.0%}")
                elif v["status"] == "LOW_CONFIDENCE":
                    for c in v["candidates"]:
                        st.progress(min(1.0, c["prob"]), text=f"{c['label']} · {c['prob']:.0%}")
                if v["status"] not in ("UNKNOWN", "RETAKE"):
                    what = "Evidence for the registered vehicle" if claim is not None else "Confidence"
                    st.progress(max(0.0, min(1.0, conf)), text=f"{what} · {conf:.0%} (needs {thr:.0%} to commit)")
            ov = res.get("other_vehicles") or []
            if ov:
                st.markdown(f":material/directions_car: **Other cars in this photo** ({len(ov)}, numbered on the image)")
                st.dataframe(pd.DataFrame([{
                    "#": i + 2, "Identification": "Not in records" if o["unknown"] else o["label"],
                    "Confidence": o["confidence"], "Status": ui.STATUS.get(o["status"], ("", "", o["status"], ""))[2]}
                    for i, o in enumerate(ov)]), hide_index=True, width="stretch",
                    column_config={"Confidence": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                                                 max_value=1)})
            if claim is None and not res["unknown"]:
                st.slider("Confidence threshold", 0.20, 0.95, float(eng.thr.identify), 0.01, key="id_thr",
                          help="Below this the app shows the top 3 instead of one answer.")
            if llm.available():
                key = f"{hashlib.sha1(image_bytes).hexdigest()}|{claim}"
                if res["unknown"]:
                    with st.spinner("Identifying the model…"):
                        op = opinion_cached(hashlib.sha1(image_bytes).hexdigest(), to_jpeg(res["images"]["crop"]),
                                            res["guess"])
                    if op:
                        st.markdown(":material/auto_awesome: **AI identification** "
                                    ":gray-badge[unverified]")
                        st.markdown(op)
                note = None
                if not res["unknown"]:      # for unknown cars the AI identification above says what matters
                    with st.spinner("Writing the officer note…"):
                        note = note_cached(key, res, hier.class_names[claim] if claim is not None else None)
                if note:
                    st.markdown(":material/description: **Officer note**")
                    st.caption(note)

        # ---------------------------------------------------------------- photo card (the evidence)
        with right.container(border=True, height="stretch"):
            view = st.segmented_control("View", [":material/image: Photo", ":material/whatshot: Heat-map"],
                                        default=":material/image: Photo", key="view", label_visibility="collapsed")
            if view and "Heat-map" in view and "gradcam" in res["images"]:
                st.image(res["images"]["gradcam"], width="stretch")
                st.caption("Warmer areas drove the prediction most (class activation map).")
            else:
                st.image(res["images"]["annotated"], width="stretch")
                st.caption(f"Detected vehicle · {res['box_source']} · score {res['det_score']:.2f} · "
                           f"{res['timings_ms']['total']:.0f} ms")

        # ---------------------------------------------------------------- look-alikes (known cars only)
        if not res["unknown"] and v["candidates"]:
            st.subheader("Look-alikes")
            st.caption("The most typical training photo of each top candidate, for a side-by-side check.")
            cols = st.columns(3)
            for col, cand in zip(cols, v["candidates"][:3]):
                with col.container(border=True, height="stretch"):
                    r = ref_image(cand["class_idx"])
                    if r:
                        st.image(r, width="stretch")
                    st.progress(min(1.0, cand["prob"]), text=f"**{cand['label']}** · {cand['prob']:.1%}")

# ================================================================== PIPELINE
with tab_pipe:
    res = st.session_state.get("result")
    if res is None:
        st.caption("Check a car first: every intermediate step of the pipeline for that photo appears here.")
    else:
        im, q = res["images"], res["quality"]
        with st.container(border=True):
            st.markdown(":material/tune: **1 · Input, quality check, adaptive enhancement**")
            a, b, c = st.columns(3)
            a.image(im["input"], caption="Input, longest side 640 px", width="stretch")
            b.image(im["crop"], caption=f"Region of interest ({res['box_source']})", width="stretch")
            c.image(im["enhanced"], caption="Enhancement: " + (", ".join(res["enhancement"]) or "none needed"),
                    width="stretch")
            k = st.columns(4)
            k[0].metric("Brightness", f"{q['brightness']:.0f}", border=True)
            k[1].metric("Contrast", f"{q['contrast']:.0f}", border=True)
            k[2].metric("Sharpness", f"{q['sharpness']:.0f}", border=True)
            k[3].metric("Issues", ", ".join(q["issues"]) or "None", border=True)
        with st.container(border=True):
            st.markdown(":material/polyline: **2 · Edges, contours, classical region of interest**")
            a, b, c = st.columns(3)
            a.image(im["edges"], caption="Auto-Canny edges (median-based thresholds)", width="stretch")
            b.image(im["contours"], caption="Largest contour, convex hull, classical box", width="stretch")
            c.image(im["classical_crop"], caption="Classical crop, no learning involved", width="stretch")
        with st.container(border=True):
            st.markdown(":material/texture: **3 · Texture and colour descriptors**")
            a, b, c = st.columns(3)
            a.image(im["hog"], caption="HOG gradient orientations (shape)", width="stretch")
            b.image(im["lbp"], caption="LBP texture codes", width="stretch")
            with c:
                st.plotly_chart(hsv_figure(res["histograms"]), config={"displayModeBar": False})
                st.caption("HSV colour histograms of the crop")
        st.caption("Stage timings for this request (ms)")
        st.dataframe([res["timings_ms"]], hide_index=True)

# ================================================================== COMPARE
with tab_cmp:
    res = st.session_state.get("result")
    if res is None:
        st.caption("Check a car first: this tab then compares the classical, deep and hybrid pipelines on that photo.")
    else:
        spec = (("classical", "Classical", "Contour ROI → HOG · LBP · GLCM · colour · geometry → SVM"),
                ("deep", "Deep", "YOLO11n crop → ConvNeXt-Tiny"),
                ("hybrid", "Hybrid (ours)", "YOLO + quality gate + enhancement + calibration + unknown check"))
        cols = st.columns(3)
        for col, (k, title, desc) in zip(cols, spec):
            with col.container(border=True, height="stretch"):
                st.markdown(f"**{title}**")
                st.caption(desc)
                df = pd.DataFrame([{"Candidate": c["label"], "Probability": c["prob"]} for c in res["methods"][k]])
                st.dataframe(df, hide_index=True, width="stretch", column_config={
                    "Probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        agree = len({res["methods"][k][0]["label"] for k in ("classical", "deep", "hybrid")})
        st.caption({1: "All three agree.", 2: "Two of three agree.", 3: "All three disagree."}[agree]
                   + " Dataset-level accuracy is on the Test results tab.")

with tab_res:
    render_results()

with tab_about:
    p = ROOT / "docs" / "about.md"
    st.markdown(p.read_text(encoding="utf-8") if p.exists() else "")
