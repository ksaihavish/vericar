"""VeriCar Streamlit app - a thin UI over ``VeriCar().analyze``.

One flow on the main tab: 1 registered vehicle (optional) -> 2 photo -> 3 verdict -> 4 evidence.
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
import streamlit as st  # noqa: E402
import streamlit.components.v1 as stc  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

import components as ui  # noqa: E402
from results_view import hsv_figure, render_results  # noqa: E402
from vericar import llm  # noqa: E402
from vericar.config import ASSETS_DIR  # noqa: E402
from vericar.hybrid import Thresholds, identify  # noqa: E402

st.set_page_config(page_title="VeriCar · Vehicle identity check", page_icon=str(ASSETS_DIR / "favicon.png"),
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)
NO_CLAIM = "None (just identify the car)"


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading models (first start downloads ~220 MB)…")
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
def analyze_cached(image_bytes: bytes, claim, gradcam: bool = False) -> dict:
    """Streamlit re-runs the script on every click; the models run once per (photo, claim).
    Small cache + Grad-CAM on demand keep memory inside a free host's limit."""
    return engine().analyze(Image.open(io.BytesIO(image_bytes)), claim=claim, gradcam=gradcam)


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
    p = ASSETS_DIR / "ref" / f"{idx:03d}.jpg"
    return str(p) if p.exists() else None


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def pick_demo(d: dict):
    st.session_state.demo = d
    st.session_state.claim_select = d.get("claim") or NO_CLAIM
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1   # clear any uploaded file


# ------------------------------------------------------------------ header
html(ui.masthead())
try:
    eng = engine()
except Exception as e:  # pragma: no cover - shown to the user
    st.error(f"The models could not be loaded ({e}). Reload the page; if it persists, run "
             "`python scripts/download_models.py` locally.")
    st.stop()
hier = eng.hier
st.session_state.setdefault("demo", None)
st.session_state.setdefault("uploader_key", 0)

tab_check, tab_pipe, tab_cmp, tab_res, tab_about = st.tabs(
    ["Check a car", "Pipeline steps", "Classical · Deep · Hybrid", "Test results", "About"])

# ================================================================== CHECK A CAR
with tab_check:
    left, right = st.columns([0.9, 1.3], gap="large")
    with left:
        html(ui.section("1", "Registered vehicle", "what the plate lookup says"))
        options = [NO_CLAIM] + sorted(hier.class_names)
        choice = st.selectbox("Registered vehicle (type to search)", options, key="claim_select",
                              help="Pick the make, model and year on the registration. Leave it on 'None' to just "
                                   "identify the car.")
        claim = None if choice == NO_CLAIM else hier.class_names.index(choice)
        if claim is not None:
            html(ui.plate(choice))

        html(ui.section("2", "Photo of the car", "JPG · PNG · WEBP, up to 15 MB"))
        f = st.file_uploader("Upload a photo (on a phone you can take one)", type=["jpg", "jpeg", "png", "webp"],
                             key=f"up{st.session_state.uploader_key}")
        image_bytes = None
        if f is not None:
            image_bytes, st.session_state.demo = f.getvalue(), None
        elif st.session_state.demo is not None:
            image_bytes = (ASSETS_DIR / "demo" / st.session_state.demo["file"]).read_bytes()

        html('<div class="vc-cap" style="margin:1rem 0 .45rem">OR TRY A DEMO CASE</div>')
        demos = demo_items()
        with st.container(key="demos"):
            cols = st.columns(len(demos) if len(demos) <= 4 else 4)
            for i, d in enumerate(demos):
                with cols[i % len(cols)]:
                    thumb = ASSETS_DIR / "demo" / "thumbs" / d["file"]
                    st.image(str(thumb if thumb.exists() else ASSETS_DIR / "demo" / d["file"]), width="stretch", alt=d.get("note", d["title"]))
                    st.button(d["title"], key=f"demo{i}", width="stretch", on_click=pick_demo, args=(d,),
                              type="primary" if st.session_state.demo == d else "secondary")
        if st.session_state.demo is not None and f is None:
            html(f'<div class="vc-cap">Selected demo: {ui.esc(st.session_state.demo.get("note", ""))}</div>')

    with right:
        html(ui.section("3", "Verdict", "strict: anything short of a confident match → manual check"))
        if image_bytes is None:
            html('<div class="vc-empty"><b>Three steps.</b><br>1 · Choose the registered vehicle, or leave it on '
                 '“None” to just identify the car.<br>2 · Upload a photo, or click a demo case.<br>3 · Read the '
                 'verdict here: <b>cleared</b>, <b>check documents</b>, <b>manual check</b> or <b>possible '
                 'clone</b>. Cars newer than 2012 are reported as <b>not in records</b> instead of guessed.</div>')
        else:
            try:
                Image.open(io.BytesIO(image_bytes)).verify()
            except (UnidentifiedImageError, OSError):
                st.error("That file isn't a readable image. Upload a JPG, PNG or WEBP photo of the car.")
                st.stop()
            try:
                with st.spinner("Detecting, classifying and verifying…"):
                    res = analyze_cached(image_bytes, claim)
            except Exception as e:  # pragma: no cover
                st.error(f"Analysis failed ({e}). Try another photo or reload the page.")
                st.stop()
            st.session_state.result = res
            if claim is not None:
                v = res["verdict"]
                conf = v["levels"][2]["prob"] * v["levels"][1]["prob"] if v["levels"] else res["confidence"]
                thr = eng.thr.match
            else:   # identify mode: the threshold slider re-thresholds instantly, no model re-run
                thr = st.session_state.get("id_thr", float(eng.thr.identify))
                v = identify(np.asarray(res["p_hybrid"]), hier, Thresholds(**{**eng.thr.as_dict(), "identify": thr}),
                             res["unknown"], res["issues"]).as_dict()
                conf = res["confidence"]
            doc, h = ui.verdict_component(v, conf, thr, claim is not None)
            stc.html(doc, height=h)
            if claim is None:
                st.slider("Confidence threshold: below it, show the top 3 instead of one answer", 0.20, 0.95,
                          float(eng.thr.identify), 0.01, key="id_thr")
                if v["status"] in ("LOW_CONFIDENCE", "UNKNOWN"):
                    html("".join(ui.candidate(c["label"], c["prob"]) for c in v["candidates"]))

            if res["unknown"] and res.get("open_set"):
                os_ = res["open_set"]
                names = " · ".join(f"{ui.esc(t['label'])} <span class='p'>{t['prob']:.0%}</span>" for t in os_["top"])
                why = "; ".join(res.get("unknown_reasons", []))
                html(f'<div class="vc-note ai" style="margin-top:.8rem"><span class="k">Open-set fallback · OpenCLIP '
                     f'ViT-B/32 zero-shot · unverified</span>Flagged because it {ui.esc(why)}. Possible post-2012 '
                     f'matches: {names}. Use as a lead for the manual check, not as an identification.</div>')

            if llm.available():
                with st.spinner("Writing the officer note…"):
                    note = note_cached(f"{hashlib.sha1(image_bytes).hexdigest()}|{claim}", res,
                                       hier.class_names[claim] if claim is not None else None)
                if note:
                    html(f'<div class="vc-note" style="margin-top:.8rem"><span class="k">Officer note · '
                         f'AI-written from the verdict</span>{ui.esc(note)}</div>')
                if res["unknown"]:
                    with st.spinner("Asking for a second opinion…"):
                        op = opinion_cached(hashlib.sha1(image_bytes).hexdigest(), to_jpeg(res["images"]["crop"]),
                                            res["guess"])
                    if op:
                        html(f'<div class="vc-note ai" style="margin-top:.6rem"><span class="k">AI second opinion · '
                             f'unverified</span>{ui.esc(op)}</div>')

            html(ui.section("4", "Evidence", f"{res['timings_ms']['total']:.0f} ms"))
            key = f"{hashlib.sha1(image_bytes).hexdigest()}|{claim}"
            if st.session_state.get("cam_for") == key:
                with st.spinner("Computing where the model looked…"):
                    res = analyze_cached(image_bytes, claim, True)
            a, b = st.columns([1.5, 1])
            a.image(res["images"]["annotated"], width="stretch", alt="Photo with the detected vehicle boxed and labelled",
                    caption=f"Detected vehicle ({res['box_source']}, score {res['det_score']:.2f})")
            if "gradcam" not in res["images"]:
                b.button("Show where the model looked (Grad-CAM)", width="stretch",
                         on_click=lambda k=key: st.session_state.update(cam_for=k))
                b.markdown('<div class="vc-cap">Highlights the image regions that drove the prediction. ' 
                           'Takes a few seconds on the free server.</div>', unsafe_allow_html=True)
            else:
                b.image(res["images"]["gradcam"], width="stretch", caption="Where the model looked (class activation map)", alt="Heat map of the regions that drove the prediction")
            html('<div class="vc-cap" style="margin:.8rem 0 .4rem">LOOK-ALIKES: the most typical training photo '
                 'of each top candidate</div>')
            cols = st.columns(3)
            for col, cand in zip(cols, v["candidates"][:3]):
                with col:
                    r = ref_image(cand["class_idx"])
                    if r:
                        st.image(r, width="stretch", alt=f"Typical training photo of {cand['label']}")
                    html(ui.candidate(cand["label"], cand["prob"]))

# ================================================================== PIPELINE
with tab_pipe:
    res = st.session_state.get("result")
    if res is None:
        html('<div class="vc-empty"><b>Nothing analysed yet.</b> Check a car first; every intermediate step of '
             'the pipeline for that photo appears here.</div>')
    else:
        im, q = res["images"], res["quality"]
        html(ui.section("1", "Input, quality check, adaptive enhancement"))
        a, b, c = st.columns(3)
        a.image(im["input"], caption="Fig. 1 · Input, longest side 640 px", width="stretch")
        b.image(im["crop"], caption=f"Fig. 2 · Region of interest ({res['box_source']})", width="stretch")
        c.image(im["enhanced"], caption="Fig. 3 · Enhancement: " + (", ".join(res["enhancement"]) or "none needed"),
                width="stretch")
        k = st.columns(4)
        for col, (lab, val) in zip(k, (("Brightness", f"{q['brightness']:.0f}"), ("Contrast", f"{q['contrast']:.0f}"),
                                       ("Sharpness", f"{q['sharpness']:.0f}"),
                                       ("Issues", ", ".join(q["issues"]) or "none"))):
            col.markdown(ui.kpi(lab, val), unsafe_allow_html=True)
        html(ui.section("2", "Edges, contours, classical ROI"))
        a, b, c = st.columns(3)
        a.image(im["edges"], caption="Fig. 4 · Auto-Canny edges (median-based thresholds)", width="stretch")
        b.image(im["contours"], caption="Fig. 5 · Largest contour, convex hull, classical box", width="stretch")
        c.image(im["classical_crop"], caption="Fig. 6 · Classical ROI, no learning involved", width="stretch")
        html(ui.section("3", "Texture and colour descriptors"))
        a, b, c = st.columns(3)
        a.image(im["hog"], caption="Fig. 7 · HOG gradient orientations (shape)", width="stretch")
        b.image(im["lbp"], caption="Fig. 8 · LBP texture codes", width="stretch")
        with c:
            st.plotly_chart(hsv_figure(res["histograms"]), config={"displayModeBar": False})
            html('<div class="vc-cap">Fig. 9 · HSV colour histograms of the crop</div>')
        html(ui.section("4", "Stage timings", "milliseconds, this request"))
        st.dataframe([res["timings_ms"]], hide_index=True)

# ================================================================== COMPARE
with tab_cmp:
    res = st.session_state.get("result")
    if res is None:
        html('<div class="vc-empty"><b>Nothing analysed yet.</b> Check a car first; this tab then compares the '
             'classical, deep and hybrid pipelines on that photo.</div>')
    else:
        cols = st.columns(3, gap="large")
        spec = (("classical", "Classical", "contour ROI → HOG · LBP · GLCM · colour · geometry → SVM"),
                ("deep", "Deep", "YOLO11n crop → ConvNeXt-Tiny"),
                ("hybrid", "Hybrid (ours)", "YOLO + quality gate + enhancement + calibration + unknown check"))
        for col, (k, title, desc) in zip(cols, spec):
            col.markdown(ui.method_block(k, title, desc, res["methods"][k]), unsafe_allow_html=True)
        agree = len({res["methods"][k][0]["label"] for k in ("classical", "deep", "hybrid")})
        verdict = {1: "All three agree.", 2: "Two of three agree.", 3: "All three disagree."}[agree]
        html(f'<div class="vc-cap" style="margin-top:1rem">{verdict} Dataset-level accuracy is on the Test results '
             f'tab.</div>')

with tab_res:
    render_results()

with tab_about:
    p = ROOT / "docs" / "about.md"
    st.markdown(p.read_text(encoding="utf-8") if p.exists() else "")
