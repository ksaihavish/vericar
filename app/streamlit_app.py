"""VeriCar Streamlit app - a thin UI over ``VeriCar().analyze``."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

import components as ui  # noqa: E402
from results_view import render_results  # noqa: E402
from vericar import llm  # noqa: E402
from vericar.config import ASSETS_DIR, MODELS_DIR  # noqa: E402

st.set_page_config(page_title="VeriCar · plate-clone verification", page_icon="🚘", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading models (first start only)…")
def engine():
    from vericar.pipeline import VeriCar
    return VeriCar()


@st.cache_data
def demo_items() -> list[dict]:
    p = ASSETS_DIR / "demo" / "demo.json"
    return json.loads(p.read_text()) if p.exists() else []


def ref_image(idx: int):
    p = ASSETS_DIR / "ref" / f"{idx:03d}.jpg"
    return str(p) if p.exists() else None


def to_jpeg(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ------------------------------------------------------------------ header
st.markdown(ui.hero(), unsafe_allow_html=True)
if not (MODELS_DIR / "hybrid.json").exists():
    st.error("Model files are missing. Run `python scripts/download_models.py` (the launchers do this for you).")
    st.stop()

try:
    eng = engine()
except Exception as e:  # pragma: no cover - shown to the user
    st.error(f"Could not load the models: {e}")
    st.stop()
hier = eng.hier

tab_verify, tab_pipe, tab_cmp, tab_res, tab_about = st.tabs(
    ["Verify", "Pipeline", "Classical vs Deep vs Hybrid", "Results", "About"])

# ------------------------------------------------------------------ input column
with tab_verify:
    left, right = st.columns([0.9, 1.4], gap="large")
    with left:
        mode = st.segmented_control("Mode", ["Verify a registration", "Identify a car"],
                                    default="Verify a registration", label_visibility="collapsed")
        mode = mode or "Verify a registration"
        claim = None
        demos = demo_items()
        if "demo" not in st.session_state:
            st.session_state.demo = None

        if mode == "Verify a registration":
            st.markdown("**Registered vehicle** <span class='vc-muted'>(from the plate lookup)</span>",
                        unsafe_allow_html=True)
            demo_claim = None
            if st.session_state.demo is not None and st.session_state.demo.get("claim"):
                demo_claim = hier.describe(hier.class_names.index(st.session_state.demo["claim"]))
            makes = list(hier.make_names)
            make = st.selectbox("Make", makes, index=makes.index(demo_claim["make"]) if demo_claim else 0)
            models_for = sorted({hier.model_names[hier.model_of[c]] for c in range(len(hier.class_names))
                                 if hier.make_names[hier.make_of[c]] == make})
            mi = models_for.index(demo_claim["model"]) if demo_claim and demo_claim["model"] in models_for else 0
            model_name = st.selectbox("Model", models_for, index=mi,
                                      format_func=lambda m: m.removeprefix(make).strip())
            classes = [c for c in range(len(hier.class_names)) if hier.model_names[hier.model_of[c]] == model_name]
            years = [int(hier.year_of[c]) for c in classes]
            yi = years.index(demo_claim["year"]) if demo_claim and demo_claim["year"] in years else 0
            year = st.selectbox("Year", years, index=yi)
            claim = classes[years.index(year)]
            id_thr = None
        else:
            id_thr = st.slider("Confidence threshold", 0.20, 0.95, float(eng.thr.identify), 0.01,
                               help="Below this the app refuses to commit to one label and shows the top 3.")

        src = st.radio("Photo source", ["Upload", "Camera", "Demo gallery"], horizontal=True,
                       label_visibility="collapsed")
        image_bytes = None
        if src == "Upload":
            f = st.file_uploader("Car photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
            if f is not None:
                image_bytes = f.getvalue()
                st.session_state.demo = None
        elif src == "Camera":
            f = st.camera_input("Take a photo", label_visibility="collapsed")
            if f is not None:
                image_bytes = f.getvalue()
        else:
            if not demos:
                st.info("No demo images bundled.")
            cols = st.columns(3)
            for i, d in enumerate(demos):
                with cols[i % 3]:
                    st.image(str(ASSETS_DIR / "demo" / d["file"]), width="stretch")
                    if st.button(d["title"], key=f"demo{i}", width="stretch"):
                        st.session_state.demo = d
                        st.rerun()
            if st.session_state.demo is not None:
                image_bytes = (ASSETS_DIR / "demo" / st.session_state.demo["file"]).read_bytes()
                st.caption(st.session_state.demo.get("note", ""))

        show_cam = st.toggle("Show where the model looked (Grad-CAM, ~2 s)", value=False)
        use_llm = st.toggle("AI officer note (Gemini)", value=llm.available(), disabled=not llm.available(),
                            help="Optional. Needs GEMINI_API_KEY. The AI never changes the verdict.")

    # ------------------------------------------------------------------ analysis
    with right:
        if image_bytes is None:
            st.markdown('<div class="vc-card"><b>How it works</b><br><span class="vc-muted">Pick the registered '
                        'make / model / year, then upload a photo. VeriCar detects the car, checks the claim level by '
                        'level and tells you to clear, check documents, or inspect. Anything short of a confident '
                        'match is sent to a manual check - by design.</span></div>', unsafe_allow_html=True)
        else:
            try:
                pil = Image.open(io.BytesIO(image_bytes))
                pil.load()
            except (UnidentifiedImageError, OSError):
                st.error("That file is not a readable image. Please upload a JPG, PNG or WEBP photo.")
                st.stop()
            try:
                with st.spinner("Detecting, classifying and verifying…"):
                    res = eng.analyze(pil, claim=claim, identify_threshold=id_thr, gradcam=show_cam)
            except Exception as e:  # pragma: no cover
                st.error(f"Analysis failed: {e}")
                st.stop()
            st.session_state.result = res
            v = res["verdict"]
            colour = ui.STATUS.get(v["status"], ("#2DD4BF",))[0]
            st.markdown(ui.verdict_card(v), unsafe_allow_html=True)
            st.write("")
            c1, c2 = st.columns([1.35, 1])
            with c1:
                st.image(res["images"]["annotated"], width="stretch",
                         caption=f"Box: {res['box_source']} · detector score {res['det_score']:.2f} · "
                                 f"{res['timings_ms']['total']:.0f} ms")
            with c2:
                if claim is not None and v["levels"]:
                    st.markdown(ui.level_rows(v["levels"]), unsafe_allow_html=True)
                    st.plotly_chart(ui.gauge(v["levels"][2]["prob"] * v["levels"][1]["prob"], eng.thr.match, colour),
                                    width="stretch", config={"displayModeBar": False})
                else:
                    st.plotly_chart(ui.gauge(res["confidence"], id_thr or eng.thr.identify, colour),
                                    width="stretch", config={"displayModeBar": False})
                    lv = res["levels"]
                    st.markdown(f"<span class='vc-muted'>Make</span> **{lv['make']['name']}** "
                                f"<span class='vc-muted'>{lv['make']['prob']:.0%}</span><br>"
                                f"<span class='vc-muted'>Model</span> **{lv['model']['name']}** "
                                f"<span class='vc-muted'>{lv['model']['prob']:.0%}</span>", unsafe_allow_html=True)

            if show_cam and "gradcam" in res["images"]:
                g1, g2 = st.columns(2)
                g1.image(res["images"]["crop"], caption="Detected car", width="stretch")
                g2.image(res["images"]["gradcam"], caption="Grad-CAM: evidence for the top prediction",
                         width="stretch")

            st.markdown("##### Look-alikes, side by side")
            st.caption("Your photo next to a reference training image of each candidate class.")
            cols = st.columns(4)
            cols[0].image(res["images"]["crop"], caption="Your photo", width="stretch")
            for col, cand in zip(cols[1:], v["candidates"][:3]):
                with col:
                    r = ref_image(cand["class_idx"])
                    if r:
                        st.image(r, width="stretch")
                    st.markdown(f'<div class="vc-cand"><div class="t">{ui.esc(cand["label"])}</div>'
                                f'<div class="p">{cand["prob"]:.1%}</div></div>', unsafe_allow_html=True)

            if use_llm:
                with st.spinner("Writing the officer note…"):
                    claim_label = hier.class_names[claim] if claim is not None else None
                    note = llm.officer_note(res, claim_label)
                if note:
                    st.markdown(f'<div class="vc-note"><b>Officer note</b> · AI-written from the verdict above<br>'
                                f'{ui.esc(note)}</div>', unsafe_allow_html=True)
                if res["unknown"]:
                    with st.spinner("Asking for a second opinion…"):
                        op = llm.second_opinion(to_jpeg(res["images"]["crop"]), res["guess"])
                    if op:
                        st.markdown(f'<div class="vc-note vc-ai"><b>AI second opinion · unverified</b><br>'
                                    f'{ui.esc(op)}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ pipeline tab
with tab_pipe:
    res = st.session_state.get("result")
    if res is None:
        st.info("Analyse a photo on the Verify tab to see every intermediate step here.")
    else:
        im, q = res["images"], res["quality"]
        st.markdown("##### 1 · Input, quality check and adaptive enhancement")
        a, b, c = st.columns(3)
        a.image(im["input"], caption="Input (resized to 640 px max side)", width="stretch")
        b.image(im["crop"], caption=f"ROI crop ({res['box_source']})", width="stretch")
        c.image(im["enhanced"], caption="After adaptive enhancement: " + (", ".join(res["enhancement"]) or "none needed"),
                width="stretch")
        k = st.columns(4)
        for col, (lab, val) in zip(k, (("Brightness", f"{q['brightness']:.0f}"), ("Contrast", f"{q['contrast']:.0f}"),
                                       ("Sharpness", f"{q['sharpness']:.0f}"), ("Issues", ", ".join(q["issues"]) or "none"))):
            col.markdown(ui.kpi(lab, val), unsafe_allow_html=True)
        st.markdown("##### 2 · Classical CV: edges, contours, ROI")
        a, b, c = st.columns(3)
        a.image(im["edges"], caption="Auto-Canny edges (median-based thresholds)", width="stretch")
        b.image(im["contours"], caption="Largest contour (cyan), convex hull (orange), classical ROI (red)",
                width="stretch")
        c.image(im["classical_crop"], caption="Classical ROI crop (no learning)", width="stretch")
        st.markdown("##### 3 · Texture and colour features")
        a, b, c = st.columns(3)
        a.image(im["hog"], caption="HOG: gradient orientations (shape)", width="stretch")
        b.image(im["lbp"], caption="LBP: local texture codes", width="stretch")
        with c:
            st.plotly_chart(ui.hist_fig(res["histograms"]), width="stretch", config={"displayModeBar": False})
            st.caption("HSV colour histograms of the car crop")
        st.markdown("##### 4 · Stage timings (ms)")
        st.dataframe([res["timings_ms"]], hide_index=True, width="stretch")

# ------------------------------------------------------------------ comparison tab
with tab_cmp:
    res = st.session_state.get("result")
    if res is None:
        st.info("Analyse a photo first - this tab compares the three approaches on that photo.")
    else:
        cols = st.columns(3)
        spec = (("classical", "Classical", ui.METHOD["classical"], "contour ROI → HOG/LBP/GLCM/colour/geometry → SVM"),
                ("deep", "Deep", ui.METHOD["deep"], "YOLO11 crop → ConvNeXt-Tiny"),
                ("hybrid", "Hybrid (ours)", ui.METHOD["hybrid"], "YOLO + quality gate + enhancement + calibrated fusion"))
        for col, (k, title, colour, desc) in zip(cols, spec):
            with col:
                top = res["methods"][k][0]
                st.markdown(f'<div class="vc-card" style="border-color:{colour}"><div class="vc-lname">{title}</div>'
                            f'<div class="vc-lval" style="margin-top:.3rem">{ui.esc(top["label"])}</div>'
                            f'<div class="vc-muted">{desc}</div></div>', unsafe_allow_html=True)
                st.plotly_chart(ui.topk_bars(res["methods"][k], colour), width="stretch",
                                config={"displayModeBar": False}, key=f"bars_{k}")
        agree = len({res["methods"][k][0]["label"] for k in ("classical", "deep", "hybrid")})
        st.caption({1: "All three agree.", 2: "Two of three agree.", 3: "All three disagree."}[agree]
                   + " Dataset-level accuracy of each approach is on the Results tab.")

with tab_res:
    render_results()

with tab_about:
    p = ROOT / "docs" / "about.md"
    st.markdown(p.read_text(encoding="utf-8") if p.exists() else "Documentation coming soon.")
