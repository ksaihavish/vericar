"""Results tab: every evaluation table and chart, read from results/*.json.

Chart rules (dataviz skill): one y-axis per chart, method colours fixed per entity
(deep = slot 1, classical = slot 2, hybrid = slot 3), legends for >= 2 series,
recessive grid, hover tooltips on every mark, a table next to every chart.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import components as ui

RESULTS = Path(__file__).resolve().parents[1] / "results"
STATUS_COL = {"MATCH": ("#17803A", "✓"), "PARTIAL": ("#D9A21B", "◐"), "INSPECT": ("#C2622D", "?"),
              "MISMATCH": ("#B42318", "✕"), "UNKNOWN": ("#6250d6", "◌"), "RETAKE": ("#8A8478", "↻")}
MCOL = {"Classical": ui.METHOD["classical"], "Deep": ui.METHOD["deep"], "Hybrid (ours)": ui.METHOD["hybrid"]}
CLAIM_COL = {"Legit claim": "#2a78d6", "Hard clone": "#eb6834", "Easy clone": "#1baf7a"}


@st.cache_data
def _load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def _layout(fig: go.Figure, height=300, y_title="", x_title="", pct_y=True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=30, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ui.INK2, size=12), hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=ui.INK2)),
        xaxis=dict(title=x_title, gridcolor=ui.GRID, zeroline=False, linecolor=ui.AXIS, tickfont=dict(color=ui.MUTED)),
        yaxis=dict(title=y_title, gridcolor=ui.GRID, zeroline=False, linecolor=ui.AXIS, tickfont=dict(color=ui.MUTED),
                   tickformat=".0%" if pct_y else None),
        hoverlabel=dict(bgcolor=ui.PANEL, bordercolor=ui.RULE, font_color=ui.INK), barcornerradius=4,
    )
    return fig


def hsv_figure(h: dict) -> go.Figure:
    """Hue / saturation / value histograms of the crop (Pipeline tab)."""
    fig = go.Figure()
    for name, col in (("hue", "#2a78d6"), ("saturation", "#eb6834"), ("value", "#1baf7a")):
        v = h[name]
        fig.add_scatter(y=v / (v.sum() + 1e-9), mode="lines", name=name, line=dict(color=col, width=2),
                        hovertemplate=f"{name} bin %{{x}}: %{{y:.1%}}<extra></extra>")
    fig = _layout(fig, 230, "", "bin")
    fig.update_yaxes(tickformat=".0%")
    return fig


def _short(name: str) -> str:
    return "Classical" if name.startswith("Classical") else "Deep" if name.startswith("Deep") else \
        "Hybrid (ours)" if name.startswith("Hybrid") else "GT-crop reference"


def _pct(x):
    return f"{x:.1%}" if isinstance(x, (int, float)) else "–"


def render_results() -> None:
    final = _load("final_test.json")
    if final is None:
        st.info("Final test-set results have not been generated yet.")
        return
    val = _load("val_fitting.json") or {}
    opt = _load("optimization_val.json") or {}
    hist = _load("training_history.json") or []
    cls_val = _load("classical_val.json") or {}
    fails = _load("failures.json") or []
    hyb = final["methods"]["Hybrid (ours)"]
    det = final["detection"]["YOLO11n fine-tuned (ours)"]
    ver = final["verification"]
    chosen = opt.get("chosen")
    st.caption(f"Official Stanford Cars **test set, {final['n_test']:,} images, evaluated once** after every "
               "parameter was frozen on the validation split. Robustness uses a fixed random 1,000-image subset.")

    k = st.columns(4)
    for col, (lab, v) in zip(k, (("Hybrid Top-1", _pct(hyb["top1"])), ("Hybrid Top-5", _pct(hyb["top5"])),
                                 ("Detector mAP@0.5", _pct(det["mAP50"])), ("Mean box IoU", f"{det['mean_iou']:.3f}"))):
        col.metric(lab, v, border=True)
    k = st.columns(4)
    lat = opt.get("rows", {}).get(chosen, {})
    for col, (lab, v) in zip(k, (("Hard-clone catch rate", _pct(ver["catch_rate_hard"])),
                                 ("False-alarm rate (legit)", _pct(ver["false_alarm_rate"])),
                                 ("Classifier size (shipped)", f"{lat.get('size_mb', 0):.0f} MB"),
                                 ("Classifier CPU latency", f"{lat.get('median_ms', 0):.0f} ms"))):
        col.metric(lab, v, border=True)

    t1, t2, t3, t4, t5, t6 = st.tabs(["Classical vs Deep vs Hybrid", "Verification", "Confidence",
                                      "Robustness", "Ablation & optimisation", "Failures & limits"])

    # ---------------------------------------------------------------- comparison
    with t1:
        rows = [{"Method": n, "Top-1": m["top1"], "Top-5": m["top5"], "Make acc.": m["make_acc"],
                 "Model acc.": m["model_acc"], "Year | model": m["year_given_model_acc"], "ECE": m["ece"]}
                for n, m in final["methods"].items()]
        df = pd.DataFrame(rows)
        c1, c2 = st.columns([1.1, 1])
        with c1:
            fig = go.Figure()
            for r in rows[:3]:
                s = _short(r["Method"])
                fig.add_bar(x=[s], y=[r["Top-1"]], name=s, marker_color=MCOL[s], width=0.55,
                            text=[_pct(r["Top-1"])], textposition="outside",
                            hovertemplate=f"{s}<br>Top-1 %{{y:.2%}}<extra></extra>")
            st.plotly_chart(_layout(fig, 300, "Top-1 accuracy").update_layout(showlegend=False, yaxis_range=[0, 1.08]),
                            config={"displayModeBar": False})
        with c2:
            st.markdown("**Where the mistakes are**: most are look-alikes inside the same make")
            eb = final["similar_classes"]["error_breakdown"]
            st.metric("Same model, other year", _pct(eb["same_model_other_year"]), border=True)
            st.metric("Same make, other model", _pct(eb["same_make_other_model"]), border=True)
            st.metric("Another make", _pct(eb["other_make"]), border=True)
        st.dataframe(df.style.format({c: "{:.1%}" for c in df.columns[1:-1]} | {"ECE": "{:.3f}"}), hide_index=True)
        st.markdown("**Detection**")
        ddf = pd.DataFrame([{"Detector": n, "mAP@0.5": d["mAP50"], "mAP@0.5:0.95": d["mAP50_95"], "Mean IoU": d["mean_iou"],
                             "IoU ≥ 0.5": d["iou_ge_0.5"], "Miss rate": d["miss_rate"]} for n, d in final["detection"].items()])
        st.dataframe(ddf.style.format({c: "{:.3f}" for c in ddf.columns[1:]}), hide_index=True)
        if cls_val:
            st.markdown("**Classical model selection and feature-family ablation (validation split)**")
            a = pd.DataFrame([{"Variant": k, "Top-1": v["top1"], "Top-5": v["top5"]} for k, v in
                              {**cls_val.get("model_selection_val", {}), **cls_val.get("feature_family_ablation_val", {})}.items()])
            st.dataframe(a.style.format({"Top-1": "{:.1%}", "Top-5": "{:.1%}"}), hide_index=True)
        conf = pd.DataFrame(final["similar_classes"]["top_confusions"])
        if len(conf):
            st.markdown("**Most-confused pairs (hybrid, test)**")
            st.dataframe(conf, hide_index=True)

    # ---------------------------------------------------------------- verification
    with t2:
        st.markdown("Every test image is checked against three claims: its **true** class (legit), the **most "
                    "similar other class** (hard clone, chosen from training embeddings) and a **random other make** "
                    "(easy clone). *Cleared* = verdict MATCH; anything else sends the car to a manual check.")
        sw = pd.DataFrame(ver["sweep"])
        fig = go.Figure()
        for col, name in (("legit", "Legit claim"), ("hard_clone", "Hard clone"), ("easy_clone", "Easy clone")):
            fig.add_scatter(x=sw["t"], y=sw[col], mode="lines", name=name, line=dict(color=CLAIM_COL[name], width=2),
                            hovertemplate=f"{name}<br>threshold %{{x:.2f}}<br>cleared %{{y:.1%}}<extra></extra>")
        t_star = final["frozen_params"]["thresholds"]["match"]
        fig.add_vline(x=t_star, line=dict(color=ui.MUTED, dash="dot", width=1),
                      annotation_text=f"chosen on val: {t_star:.2f}", annotation_font_color=ui.INK2)
        st.plotly_chart(_layout(fig, 330, "Share of claims cleared (MATCH)", "MATCH threshold"),
                        config={"displayModeBar": False})
        order = ["MATCH", "PARTIAL", "INSPECT", "MISMATCH", "UNKNOWN", "RETAKE"]
        fig = go.Figure()
        for s in order:
            colr, icon = STATUS_COL[s]
            fig.add_bar(y=["Legit claim", "Hard clone", "Easy clone"],
                        x=[ver["legit"][s], ver["hard_clone"][s], ver["easy_clone"][s]], orientation="h",
                        name=f"{icon} {s}", marker=dict(color=colr, line=dict(color=ui.PANEL, width=2)),
                        hovertemplate=f"{s}: %{{x:.1%}}<extra></extra>")
        fig.update_layout(barmode="stack")
        st.plotly_chart(_layout(fig, 250, "", "Share of claims", pct_y=False).update_xaxes(tickformat=".0%"),
                        config={"displayModeBar": False})
        d = ver.get("deep_only_same_thresholds", {})
        st.dataframe(pd.DataFrame([
            {"System": "Hybrid (ours)", "Hard-clone catch": ver["catch_rate_hard"], "Easy-clone catch": ver["catch_rate_easy"],
             "False alarms": ver["false_alarm_rate"]},
            {"System": "Deep only, same thresholds", "Hard-clone catch": d.get("catch_rate_hard"),
             "Easy-clone catch": d.get("catch_rate_easy"), "False alarms": d.get("false_alarm_rate")}]
        ).style.format({c: "{:.1%}" for c in ("Hard-clone catch", "Easy-clone catch", "False alarms")}), hide_index=True)
        ud = final.get("unknown_detection", {})
        if "probe_n" in ud:
            st.markdown(f"**Cars outside the dataset:** {ud['probe_flag_rate']:.0%} of {ud['probe_n']} modern-car photos "
                        f"were flagged UNKNOWN (AUROC {ud['auroc_knn']:.3f} vs {ud['auroc_msp']:.3f} for plain softmax "
                        f"confidence), while {ud['test_flag_rate']:.1%} of in-dataset test images were.")
        else:
            st.markdown(f"**Unknown-car check:** {ud.get('test_flag_rate', 0):.1%} of in-dataset test images flagged "
                        "(threshold = 95th percentile on validation).")

    # ---------------------------------------------------------------- confidence
    with t3:
        c1, c2 = st.columns(2)
        conf_ = final["confidence"]
        with c1:
            fig = go.Figure()
            for key, name in (("coverage_deep", "Deep"), ("coverage_hybrid", "Hybrid (ours)")):
                cv = pd.DataFrame(conf_[key]).dropna()
                fig.add_scatter(x=cv["coverage"], y=cv["accuracy"], mode="lines+markers", name=name,
                                line=dict(color=MCOL[name], width=2), marker=dict(size=8),
                                customdata=cv["threshold"],
                                hovertemplate=f"{name}<br>threshold %{{customdata:.2f}}<br>coverage %{{x:.1%}}"
                                              f"<br>accuracy %{{y:.1%}}<extra></extra>")
            fig.update_xaxes(tickformat=".0%", autorange="reversed")
            st.plotly_chart(_layout(fig, 320, "Accuracy of answered", "Coverage (share answered)"),
                            config={"displayModeBar": False})
            a = conf_["at_identify_threshold"]
            st.caption(f"At the app's default threshold ({conf_['identify_threshold']:.2f}) the hybrid answers "
                       f"{a['coverage']:.0%} of images at {a['accuracy']:.1%} accuracy; for the rest it shows the top 3, "
                       f"which contain the right class {conf_['top3_acc_when_low_confidence']:.0%} of the time.")
        with c2:
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                            line=dict(color=ui.MUTED, dash="dot", width=1), hoverinfo="skip")
            for key, name in (("reliability_deep", "Deep"), ("reliability_hybrid", "Hybrid (ours)")):
                rb = pd.DataFrame(conf_[key])
                fig.add_scatter(x=rb["conf"], y=rb["acc"], mode="lines+markers", name=name,
                                line=dict(color=MCOL[name], width=2), marker=dict(size=8), customdata=rb["n"],
                                hovertemplate=f"{name}<br>confidence %{{x:.2f}}<br>accuracy %{{y:.1%}}<br>n=%{{customdata}}<extra></extra>")
            st.plotly_chart(_layout(fig, 320, "Observed accuracy", "Predicted confidence").update_xaxes(tickformat=".0%"),
                            config={"displayModeBar": False})
            st.caption(f"Temperature scaling (T = {final['frozen_params']['temperature']:.2f}, fitted on validation). "
                       f"ECE deep {final['methods']['Deep (YOLO crop + ConvNeXt-T)']['ece']:.3f} → hybrid {hyb['ece']:.3f}.")
        if hist:
            h = pd.DataFrame(hist)
            h["step"] = range(1, len(h) + 1)
            fig = go.Figure()
            for col, name, colr, dash in (("train_top1_aug", "Train (augmented batches)", "#2a78d6", "solid"),
                                          ("train_top1_clean", "Train (clean subset)", "#2a78d6", "dot"),
                                          ("val_top1", "Validation", "#eb6834", "solid")):
                fig.add_scatter(x=h["step"], y=h[col], mode="lines+markers", name=name, line=dict(color=colr, width=2, dash=dash),
                                marker=dict(size=8), customdata=h["stage"],
                                hovertemplate=f"{name}<br>epoch %{{x}} (%{{customdata}})<br>%{{y:.1%}}<extra></extra>")
            st.markdown("**Training curves** - head-only stage, then top blocks unfrozen; best checkpoint by validation")
            st.plotly_chart(_layout(fig, 300, "Top-1", "Epoch"), config={"displayModeBar": False})

    # ---------------------------------------------------------------- robustness
    with t4:
        rb = final["robustness"]
        names = list(rb.keys())
        fig = go.Figure()
        for key, name in (("classical_top1", "Classical"), ("deep_top1", "Deep"), ("hybrid_top1", "Hybrid (ours)")):
            fig.add_scatter(y=names, x=[rb[n][key] for n in names], mode="markers", name=name,
                            marker=dict(color=MCOL[name], size=11, line=dict(color=ui.PANEL, width=2)),
                            hovertemplate=f"{name}<br>%{{y}}: %{{x:.1%}}<extra></extra>")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(_layout(fig, 460, "", "Top-1 accuracy", pct_y=False).update_xaxes(tickformat=".0%", range=[0, 1]),
                        config={"displayModeBar": False})
        rdf = pd.DataFrame([{"Condition": n, "Classical": r["classical_top1"], "Deep": r["deep_top1"],
                             "Hybrid": r["hybrid_top1"], "Hybrid RETAKE rate": r["hybrid_retake_rate"],
                             "Enhanced": r["enhanced_rate"], "Box IoU": r["det_mean_iou"]} for n, r in rb.items()])
        st.dataframe(rdf.style.format({c: "{:.1%}" for c in rdf.columns[1:-1]} | {"Box IoU": "{:.3f}"}), hide_index=True)
        vp = final["viewpoint_proxy"]
        st.markdown("**Viewing angle (proxy: box width / height)** - Stanford Cars has no pose labels")
        st.dataframe(pd.DataFrame([{"View": k, "n": v["n"], "Classical": v["classical_top1"], "Deep": v["deep_top1"],
                                    "Hybrid": v["hybrid_top1"]} for k, v in vp.items()]).style.format(
            {"Classical": "{:.1%}", "Deep": "{:.1%}", "Hybrid": "{:.1%}"}), hide_index=True)

    # ---------------------------------------------------------------- ablation / optimisation
    with t5:
        ab = final["ablation"]
        base = ab["Full hybrid"]
        adf = pd.DataFrame([{"Variant": n, "Top-1": a["top1"], "Δ Top-1 (pts)": (a["top1"] - base["top1"]) * 100,
                             "ECE": a["ece"], "Hard-clone catch": a["catch_rate_hard"],
                             "False alarms": a["false_alarm_rate"]} for n, a in ab.items()])
        st.markdown("**Ablation** - remove one component at a time (test set)")
        st.dataframe(adf.style.format({"Top-1": "{:.1%}", "Δ Top-1 (pts)": "{:+.2f}", "ECE": "{:.3f}",
                                       "Hard-clone catch": "{:.1%}", "False alarms": "{:.1%}"}), hide_index=True)
        if opt:
            st.markdown(f"**Optimisation** - accuracy on validation crops, latency with 2 CPU threads; shipped: `{chosen}`")
            odf = pd.DataFrame([{"Variant": k, "Size (MB)": v["size_mb"], "Latency (ms)": v["median_ms"],
                                 "Val Top-1": v.get("top1")} for k, v in opt["rows"].items()])
            st.dataframe(odf.style.format({"Size (MB)": "{:.1f}", "Latency (ms)": "{:.0f}", "Val Top-1": "{:.2%}"},
                                          na_rep="–"), hide_index=True)

    # ---------------------------------------------------------------- failures
    with t6:
        if not fails:
            st.info("Failure-case analysis will appear here.")
        for f in fails:
            st.markdown(f"**{f['title']}**: true *{f['true']}*, predicted *{f['pred']}* ({f['conf']:.0%})")
            p = RESULTS / "failures" / f["image"]
            if p.exists():
                st.image(str(p), width="stretch")
            st.markdown(f["explanation"])
            st.divider()
