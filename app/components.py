"""Presentation helpers for the Streamlit app (HTML/CSS snippets, plotly figures)."""
from __future__ import annotations

import html

import plotly.graph_objects as go

# Status colours (dataviz status palette) - always shown with an icon + label, never colour alone.
STATUS = {
    "MATCH": ("#0ca30c", "✓", "Cleared"),
    "CONFIDENT": ("#0ca30c", "✓", "Identified"),
    "PARTIAL": ("#fab219", "◐", "Check documents"),
    "INSPECT": ("#ec835a", "?", "Manual check"),
    "LOW_CONFIDENCE": ("#fab219", "?", "Low confidence"),
    "MISMATCH": ("#d03b3b", "✕", "Possible clone"),
    "UNKNOWN": ("#9085e9", "◌", "Outside knowledge"),
    "RETAKE": ("#898781", "↻", "Retake photo"),
}
# Method identity colours (categorical slots 1-3, dark steps; validated all-pairs on #0D1424).
METHOD = {"deep": "#3987e5", "classical": "#d95926", "hybrid": "#199e70"}
INK, INK2, MUTED, GRID = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.vc-hero { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  padding: 1.1rem 1.3rem; border:1px solid #1F2A44; border-radius:16px;
  background: radial-gradient(1200px 200px at 0% 0%, rgba(45,212,191,.12), transparent 60%), #0D1424; }
.vc-brand { font-weight:800; font-size:1.65rem; letter-spacing:-.03em; }
.vc-brand span { color:#2DD4BF; }
.vc-sub { color:#94A3B8; font-size:.92rem; margin-top:.15rem; }
.vc-chips { display:flex; gap:.4rem; flex-wrap:wrap; }
.vc-chip { font-size:.75rem; color:#CBD5E1; border:1px solid #243049; background:#0F172A; padding:.22rem .6rem; border-radius:999px; }
.vc-card { border:1px solid #1F2A44; border-radius:16px; padding:1.1rem 1.2rem; background:#0D1424; }
.vc-verdict { border-radius:16px; padding:1.1rem 1.25rem; border:1px solid var(--c); background:
  linear-gradient(135deg, color-mix(in srgb, var(--c) 18%, transparent), transparent 70%), #0D1424; }
.vc-badge { display:inline-flex; align-items:center; gap:.45rem; font-weight:700; font-size:.78rem; letter-spacing:.08em;
  color:#0A0F1C; background:var(--c); padding:.28rem .7rem; border-radius:999px; text-transform:uppercase; }
.vc-head { font-size:1.45rem; font-weight:750; margin:.6rem 0 .25rem; letter-spacing:-.02em; }
.vc-detail { color:#CBD5E1; font-size:.95rem; line-height:1.5; }
.vc-level { display:grid; grid-template-columns: 70px 1fr 64px; gap:.7rem; align-items:center; padding:.5rem 0;
  border-bottom:1px dashed #1F2A44; }
.vc-level:last-child { border-bottom:none; }
.vc-lname { color:#94A3B8; font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; }
.vc-lval { font-weight:600; font-size:.95rem; }
.vc-bar { height:7px; background:#1E293B; border-radius:99px; overflow:hidden; margin-top:.35rem; }
.vc-bar > div { height:100%; border-radius:99px; background:var(--c); }
.vc-pct { font-family:'JetBrains Mono', monospace; font-weight:700; text-align:right; color:var(--c); }
.vc-cand { border:1px solid #1F2A44; border-radius:14px; padding:.55rem; background:#0D1424; height:100%; }
.vc-cand .t { font-weight:650; font-size:.88rem; line-height:1.25; margin-top:.4rem; }
.vc-cand .p { font-family:'JetBrains Mono', monospace; color:#2DD4BF; font-weight:700; }
.vc-muted { color:#94A3B8; font-size:.85rem; }
.vc-kpi { border:1px solid #1F2A44; border-radius:14px; padding:.8rem 1rem; background:#0D1424; }
.vc-kpi .v { font-family:'JetBrains Mono', monospace; font-size:1.55rem; font-weight:700; color:#F1F5F9; }
.vc-kpi .l { color:#94A3B8; font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; }
.vc-note { border-left:3px solid #2DD4BF; padding:.6rem .9rem; background:#0F172A; border-radius:8px; color:#E2E8F0; }
.vc-ai { border-left:3px solid #A78BFA; }
div[data-testid="stTabs"] button p { font-weight:600; }
@media (max-width: 640px) { .vc-head { font-size:1.2rem; } .vc-brand { font-size:1.35rem; } }
</style>
"""


def esc(s) -> str:
    return html.escape(str(s))


def hero() -> str:
    chips = "".join(f'<span class="vc-chip">{c}</span>' for c in
                    ("Stanford Cars · 196 models", "YOLO11 + ConvNeXt", "Classical CV", "Strict by design"))
    return (f'<div class="vc-hero"><div><div class="vc-brand">Veri<span>Car</span></div>'
            f'<div class="vc-sub">Does this car match its registration? Fine-grained make · model · year verification '
            f'for spotting cloned plates.</div></div><div class="vc-chips">{chips}</div></div>')


def verdict_card(v: dict) -> str:
    col, icon, tag = STATUS.get(v["status"], ("#2DD4BF", "•", v["status"]))
    return (f'<div class="vc-verdict" style="--c:{col}"><span class="vc-badge">{icon} {esc(v["status"].replace("_", " "))}'
            f' · {esc(tag)}</span><div class="vc-head">{esc(v["headline"])}</div>'
            f'<div class="vc-detail">{esc(v["detail"])}</div></div>')


def level_rows(levels: list[dict]) -> str:
    rows = []
    for lv in levels:
        col = {"ok": "#22C55E", "unsure": "#F59E0B", "fail": "#EF4444"}[lv["state"]]
        mark = {"ok": "✓", "unsure": "?", "fail": "✕"}[lv["state"]]
        pct = max(0.0, min(1.0, lv["prob"]))
        rows.append(f'<div class="vc-level" style="--c:{col}"><div class="vc-lname">{esc(lv["level"])}</div>'
                    f'<div><div class="vc-lval">{mark} {esc(lv["claimed"])}</div><div class="vc-bar">'
                    f'<div style="width:{pct * 100:.1f}%"></div></div></div><div class="vc-pct">{pct:.0%}</div></div>')
    return '<div class="vc-card">' + "".join(rows) + "</div>"


def kpi(label: str, value: str) -> str:
    return f'<div class="vc-kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div></div>'


def gauge(prob: float, threshold: float, colour: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 34, "color": "#F1F5F9"}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#94A3B8"}},
               "bar": {"color": colour, "thickness": 0.28}, "bgcolor": "#111827", "borderwidth": 0,
               "threshold": {"line": {"color": "#F8FAFC", "width": 3}, "thickness": 0.8, "value": threshold * 100}},
        title={"text": "Confidence (white line = threshold)", "font": {"size": 13, "color": "#94A3B8"}}))
    fig.update_layout(height=210, margin=dict(l=34, r=40, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def topk_bars(cands: list[dict], colour: str, height: int = 230) -> go.Figure:
    names = [c["label"] for c in cands][::-1]
    vals = [c["prob"] * 100 for c in cands][::-1]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colour,
                           text=[f"{v:.1f}%" for v in vals], textposition="outside", cliponaxis=False))
    fig.update_layout(height=height, margin=dict(l=0, r=40, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(range=[0, 115], visible=False),
                      yaxis=dict(tickfont=dict(color="#CBD5E1", size=11)), font=dict(color="#CBD5E1"))
    return fig


def hist_fig(h: dict) -> go.Figure:
    fig = go.Figure()
    for name, col in (("hue", "#2DD4BF"), ("saturation", "#A78BFA"), ("value", "#F59E0B")):
        v = h[name]
        fig.add_trace(go.Scatter(y=v / (v.sum() + 1e-9), mode="lines", name=name, line=dict(color=col, width=2),
                                 fill="tozeroy", opacity=0.6))
    fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", font=dict(color="#CBD5E1")),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig
