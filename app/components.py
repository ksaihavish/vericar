"""VeriCar design system: an 'inspection report' look - paper, ink, hairlines, a number plate
for the claimed vehicle and a rubber-stamp verdict. Plain HTML/CSS; the verdict card is a
small component animated with GSAP (disabled under prefers-reduced-motion)."""
from __future__ import annotations

import html

# ---------------------------------------------------------------- tokens
PAPER, PANEL, INK, INK2, MUTED, RULE = "#F3F0E8", "#FBFAF6", "#1B1814", "#55504A", "#8A8478", "#DCD6C8"
PLATE = "#F2C230"
# Status colours (dataviz status set, darkened for text on paper) - always icon + label, never colour alone.
STATUS = {
    "MATCH": ("#17803A", "✓", "Cleared"),
    "CONFIDENT": ("#17803A", "✓", "Identified"),
    "PARTIAL": ("#A86A00", "◐", "Check documents"),
    "INSPECT": ("#B4511F", "?", "Manual check"),
    "LOW_CONFIDENCE": ("#A86A00", "?", "Low confidence"),
    "MISMATCH": ("#B42318", "✕", "Possible clone"),
    "UNKNOWN": ("#5B4BB7", "◌", "Outside knowledge"),
    "RETAKE": ("#5F6B7A", "↻", "Retake photo"),
}
STAMP_TEXT = {"MATCH": "CLEARED", "CONFIDENT": "IDENTIFIED", "PARTIAL": "CHECK YEAR", "INSPECT": "MANUAL CHECK",
              "LOW_CONFIDENCE": "UNSURE", "MISMATCH": "POSSIBLE CLONE", "UNKNOWN": "NOT IN RECORDS",
              "RETAKE": "RETAKE PHOTO"}
# Method identity (dataviz categorical slots 1-3, light steps; validated all-pairs on #FBFAF6).
METHOD = {"deep": "#2a78d6", "classical": "#eb6834", "hybrid": "#1baf7a"}
GRID, AXIS = "#E7E2D6", "#C9C2B3"

FONTS = ("https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Instrument+Sans:wght@400;500;600;700"
         "&family=JetBrains+Mono:wght@400;600&family=Barlow+Condensed:wght@600;700&display=swap")

CSS = f"""
<style>
@import url('{FONTS}');
:root {{ color-scheme: light; }}
html, body, .stApp, [class*="css"], button, input, textarea, select, label {{
  font-family: 'Instrument Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1240px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; text-wrap: balance; }}
* {{ -webkit-tap-highlight-color: rgba(27,24,20,.08); }}
button, [role="tab"], [role="radio"] {{ touch-action: manipulation; }}
:focus-visible {{ outline: 2px solid {INK} !important; outline-offset: 2px; }}

/* masthead */
.vc-mast {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1.5rem; flex-wrap:wrap;
  padding-bottom:.9rem; border-bottom: 3px double {INK}; margin-bottom: .4rem; }}
.vc-word {{ font-family:'Instrument Serif', Georgia, serif; font-size: 2.9rem; line-height: .9; letter-spacing:-.02em; }}
.vc-word em {{ font-style: italic; color: {INK2}; }}
.vc-kicker {{ font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; color:{INK2}; margin-top:.55rem; }}
.vc-meta {{ display:grid; grid-template-columns:auto auto; gap:.15rem 1rem; font-family:'JetBrains Mono', monospace;
  font-size:.72rem; color:{INK2}; }}
.vc-meta b {{ color:{INK}; font-weight:600; }}

/* tabs */
div[data-testid="stTabs"] [role="tablist"] {{ gap: 1.6rem; border-bottom: 1px solid {RULE}; }}
div[data-testid="stTabs"] button[role="tab"] {{ padding: .55rem 0; }}
div[data-testid="stTabs"] button[role="tab"] p {{ font-size:.78rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase; }}

/* section labels */
.vc-sec {{ display:flex; align-items:baseline; gap:.6rem; margin: 1.1rem 0 .55rem; padding-top:.5rem; border-top:1px solid {RULE}; }}
.vc-sec .n {{ font-family:'JetBrains Mono', monospace; font-size:.72rem; color:{MUTED}; }}
.vc-sec .t {{ font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }}
.vc-sec .h {{ font-size:.82rem; color:{INK2}; margin-left:auto; text-align:right; }}

/* number plate */
.vc-plate {{ display:flex; align-items:stretch; border:2px solid {INK}; border-radius:7px; overflow:hidden;
  background:{PLATE}; box-shadow: 0 1px 0 rgba(0,0,0,.25); max-width: 520px; }}
.vc-plate .band {{ background:#1E3A8A; color:#F8F5EC; font:700 .7rem/1 'Barlow Condensed', sans-serif;
  letter-spacing:.08em; writing-mode: vertical-rl; transform: rotate(180deg); padding:.4rem .3rem; text-align:center; }}
.vc-plate .body {{ padding:.45rem .9rem .4rem; min-width:0; }}
.vc-plate .lab {{ font: 600 .62rem/1 'JetBrains Mono', monospace; letter-spacing:.14em; color:#3B3112; text-transform:uppercase; }}
.vc-plate .txt {{ font: 700 1.75rem/1.05 'Barlow Condensed', sans-serif; letter-spacing:.04em; color:#111;
  text-transform:uppercase; overflow-wrap:anywhere; }}

/* panels, captions, notes */
.vc-panel {{ background:{PANEL}; border:1px solid {RULE}; border-radius:4px; padding:1rem 1.1rem; }}
.vc-empty {{ border:1px dashed {AXIS}; border-radius:4px; padding:1.4rem 1.2rem; color:{INK2}; background:{PANEL}; }}
.vc-empty b {{ color:{INK}; }}
.vc-cap {{ font-family:'JetBrains Mono', monospace; font-size:.72rem; color:{INK2}; margin-top:.3rem; }}
.vc-note {{ border-left:3px solid {INK}; padding:.65rem .9rem; background:{PANEL}; font-size:.95rem; line-height:1.55; }}
.vc-note .k {{ font: 600 .66rem/1 'JetBrains Mono', monospace; letter-spacing:.14em; text-transform:uppercase; color:{INK2};
  display:block; margin-bottom:.35rem; }}
.vc-note.ai {{ border-left-color:#5B4BB7; }}

/* look-alike cards */
.vc-cand {{ border-top:2px solid {INK}; padding-top:.45rem; }}
.vc-cand .t {{ font-weight:600; font-size:.88rem; line-height:1.25; min-height:2.2em; }}
.vc-cand .p {{ font-family:'JetBrains Mono', monospace; font-size:.8rem; color:{INK2}; font-variant-numeric: tabular-nums; }}
.vc-cand .bar {{ height:4px; background:{GRID}; margin-top:.3rem; }}
.vc-cand .bar i {{ display:block; height:100%; background:{INK}; }}

/* kpi + method lists */
.vc-kpi {{ border-top:2px solid {INK}; padding:.5rem 0 .2rem; }}
.vc-kpi .l {{ font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:{INK2}; }}
.vc-kpi .v {{ font-family:'Instrument Serif', Georgia, serif; font-size:2.1rem; line-height:1.05; font-variant-numeric: tabular-nums; }}
.vc-list {{ list-style:none; padding:0; margin:.4rem 0 0; }}
.vc-list li {{ display:grid; grid-template-columns: 1fr auto; gap:.2rem .6rem; padding:.38rem 0; border-bottom:1px solid {GRID}; font-size:.86rem; }}
.vc-list li .b {{ grid-column: 1 / -1; height:4px; background:{GRID}; }}
.vc-list li .b i {{ display:block; height:100%; }}
.vc-list li .p {{ font-family:'JetBrains Mono', monospace; color:{INK2}; font-variant-numeric: tabular-nums; }}
.vc-mhead {{ border-top:3px solid var(--c); padding-top:.5rem; }}
.vc-mhead .k {{ font: 600 .68rem/1 'JetBrains Mono', monospace; letter-spacing:.14em; text-transform:uppercase; color:{INK2}; }}
.vc-mhead .top {{ font-family:'Instrument Serif', Georgia, serif; font-size:1.45rem; line-height:1.15; margin:.3rem 0 .2rem; }}
.vc-mhead .d {{ font-size:.8rem; color:{INK2}; }}

/* streamlit widgets on paper */
[data-testid="stFileUploaderDropzone"] {{ background:{PANEL}; border:1px dashed {AXIS}; border-radius:4px; }}
div[data-baseweb="select"] > div {{ background:{PANEL}; border-color:{RULE}; border-radius:4px; }}
[data-testid="stImageCaption"], [data-testid="caption"] {{ font-family:'JetBrains Mono', monospace; font-size:.72rem; color:{INK2}; }}
.stButton > button {{ border-radius:3px; border:1px solid {INK}; background:{PANEL}; color:{INK}; font-weight:600;
  transition: background-color .15s ease, color .15s ease; }}
.stButton > button:hover, .stButton > button[data-testid="stBaseButton-primary"] {{ background:{INK}; color:{PANEL};
  border-color:{INK}; }}
/* demo cases: uniform cards; a swipeable strip on phones */
.st-key-demos [data-testid="stImage"] img {{ border-radius:3px; }}
.st-key-demos [data-testid="stElementToolbar"] {{ display:none; }}
.st-key-demos .stButton > button {{ font-size:.8rem; padding:.25rem .4rem; min-height:2.1rem; white-space:nowrap; }}
@media (max-width: 640px) {{
  .vc-word {{ font-size:2.2rem; }} .vc-plate .txt {{ font-size:1.35rem; }} .vc-meta {{ display:none; }}
  .st-key-demos [data-testid="stHorizontalBlock"] {{ flex-wrap:nowrap !important; overflow-x:auto; gap:.6rem;
    scroll-snap-type:x mandatory; padding-bottom:.4rem; }}
  .st-key-demos [data-testid="stColumn"] {{ min-width:132px !important; width:132px !important; flex:0 0 132px !important;
    scroll-snap-align:start; }}
}}
</style>
"""


def esc(s) -> str:
    return html.escape(str(s))


def masthead() -> str:
    meta = (("Detector", "YOLO11n · fine-tuned"), ("Classifier", "ConvNeXt-Tiny · ONNX"),
            ("Classes", "196 make · model · year"), ("Records", "Stanford Cars ≤ 2012"))
    cells = "".join(f"<span>{esc(k)}</span><b>{esc(v)}</b>" for k, v in meta)
    return (f'<div class="vc-mast"><div><div class="vc-word" translate="no">Veri<em>Car</em></div>'
            f'<div class="vc-kicker">Vehicle identity check · plate-clone screening</div></div>'
            f'<div class="vc-meta">{cells}</div></div>')


def section(n: str, title: str, hint: str = "") -> str:
    return (f'<div class="vc-sec"><span class="n">{esc(n)}</span><span class="t">{esc(title)}</span>'
            f'<span class="h">{esc(hint)}</span></div>')


def plate(text: str, label: str = "Registered vehicle") -> str:
    return (f'<div class="vc-plate" role="img" aria-label="{esc(label)}: {esc(text)}"><div class="band">IND</div>'
            f'<div class="body"><div class="lab">{esc(label)}</div><div class="txt" translate="no">{esc(text)}</div></div></div>')


def kpi(label: str, value: str) -> str:
    return f'<div class="vc-kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div></div>'


def candidate(label: str, prob: float) -> str:
    return (f'<div class="vc-cand"><div class="t">{esc(label)}</div><div class="p">{prob:.1%}</div>'
            f'<div class="bar"><i style="width:{max(0.0, min(1.0, prob)) * 100:.1f}%"></i></div></div>')


def method_block(key: str, title: str, desc: str, cands: list[dict]) -> str:
    col = METHOD[key]
    items = "".join(
        f'<li><span>{esc(c["label"])}</span><span class="p">{c["prob"]:.1%}</span>'
        f'<span class="b"><i style="width:{c["prob"] * 100:.1f}%;background:{col}"></i></span></li>' for c in cands)
    return (f'<div class="vc-mhead" style="--c:{col}"><div class="k">{esc(title)}</div>'
            f'<div class="top">{esc(cands[0]["label"])}</div><div class="d">{esc(desc)}</div></div>'
            f'<ul class="vc-list">{items}</ul>')


def verdict_component(v: dict, conf: float, threshold: float, claim_mode: bool) -> tuple[str, int]:  # noqa: C901
    """Self-contained HTML (iframe) for the verdict: stamp + headline + level bars, animated with GSAP."""
    col, icon, tag = STATUS.get(v["status"], (INK, "•", v["status"]))
    stamp = STAMP_TEXT.get(v["status"], v["status"])
    rows = ""
    if claim_mode and v.get("levels"):
        for lv in v["levels"]:
            c = {"ok": "#17803A", "unsure": "#A86A00", "fail": "#B42318"}[lv["state"]]
            m = {"ok": "✓", "unsure": "?", "fail": "✕"}[lv["state"]]
            p = max(0.0, min(1.0, lv["prob"]))
            rows += (f'<div class="row"><span class="lv">{esc(lv["level"])}</span><span class="cl">'
                     f'<span class="mk" style="color:{c}" aria-hidden="true">{m}</span> {esc(lv["claimed"])}</span>'
                     f'<span class="pc" data-v="{p * 100:.1f}">{p * 100:.0f}%</span>'
                     f'<span class="bar"><i style="background:{c}" data-w="{p:.4f}"></i></span></div>')
        h = 388
    else:
        h = 282
    h += 22 * max(0, len(v["detail"]) // 90 - 1)          # long explanations wrap onto extra lines
    mlabel = "Evidence for the registered vehicle" if claim_mode else "Confidence"
    meter = (f'<div class="meter"><div class="ml"><span>{mlabel}</span><span class="pc" data-v="{conf * 100:.1f}">'
             f'{conf * 100:.0f}%</span></div><div class="track"><i style="background:{col}" data-w="{conf:.4f}"></i>'
             f'<b style="left:{threshold * 100:.1f}%" title="threshold"></b></div>'
             f'<div class="mt">Threshold {threshold:.0%} · below it the app will not commit</div></div>')
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="{FONTS}" rel="stylesheet">
<style>
:root {{ color-scheme: light; }}
body {{ margin:0; background:transparent; color:{INK}; font-family:'Instrument Sans',system-ui,sans-serif; }}
.card {{ background:{PANEL}; border:1px solid {RULE}; border-radius:4px; padding:18px 20px; position:relative; overflow:hidden; }}
.stamp {{ display:inline-flex; gap:10px; align-items:center; border:3px double {col}; color:{col}; padding:6px 14px;
  font:700 22px/1 'Barlow Condensed',sans-serif; letter-spacing:.12em; transform:rotate(-3deg); transform-origin:left center;
  border-radius:3px; mix-blend-mode:multiply; }}
.stamp small {{ font:600 11px/1 'JetBrains Mono',monospace; letter-spacing:.1em; opacity:.85; }}
h2 {{ font-family:'Instrument Serif',Georgia,serif; font-weight:400; font-size:26px; line-height:1.15; margin:14px 0 6px; text-wrap:balance; }}
p {{ margin:0; color:{INK2}; font-size:14.5px; line-height:1.55; }}
.rows {{ margin-top:14px; border-top:1px solid {RULE}; }}
.row {{ display:grid; grid-template-columns:62px 1fr 52px; gap:2px 10px; align-items:center; padding:8px 0; border-bottom:1px solid #EEE9DE; }}
.lv {{ font:600 10.5px/1 'JetBrains Mono',monospace; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; }}
.cl {{ font-weight:600; font-size:14px; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.pc {{ font:600 13px/1 'JetBrains Mono',monospace; text-align:right; font-variant-numeric:tabular-nums; }}
.bar {{ grid-column:2 / 4; height:4px; background:#E7E2D6; }}
.bar i, .track i {{ display:block; height:100%; transform-origin:left center; }}
.bar i {{ width:100%; }}
.meter {{ margin-top:14px; }}
.ml {{ display:flex; justify-content:space-between; font:600 10.5px/1 'JetBrains Mono',monospace; letter-spacing:.12em;
  text-transform:uppercase; color:{MUTED}; margin-bottom:6px; }}
.ml .pc {{ color:{INK}; font-size:13px; letter-spacing:0; }}
.track {{ position:relative; height:8px; background:#E7E2D6; }}
.track i {{ width:100%; }}
.track b {{ position:absolute; top:-4px; width:2px; height:16px; background:{INK}; }}
.mt {{ font:11px/1.4 'JetBrains Mono',monospace; color:{MUTED}; margin-top:6px; }}
</style></head><body>
<div class="card" role="status" aria-live="polite">
  <div class="stamp" id="stamp" aria-label="{esc(tag)}"><span aria-hidden="true">{icon}</span>{esc(stamp)}</div>
  <h2>{esc(v["headline"])}</h2>
  <p>{esc(v["detail"])}</p>
  {'<div class="rows">' + rows + '</div>' if rows else ''}
  {meter}
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script>
(function () {{
  var bars = document.querySelectorAll("[data-w]");
  bars.forEach(function (b) {{ b.style.transform = "scaleX(" + b.dataset.w + ")"; }});
  if (!window.gsap) return;                       // no network: static, still correct
  var mm = gsap.matchMedia();
  mm.add("(prefers-reduced-motion: no-preference)", function () {{
    var tl = gsap.timeline({{ defaults: {{ ease: "power3.out" }} }});
    tl.from("#stamp", {{ scale: 1.7, autoAlpha: 0, rotation: -12, duration: .45, ease: "back.out(2.2)" }})
      .from("h2, p", {{ y: 8, autoAlpha: 0, duration: .35, stagger: .06 }}, "-=.15")
      .from(".row", {{ y: 6, autoAlpha: 0, duration: .3, stagger: .07 }}, "-=.2")
      .addLabel("bars")
      .fromTo("[data-w]", {{ scaleX: 0 }}, {{ scaleX: function (i, el) {{ return +el.dataset.w; }}, duration: .7,
               stagger: .07 }}, "bars");
    document.querySelectorAll(".pc").forEach(function (el, i) {{      // count numbers up with the bars
      var o = {{ v: 0 }}, end = +el.dataset.v;
      tl.to(o, {{ v: end, duration: .7, ease: "power2.out",
                 onUpdate: function () {{ el.textContent = Math.round(o.v) + "%"; }} }}, "bars+=" + (i * .07));
    }});
  }});
}})();
</script></body></html>"""
    return doc, h
