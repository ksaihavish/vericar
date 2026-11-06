"""VeriCar presentation helpers.

Design follows Streamlit's bundled design references: colours, fonts, radii and borders are set
in .streamlit/config.toml, and the layout uses native elements (bordered containers, status
callouts, badges, progress bars, metrics). The only custom HTML is the number plate, which is
content, not theming.
"""
from __future__ import annotations

import html

# verdict -> (callout kind, Material icon, short label, badge colour)
STATUS = {
    "MATCH": ("success", ":material/verified:", "Cleared", "green"),
    "CONFIDENT": ("success", ":material/check_circle:", "Identified", "green"),
    "PARTIAL": ("warning", ":material/rule:", "Check documents", "orange"),
    "INSPECT": ("warning", ":material/manage_search:", "Manual check", "orange"),
    "LOW_CONFIDENCE": ("warning", ":material/help:", "Unsure: top 3 shown", "orange"),
    "MISMATCH": ("error", ":material/gpp_bad:", "Possible clone", "red"),
    "UNKNOWN": ("info", ":material/travel_explore:", "Not in records", "violet"),
    "RETAKE": ("warning", ":material/photo_camera:", "Retake photo", "gray"),
}
LEVEL_ICON = {"ok": ":material/check_circle:", "unsure": ":material/help:", "fail": ":material/cancel:"}
# Method identity colours (dataviz categorical slots 1-3, light steps; validated all-pairs on the paper surface).
METHOD = {"deep": "#2a78d6", "classical": "#eb6834", "hybrid": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS, PANEL, RULE = "#1B1814", "#55504A", "#8A8478", "#E7E2D6", "#C9C2B3", "#FBFAF6", "#DCD6C8"


def esc(s) -> str:
    return html.escape(str(s))


def plate(text: str, label: str = "Registered vehicle") -> str:
    """A number-plate block for the claimed vehicle (rendered with st.html)."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700&family=JetBrains+Mono:wght@600&display=swap');
.vc-plate {{ display:flex; align-items:stretch; border:2px solid #1B1814; border-radius:7px; overflow:hidden;
  background:#F2C230; }}
.vc-plate .band {{ background:#1E3A8A; color:#F8F5EC; font:700 11px/1 'Barlow Condensed',sans-serif;
  letter-spacing:.08em; writing-mode:vertical-rl; transform:rotate(180deg); padding:6px 4px; text-align:center; }}
.vc-plate .body {{ padding:6px 14px 6px; min-width:0; }}
.vc-plate .lab {{ font:600 10px/1 'JetBrains Mono',monospace; letter-spacing:.14em; color:#3B3112; text-transform:uppercase; }}
.vc-plate .txt {{ font:700 24px/1.1 'Barlow Condensed',sans-serif; letter-spacing:.04em; color:#111;
  text-transform:uppercase; overflow-wrap:anywhere; }}
</style>
<div class="vc-plate" role="img" aria-label="{esc(label)}: {esc(text)}"><div class="band">IND</div>
<div class="body"><div class="lab">{esc(label)}</div><div class="txt" translate="no">{esc(text)}</div></div></div>"""
