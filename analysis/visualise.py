"""
Stage 2: Visualise analysis results as an HTML report.
Reads a JSON file from the "JSON files" folder and opens a report in the browser.

Usage: python visualise.py
"""

import json
import sys
import webbrowser
from pathlib import Path

ROOT       = Path(__file__).parent.parent          # Serendipity/
JSON_DIR   = ROOT / "data" / "reports"
REPORTS_DIR = ROOT / "data" / "reports"

AUDIO_EMOTION_COLOURS = {
    "happy":   "#f59e0b",
    "neutral": "#6b7280",
    "angry":   "#ef4444",
    "sad":     "#3b82f6",
}


# ─── File picker ──────────────────────────────────────────────────────────────

def pick_json_file() -> Path:
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in: {JSON_DIR}")
        print("Run analyse.py first to generate a JSON file.")
        sys.exit(1)

    print("JSON files available:\n")
    for i, f in enumerate(json_files, 1):
        print(f"  [{i}] {f.name}")

    print()
    while True:
        choice = input(f"Select a file [1-{len(json_files)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(json_files):
            return json_files[int(choice) - 1]
        print("  Invalid choice, try again.")


# ─── HTML helpers ─────────────────────────────────────────────────────────────

def bar(pct: float, colour: str, bold: bool = False) -> str:
    """A labelled horizontal bar."""
    weight = "600" if bold else "400"
    return f"""
    <div class="bar-track">
      <div class="bar-fill" style="width:{pct*100:.1f}%;background:{colour};"></div>
    </div>
    <span class="bar-pct" style="font-weight:{weight};">{pct*100:.1f}%</span>"""


def section(title: str, content: str) -> str:
    return f"""
  <div class="card">
    <h2>{title}</h2>
    {content}
  </div>"""


# ─── Section builders ─────────────────────────────────────────────────────────

def build_header(d: dict) -> str:
    t = d["transcript"]
    dur = int(t["total_duration_sec"])
    mins, secs = divmod(dur, 60)
    dur_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    date_str = d["analysed_at"].replace("T", " ")
    return f"""
  <div class="card header-card">
    <div class="header-icon">🎵</div>
    <div>
      <h1>{d['file']}</h1>
      <p class="meta">
        Analysed: {date_str}
        &nbsp;•&nbsp; {dur_str}
        &nbsp;•&nbsp; {t['segment_count']} segments
        &nbsp;•&nbsp; {t['language'].upper()} ({t['language_confidence']*100:.0f}% confidence)
      </p>
    </div>
  </div>"""


def build_audio_emotion(d: dict) -> str:
    ae = d["audio_emotion"]
    dominant = ae["dominant"]
    rows = ""
    for label, score in sorted(ae["scores"].items(), key=lambda x: x[1], reverse=True):
        colour = AUDIO_EMOTION_COLOURS.get(label, "#6b7280")
        is_dominant = label == dominant
        badge = " <span class='badge'>dominant</span>" if is_dominant else ""
        rows += f"""
    <div class="bar-row">
      <span class="bar-label" style="{'font-weight:600;color:#111;' if is_dominant else ''}">{label}{badge}</span>
      {bar(score, colour, bold=is_dominant)}
    </div>"""
    return section("🔊 Audio Emotion <span class='subtitle'>tone of voice</span>", rows)


def build_text_emotion(d: dict) -> str:
    te = d["text_emotion"]
    dominant = te["dominant"]
    scores = te["scores"]  # already sorted desc by analyse.py
    top10 = list(scores.items())[:10]
    rows = ""
    for i, (label, score) in enumerate(top10):
        colour = "#6d28d9" if i < 3 else "#8b5cf6"
        is_dominant = label == dominant
        badge = " <span class='badge'>dominant</span>" if is_dominant else ""
        rows += f"""
    <div class="bar-row">
      <span class="bar-label" style="{'font-weight:600;color:#111;' if is_dominant else ''}">{label}{badge}</span>
      {bar(score, colour, bold=is_dominant)}
    </div>"""
    remaining = len(scores) - 10
    if remaining > 0:
        rows += f"<p class='note'>+ {remaining} more emotions with lower scores</p>"
    return section(f"💬 Text Emotion <span class='subtitle'>top 10 of {len(scores)}</span>", rows)


def build_likes_dislikes(d: dict) -> str:
    likes    = d.get("likes", [])
    dislikes = d.get("dislikes", [])

    def items(lst: list, colour: str, icon: str) -> str:
        if not lst:
            return f"<p class='empty'>None detected</p>"
        return "".join(f"<li style='color:{colour};'><span style='color:{colour};'>{icon}</span> {item}</li>" for item in lst)

    content = f"""
    <div class="two-col">
      <div>
        <h3 style="color:#16a34a;">✓ Likes</h3>
        <ul class="opinion-list">{items(likes, '#16a34a', '●')}</ul>
      </div>
      <div>
        <h3 style="color:#dc2626;">✗ Dislikes</h3>
        <ul class="opinion-list">{items(dislikes, '#dc2626', '●')}</ul>
      </div>
    </div>"""
    return section("👍 Likes &amp; Dislikes", content)


def build_person_type(d: dict) -> str:
    pt = d["person_type"]
    top_traits = set(pt["top_traits"])
    scores = pt["scores"]
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rows = ""
    for label, score in sorted_scores:
        is_top = label in top_traits
        colour = "#10b981" if is_top else "#94a3b8"
        star = "★ " if is_top else "  "
        rows += f"""
    <div class="bar-row">
      <span class="bar-label" style="{'font-weight:600;color:#111;' if is_top else 'color:#6b7280;'}">{star}{label}</span>
      {bar(score, colour, bold=is_top)}
    </div>"""
    return section("🧠 Person Type", rows)


# ─── Full HTML ─────────────────────────────────────────────────────────────────

def build_html(data: dict) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{data['file']}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f1f5f9;
      color: #334155;
      padding: 2rem 1rem;
    }}
    .container {{ max-width: 780px; margin: 0 auto; display: flex; flex-direction: column; gap: 1.25rem; }}
    .card {{
      background: #fff;
      border-radius: 12px;
      padding: 1.5rem 1.75rem;
      box-shadow: 0 1px 3px rgba(0,0,0,.08), 0 4px 12px rgba(0,0,0,.04);
    }}
    .header-card {{ display: flex; align-items: center; gap: 1.25rem; }}
    .header-icon {{ font-size: 2.5rem; line-height: 1; }}
    h1 {{ font-size: 1.1rem; font-weight: 700; color: #0f172a; word-break: break-word; }}
    .meta {{ font-size: 0.8rem; color: #94a3b8; margin-top: 0.3rem; }}
    h2 {{
      font-size: 0.75rem; font-weight: 700; letter-spacing: .08em;
      text-transform: uppercase; color: #64748b;
      margin-bottom: 1rem;
    }}
    h2 .subtitle {{ font-weight: 400; text-transform: none; letter-spacing: 0; color: #94a3b8; font-size: 0.75rem; }}
    h3 {{ font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; }}
    .bar-row {{
      display: grid;
      grid-template-columns: 140px 1fr 52px;
      align-items: center;
      gap: 0.6rem;
      margin-bottom: 0.55rem;
    }}
    .bar-label {{ font-size: 0.85rem; color: #475569; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .bar-track {{ background: #e2e8f0; border-radius: 99px; height: 10px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 99px; transition: width 0.3s; }}
    .bar-pct {{ font-size: 0.78rem; color: #64748b; text-align: right; }}
    .badge {{
      display: inline-block; font-size: 0.6rem; font-weight: 600;
      background: #eff6ff; color: #3b82f6; border-radius: 4px;
      padding: 1px 5px; vertical-align: middle; margin-left: 4px;
      text-transform: uppercase; letter-spacing: .05em;
    }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
    .opinion-list {{ list-style: none; margin-top: 0.25rem; }}
    .opinion-list li {{ font-size: 0.85rem; padding: 0.3rem 0; line-height: 1.4; }}
    .empty {{ font-size: 0.82rem; color: #94a3b8; font-style: italic; }}
    .note {{ font-size: 0.75rem; color: #94a3b8; margin-top: 0.5rem; }}
    @media (max-width: 500px) {{
      .two-col {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 110px 1fr 48px; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    {build_header(data)}
    {build_audio_emotion(data)}
    {build_text_emotion(data)}
    {build_likes_dislikes(data)}
    {build_person_type(data)}
  </div>
</body>
</html>"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    json_file = pick_json_file()

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / Path(data["file"]).with_suffix(".html").name
    out_path.write_text(build_html(data), encoding="utf-8")

    print(f"\nReport saved to: {out_path}")
    print("Opening in browser...")
    webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
