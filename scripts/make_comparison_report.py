"""
make_comparison_report.py — Generate multi-config HTML comparison report.

Reads V1/V2/V3 benchmark JSON files and produces a side-by-side HTML report.

Usage:
    python scripts/make_comparison_report.py
    python scripts/make_comparison_report.py --out reports/comparison_all.html
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

# ── Config definitions ──────────────────────────────────────────────────────

CONFIGS = [
    {
        "label": "V1",
        "name": "Baseline",
        "desc": "Original 63 queries &bull; No unknown filter &bull; Single query per concept",
        "file": "artifacts/benchmark_results.json",
        "color": "#e74c3c",
        "tag_color": "#fadbd8",
    },
    {
        "label": "V2",
        "name": "Better queries + unknown filter",
        "desc": "90 domain-specific queries &bull; Unknown images filtered &bull; Single query per concept",
        "file": "artifacts/benchmark_V2_Better_queries_plus_unknown_filter.json",
        "color": "#f39c12",
        "tag_color": "#fdebd0",
    },
    {
        "label": "V3",
        "name": "Multi-query RRF ensemble + filter",
        "desc": "56 concepts &times; 3 query variants &bull; RRF fusion (k=60) &bull; Unknown images filtered",
        "file": "artifacts/benchmark_V3_Multi-query_RRF_ensemble_plus_filter.json",
        "color": "#27ae60",
        "tag_color": "#d5f5e3",
    },
]

ORDERED_CATS = [
    "Dermatopathology", "Gastrointestinal", "Pulmonary", "Gynecologic",
    "Soft tissue", "Genitourinary", "Hematopathology", "Renal", "Bone",
    "Neuropathology", "Breast", "Endocrine", "Head and Neck", "Cardiac",
    "Hepatopathology", "Ophthalmic", "Cytopathology",
]


def load_results(cfg: dict) -> dict:
    p = Path(cfg["file"])
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def bar(value: float, color: str, width: int = 160) -> str:
    pct = max(0.0, min(1.0, value))
    filled = int(pct * width)
    return (
        f'<div style="display:inline-flex;align-items:center;gap:6px;">'
        f'<div style="width:{width}px;height:14px;background:#eee;border-radius:7px;overflow:hidden;">'
        f'<div style="width:{filled}px;height:100%;background:{color};border-radius:7px;"></div>'
        f'</div>'
        f'<span style="font-size:12px;font-weight:600;color:#333;">{pct:.3f}</span>'
        f'</div>'
    )


def delta_badge(v: float, ref: float) -> str:
    diff = v - ref
    if abs(diff) < 0.0005:
        return '<span style="color:#888;font-size:11px;">—</span>'
    color = "#27ae60" if diff > 0 else "#e74c3c"
    sign = "+" if diff > 0 else ""
    return f'<span style="color:{color};font-size:11px;font-weight:700;">{sign}{diff:.3f}</span>'


def _agg_row(results: list[dict], encoder: str) -> dict:
    return {c["label"]: r[encoder]["aggregate"] for c, r in zip(CONFIGS, results)}


def summary_table(results: list[dict]) -> str:
    rows_html = ""
    for encoder, enc_label, enc_color in [
        ("quilt", "QuiltNet-B-32", "#8e44ad"),
        ("clip",  "CLIP ViT-B/32",  "#2980b9"),
    ]:
        agg = [r[encoder]["aggregate"] for r in results]
        v1_p = agg[0]["precision_at_k"]

        rows_html += f"""
        <tr style="background:#f8f9fa;">
          <td colspan="5" style="padding:8px 12px;font-size:13px;font-weight:700;color:{enc_color};
                                  border-bottom:2px solid #dee2e6;">{enc_label}</td>
        </tr>"""
        for i, (cfg, a) in enumerate(zip(CONFIGS, agg)):
            is_best = (a["precision_at_k"] == max(x["precision_at_k"] for x in agg))
            best_star = " &#9733;" if is_best else ""
            n = a["n_queries"]
            rows_html += f"""
        <tr>
          <td style="padding:10px 12px;">
            <span style="display:inline-block;padding:3px 8px;border-radius:4px;
                         background:{cfg['tag_color']};color:{cfg['color']};
                         font-weight:700;font-size:12px;">{cfg['label']}</span>
            &nbsp; {cfg['name']}{best_star}
            <div style="font-size:11px;color:#888;margin-top:2px;">{n} queries / concepts</div>
          </td>
          <td style="padding:10px 12px;">{bar(a['precision_at_k'], cfg['color'])}</td>
          <td style="padding:10px 12px;font-size:12px;">{a['map_at_k']:.3f}</td>
          <td style="padding:10px 12px;font-size:12px;">{a['ndcg_at_k']:.3f}</td>
          <td style="padding:10px 12px;">{delta_badge(a['precision_at_k'], v1_p)}</td>
        </tr>"""

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <thead>
        <tr style="background:#343a40;color:#fff;">
          <th style="padding:10px 12px;text-align:left;font-weight:600;">Configuration</th>
          <th style="padding:10px 12px;text-align:left;font-weight:600;">Precision@10</th>
          <th style="padding:10px 12px;text-align:left;font-weight:600;">mAP@10</th>
          <th style="padding:10px 12px;text-align:left;font-weight:600;">nDCG@10</th>
          <th style="padding:10px 12px;text-align:left;font-weight:600;">&#916; vs V1</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>"""


def _cell_bg(v: float) -> str:
    """Background color: white (0) → green (1)."""
    r = int(255 - v * 140)
    g = int(255 - v * 30)
    b = int(255 - v * 140)
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f"rgb({r},{g},{b})"


def category_table(results: list[dict]) -> str:
    all_cats: set[str] = set()
    for r in results:
        for enc in ("quilt", "clip"):
            all_cats.update(r[enc]["per_category"].keys())

    cats = [c for c in ORDERED_CATS if c in all_cats]
    cats += sorted(all_cats - set(ORDERED_CATS))

    header_cols = ""
    for cfg in CONFIGS:
        header_cols += (
            f'<th colspan="2" style="padding:8px 10px;text-align:center;'
            f'background:{cfg["color"]};color:#fff;font-weight:600;font-size:12px;">'
            f'{cfg["label"]}: QuiltNet &nbsp;|&nbsp; CLIP</th>'
        )

    rows_html = ""
    for cat in cats:
        cells = ""
        quilt_vals = []
        for r in results:
            q_p = r["quilt"]["per_category"].get(cat, {}).get("precision_at_k", None)
            c_p = r["clip"]["per_category"].get(cat, {}).get("precision_at_k", None)
            if q_p is not None:
                quilt_vals.append(q_p)

            q_str = f'<span style="font-weight:700;">{q_p:.3f}</span>' if q_p is not None else "—"
            c_str = f'{c_p:.3f}' if c_p is not None else "—"
            q_bg = _cell_bg(q_p) if q_p is not None else "#f8f9fa"
            c_bg = _cell_bg(c_p) if c_p is not None else "#f8f9fa"

            cells += (
                f'<td style="padding:7px 10px;text-align:center;background:{q_bg};'
                f'font-size:12px;">{q_str}</td>'
                f'<td style="padding:7px 10px;text-align:center;background:{c_bg};'
                f'font-size:12px;color:#555;">{c_str}</td>'
            )

        # Best-config indicator for QuiltNet
        best_idx = quilt_vals.index(max(quilt_vals)) if quilt_vals else -1
        label_col = (
            f'<td style="padding:7px 12px;font-size:12px;font-weight:600;'
            f'white-space:nowrap;">{cat}</td>'
        )
        rows_html += f"<tr>{label_col}{cells}</tr>"

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
      <thead>
        <tr style="background:#f8f9fa;">
          <th style="padding:8px 12px;text-align:left;font-weight:600;font-size:12px;">Category</th>
          {header_cols}
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>"""


def improvement_cards(results: list[dict]) -> str:
    v1 = results[0]["quilt"]["per_category"]
    v3 = results[2]["quilt"]["per_category"]
    v2 = results[1]["quilt"]["per_category"]

    deltas_v3 = []
    for cat in ORDERED_CATS:
        p1 = v1.get(cat, {}).get("precision_at_k", None)
        p3 = v3.get(cat, {}).get("precision_at_k", None)
        if p1 is not None and p3 is not None:
            deltas_v3.append((cat, p1, p3, p3 - p1))

    deltas_v3.sort(key=lambda x: -x[3])
    top_gains = deltas_v3[:5]
    top_losses = [x for x in deltas_v3 if x[3] < -0.01][:3]

    def mini_card(cat, p1, p3, diff):
        color = "#27ae60" if diff >= 0 else "#e74c3c"
        sign = "+" if diff >= 0 else ""
        return f"""
        <div style="background:#fff;border:1px solid #dee2e6;border-radius:8px;
                    padding:12px 16px;margin-bottom:8px;">
          <div style="font-weight:700;font-size:13px;">{cat}</div>
          <div style="display:flex;align-items:center;gap:12px;margin-top:6px;">
            <span style="font-size:11px;color:#888;">V1: {p1:.3f}</span>
            <span style="font-size:16px;color:#888;">&#8594;</span>
            <span style="font-size:13px;font-weight:700;color:{color};">{p3:.3f}</span>
            <span style="background:{color};color:#fff;padding:2px 7px;border-radius:10px;
                         font-size:11px;font-weight:700;">{sign}{diff:.3f}</span>
          </div>
        </div>"""

    gains_html = "".join(mini_card(*x) for x in top_gains)
    losses_html = "".join(mini_card(*x) for x in top_losses) if top_losses else \
        '<p style="color:#888;font-size:12px;">No significant regressions.</p>'

    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
      <div>
        <h4 style="margin:0 0 12px;color:#27ae60;">Top Gains (V1 &#8594; V3, QuiltNet)</h4>
        {gains_html}
      </div>
      <div>
        <h4 style="margin:0 0 12px;color:#e74c3c;">Regressions (V1 &#8594; V3, QuiltNet)</h4>
        {losses_html}
      </div>
    </div>"""


def write_report(out_path: Path):
    results = [load_results(c) for c in CONFIGS]

    v1_quilt = results[0]["quilt"]["aggregate"]["precision_at_k"]
    v3_quilt = results[2]["quilt"]["aggregate"]["precision_at_k"]
    best_label = max(
        zip(CONFIGS, results),
        key=lambda x: x[1]["quilt"]["aggregate"]["precision_at_k"]
    )[0]["label"]

    sum_table = summary_table(results)
    cat_table  = category_table(results)
    cards      = improvement_cards(results)

    config_badges = ""
    for cfg in CONFIGS:
        config_badges += f"""
        <div style="background:#fff;border:2px solid {cfg['color']};border-radius:10px;
                    padding:16px 20px;flex:1;min-width:200px;">
          <div style="font-size:22px;font-weight:800;color:{cfg['color']};">{cfg['label']}</div>
          <div style="font-size:14px;font-weight:600;margin:4px 0 8px;">{cfg['name']}</div>
          <div style="font-size:11px;color:#666;line-height:1.5;">{cfg['desc']}</div>
          <div style="margin-top:10px;font-size:16px;font-weight:700;color:{cfg['color']};">
            QuiltNet P@10 = {results[CONFIGS.index(cfg)]['quilt']['aggregate']['precision_at_k']:.3f}
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Histopathology Search &mdash; Benchmark Comparison V1/V2/V3</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
             margin:0; padding:0; background:#f5f6fa; color:#2c3e50; }}
    .container {{ max-width:1200px; margin:0 auto; padding:32px 24px; }}
    h1 {{ font-size:26px; font-weight:800; margin:0 0 4px; }}
    h2 {{ font-size:18px; font-weight:700; margin:28px 0 12px; border-left:4px solid #3498db;
           padding-left:12px; }}
    .subtitle {{ color:#666; font-size:14px; margin-bottom:28px; }}
    .card {{ background:#fff; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,.08);
              padding:24px; margin-bottom:24px; }}
    .winner-badge {{ display:inline-block;padding:6px 16px;border-radius:20px;
                     background:#27ae60;color:#fff;font-weight:700;font-size:13px;
                     margin-bottom:16px; }}
    table tr:nth-child(even) {{ background:#fafafa; }}
    table td, table th {{ border-bottom:1px solid #eee; }}
    .note {{ font-size:11px; color:#888; margin-top:8px; }}
    .metric-note {{ background:#eaf4fb; border-left:4px solid #3498db; padding:10px 14px;
                    border-radius:4px; font-size:12px; color:#2c3e50; margin-top:16px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Histopathology Image Search &mdash; Benchmark Comparison</h1>
    <div class="subtitle">QuiltNet-B-32 vs CLIP ViT-B/32 &nbsp;&bull;&nbsp;
      FAISS IndexFlatIP (cosine similarity) &nbsp;&bull;&nbsp;
      Corpus: 64,052 images (Quilt-1M) &nbsp;&bull;&nbsp;
      Generated {date.today().isoformat()}
    </div>

    <!-- Config overview -->
    <div class="card">
      <h2 style="margin-top:0;">Configurations</h2>
      <div style="display:flex;gap:16px;flex-wrap:wrap;">
        {config_badges}
      </div>
      <div class="metric-note">
        <b>Note:</b> V1 uses 63 queries, V2 uses 90 queries, V3 uses 56 concepts (&#215;3 query variants via RRF).
        Precision@10 values are not directly comparable across configs due to different query sets and concept counts &mdash;
        use the per-category breakdown for a fair per-domain view.
        Best overall configuration: <b>{best_label}</b> (QuiltNet Precision@10).
      </div>
    </div>

    <!-- Aggregate metrics -->
    <div class="card">
      <h2 style="margin-top:0;">Aggregate Metrics</h2>
      {sum_table}
      <div class="note">* nDCG IDCG computed assuming &ge;K relevant documents exist per category (valid for large-corpus categories).</div>
    </div>

    <!-- Per-category comparison -->
    <div class="card">
      <h2 style="margin-top:0;">Per-Category Precision@10</h2>
      <p style="font-size:12px;color:#666;margin:0 0 12px;">
        Cell color: white = 0.0, green = 1.0. Bold = QuiltNet, plain = CLIP.
        Category query counts differ across configs (see JSON files).
      </p>
      {cat_table}
    </div>

    <!-- Improvement highlights -->
    <div class="card">
      <h2 style="margin-top:0;">Key Improvements: V1 &rarr; V3 (QuiltNet)</h2>
      {cards}
    </div>

    <!-- Methodology -->
    <div class="card">
      <h2 style="margin-top:0;">Methodology</h2>
      <table style="width:100%;border-collapse:collapse;font-size:12px;">
        <tr style="background:#343a40;color:#fff;">
          <th style="padding:8px 12px;text-align:left;">Aspect</th>
          <th style="padding:8px 12px;text-align:left;">V1 Baseline</th>
          <th style="padding:8px 12px;text-align:left;">V2 Better Queries</th>
          <th style="padding:8px 12px;text-align:left;">V3 RRF Ensemble</th>
        </tr>
        <tr>
          <td style="padding:8px 12px;font-weight:600;">Query set</td>
          <td style="padding:8px 12px;">63 broad queries</td>
          <td style="padding:8px 12px;">90 domain-specific queries</td>
          <td style="padding:8px 12px;">56 concepts &times; 3 variants = 168 FAISS searches</td>
        </tr>
        <tr style="background:#f8f9fa;">
          <td style="padding:8px 12px;font-weight:600;">Unknown filter</td>
          <td style="padding:8px 12px;">No (33% unknowns pollute top-K)</td>
          <td style="padding:8px 12px;">Yes (top-K&times;3, filter, take top-K)</td>
          <td style="padding:8px 12px;">Yes (same approach)</td>
        </tr>
        <tr>
          <td style="padding:8px 12px;font-weight:600;">Fusion</td>
          <td style="padding:8px 12px;">Single query, no fusion</td>
          <td style="padding:8px 12px;">Single query, no fusion</td>
          <td style="padding:8px 12px;">RRF: score = &sum; 1/(60 + rank<sub>i</sub>)</td>
        </tr>
        <tr style="background:#f8f9fa;">
          <td style="padding:8px 12px;font-weight:600;">Encoders</td>
          <td style="padding:8px 12px;" colspan="3">
            QuiltNet-B-32 (wisdomik/QuiltNet-B-32 via open_clip, histopathology fine-tuned) &nbsp;|&nbsp;
            CLIP ViT-B/32 (openai/clip-vit-base-patch32 via HuggingFace)
          </td>
        </tr>
        <tr>
          <td style="padding:8px 12px;font-weight:600;">Ground truth</td>
          <td style="padding:8px 12px;" colspan="3">
            Binary relevance: result is relevant if <code>pathology</code> label &isin; expected category set.
            Precision@K = |relevant &cap; top-K| / K.
          </td>
        </tr>
      </table>
    </div>

  </div>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] Report written to {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/comparison_v1_v2_v3.html")
    args = parser.parse_args()
    write_report(Path(args.out))


if __name__ == "__main__":
    main()
