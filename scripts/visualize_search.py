"""
visualize_search.py — Side-by-side CLIP vs QuiltNet search comparison HTML report

Usage
-----
python scripts/visualize_search.py --query "lung tissue with cancer cells"
python scripts/visualize_search.py --query "lymph node" --topk 5
python scripts/visualize_search.py --query "invasive ductal carcinoma" --topk 10 --captions artifacts/captions_quilt_test.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Dict

from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def _find_image(img_id: str, img_dir: Path) -> Path | None:
    direct = img_dir / img_id
    if direct.exists():
        return direct
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        stem = Path(img_id).stem
        candidate = img_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def _to_base64(img_path: Path, size: int = 200) -> str:
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_captions(art_dir: str, captions_arg: str | None) -> Dict[str, dict]:
    if captions_arg:
        cap_path = Path(captions_arg)
    else:
        cap_path = Path(art_dir).parent / "captions_quilt.jsonl"

    if not cap_path.exists():
        return {}

    cap_map: Dict[str, dict] = {}
    with open(cap_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fname = Path(rec.get("path", "")).name
            if fname:
                cap_map[fname] = {
                    "caption":   rec.get("caption", ""),
                    "pathology": rec.get("pathology", ""),
                }
    print(f"[INFO] Captions loaded: {len(cap_map)} entries from {cap_path}")
    return cap_map


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #f5f5f5; color: #222; padding: 24px; }
h1 { font-size: 1.3rem; margin-bottom: 4px; }
.meta { color: #666; font-size: 0.85rem; margin-bottom: 24px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.col-header { font-size: 1rem; font-weight: 600; margin-bottom: 12px;
              padding: 8px 12px; border-radius: 6px; }
.clip-header  { background: #dbeafe; color: #1e40af; }
.quilt-header { background: #dcfce7; color: #166534; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; }
.card { background: #fff; border-radius: 8px; overflow: hidden;
        width: 200px; box-shadow: 0 1px 4px rgba(0,0,0,.12); }
.card img { width: 200px; height: 200px; object-fit: cover; display: block; }
.card.overlap { outline: 3px solid #16a34a; }
.card-body { padding: 8px; }
.rank { font-size: 0.7rem; color: #888; }
.score { font-weight: 700; font-size: 0.9rem; }
.pathology { display: inline-block; margin-top: 4px; padding: 2px 6px;
             border-radius: 4px; background: #f0f0f0;
             font-size: 0.7rem; color: #444; }
.caption { margin-top: 6px; font-size: 0.72rem; color: #555;
           line-height: 1.4; max-height: 4.2em; overflow: hidden; }
.no-img { width: 200px; height: 200px; background: #eee;
          display: flex; align-items: center; justify-content: center;
          font-size: 0.75rem; color: #999; }
"""


def _card_html(rank: int, result: dict, b64: str | None, is_overlap: bool) -> str:
    overlap_class = " overlap" if is_overlap else ""
    score_str = f"{result['score']:.4f}"
    pathology = result.get("pathology") or ""
    caption   = (result.get("caption") or "")[:150]
    fname     = result["id"]

    if b64:
        img_tag = f'<img src="data:image/jpeg;base64,{b64}" alt="{fname}">'
    else:
        img_tag = f'<div class="no-img">Image not found<br>{fname[:20]}</div>'

    path_badge = f'<span class="pathology">{pathology}</span>' if pathology else ""
    cap_block  = f'<div class="caption">{caption}</div>' if caption else ""

    return f"""
    <div class="card{overlap_class}">
      {img_tag}
      <div class="card-body">
        <span class="rank">#{rank:02d}</span>
        <span class="score"> {score_str}</span>
        {path_badge}
        {cap_block}
      </div>
    </div>"""


def _col_html(results: List[dict], img_dir: Path,
              other_ids: set, label: str, header_class: str) -> str:
    cards = ""
    for rank, r in enumerate(results, start=1):
        img_path = _find_image(r["id"], img_dir)
        b64 = _to_base64(img_path) if img_path else None
        overlap = r["id"] in other_ids
        cards += _card_html(rank, r, b64, overlap)

    return f"""
    <div>
      <div class="col-header {header_class}">{label}</div>
      <div class="cards">{cards}</div>
    </div>"""


def build_html(query: str, clip_results: List[dict], quilt_results: List[dict],
               img_dir: Path) -> str:
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    clip_ids  = {r["id"] for r in clip_results}
    quilt_ids = {r["id"] for r in quilt_results}
    overlap_count = len(clip_ids & quilt_ids)

    clip_col  = _col_html(clip_results,  img_dir, quilt_ids, "Generic CLIP",  "clip-header")
    quilt_col = _col_html(quilt_results, img_dir, clip_ids,  "QuiltNet",       "quilt-header")

    overlap_note = (
        f"&nbsp;·&nbsp;<span style='color:#16a34a'>&#9632; {overlap_count} shared result(s)</span>"
        if overlap_count else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Search: {query}</title>
<style>{_CSS}</style>
</head>
<body>
  <h1>Query: &ldquo;{query}&rdquo;</h1>
  <p class="meta">{ts}{overlap_note}</p>
  <div class="grid">
    {clip_col}
    {quilt_col}
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side CLIP vs QuiltNet search visualization"
    )
    parser.add_argument("--query",      required=True, help="Text query")
    parser.add_argument("--topk",       type=int, default=5)
    parser.add_argument("--img_dir",    default="data/images_test",
                        help="Directory containing images")
    parser.add_argument("--clip_dir",   default="artifacts/clip")
    parser.add_argument("--quilt_dir",  default="artifacts/quilt")
    parser.add_argument("--captions",   default=None,
                        help="Path to captions JSONL (default: auto-detect)")
    parser.add_argument("--out_dir",    default="reports")
    args = parser.parse_args()

    # Lazy imports — keep --help fast
    sys.path.insert(0, str(Path(__file__).parent))
    from encoders import get_encoder          # noqa: PLC0415
    from faiss_search import (                # noqa: PLC0415
        _load_index, _load_embeddings, search,
    )

    img_dir  = Path(args.img_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load both indexes ---
    clip_index,  clip_metric  = _load_index(args.clip_dir)
    quilt_index, quilt_metric = _load_index(args.quilt_dir)

    _, clip_ids  = _load_embeddings(args.clip_dir)
    _, quilt_ids = _load_embeddings(args.quilt_dir)

    # --- Caption maps ---
    clip_cap_map  = _load_captions(args.clip_dir,  args.captions)
    quilt_cap_map = _load_captions(args.quilt_dir, args.captions)

    # --- Encoders ---
    print("[INFO] Loading encoders...")
    clip_encoder  = get_encoder("clip")
    quilt_encoder = get_encoder("quilt")

    # --- Search ---
    print(f"[INFO] Searching: '{args.query}'  topk={args.topk}")
    clip_results  = search(args.query, clip_encoder,  clip_index,
                           clip_ids,  clip_cap_map,  clip_metric,  args.topk)
    quilt_results = search(args.query, quilt_encoder, quilt_index,
                           quilt_ids, quilt_cap_map, quilt_metric, args.topk)

    # --- Build HTML ---
    html = build_html(args.query, clip_results, quilt_results, img_dir)

    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"search_{_slug(args.query)}_{ts_str}.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"[OK] Report saved: {out_path}")


if __name__ == "__main__":
    main()
