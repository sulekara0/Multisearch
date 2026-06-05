"""
compare_models.py — Benchmark CLIP vs QuiltNet on histopathology queries.

Metrics (binary relevance, category-based ground truth):
    Precision@K  — fraction of top-K results whose pathology is in expected
    mAP@K        — mean average precision
    nDCG@K       — normalised discounted cumulative gain (rank-aware)
    Latency      — encode + FAISS search wall time (ms)

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --queries artifacts/eval_queries.json --topk 10
    python scripts/compare_models.py --queries artifacts/eval_queries.json --topk 10 --out reports/comparison_64k.html
"""

from __future__ import annotations

import argparse
import base64
import html as _html
import json
import math
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from faiss_search import load_index, load_embeddings, load_captions, search
from encoders import get_encoder
from reranker import rerank_results
from cross_encoder_reranker import load_cross_encoder, cross_encoder_rerank

_UNKNOWN_LABELS = {"unknown", ""}


def search_filtered(
    query: str,
    encoder,
    index,
    ids: List[str],
    caption_map: Dict[str, dict],
    metric: str,
    topk: int,
) -> List[dict]:
    """Search top-K*3, strip results with unknown/empty pathology, return top-K."""
    raw = search(
        query=query, encoder=encoder, index=index,
        ids=ids, caption_map=caption_map, metric=metric, topk=topk * 3,
    )
    filtered = [r for r in raw if r.get("pathology", "").lower() not in _UNKNOWN_LABELS]
    return filtered[:topk]


def search_rrf(
    query_variants: List[str],
    encoder,
    index,
    ids: List[str],
    caption_map: Dict[str, dict],
    metric: str,
    topk: int,
    rrf_k: int = 60,
    filter_unknown: bool = True,
) -> List[dict]:
    """Reciprocal Rank Fusion over multiple query variants.

    Each variant fetches top-K*3 candidates (with optional unknown filter).
    RRF score: sum(1 / (rrf_k + rank_i)) across all variants.
    Returns top-K results sorted by fused score.
    """
    from collections import defaultdict

    doc_scores: dict = defaultdict(float)
    doc_cache: dict = {}

    fn = search_filtered if filter_unknown else search
    for q_text in query_variants:
        results = fn(
            query=q_text, encoder=encoder, index=index,
            ids=ids, caption_map=caption_map, metric=metric, topk=topk * 3,
        )
        for rank, r in enumerate(results):
            doc_scores[r["id"]] += 1.0 / (rrf_k + rank)
            doc_cache[r["id"]] = r

    sorted_ids = sorted(doc_scores, key=lambda d: -doc_scores[d])
    return [doc_cache[did] for did in sorted_ids[:topk]]


# ---------------------------------------------------------------------------
# Metric functions  (binary relevance)
# ---------------------------------------------------------------------------

def _relevant(result: dict, expected: List[str]) -> bool:
    return bool(expected) and result["pathology"] in expected


def precision_at_k(results: List[dict], expected: List[str], k: int) -> float:
    if not expected:
        return math.nan
    hits = sum(1 for r in results[:k] if _relevant(r, expected))
    return hits / k


def average_precision_at_k(results: List[dict], expected: List[str], k: int) -> float:
    if not expected:
        return math.nan
    n_hits, ap = 0, 0.0
    for i, r in enumerate(results[:k], start=1):
        if _relevant(r, expected):
            n_hits += 1
            ap += n_hits / i
    return (ap / n_hits) if n_hits > 0 else 0.0


def ndcg_at_k(results: List[dict], expected: List[str], k: int) -> float:
    if not expected:
        return math.nan
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i in range(1, min(k, len(results)) + 1)
        if _relevant(results[i - 1], expected)
    )
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, k + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def eval_query(
    query_text: str,
    expected: List[str],
    encoder,
    index,
    ids: List[str],
    caption_map: Dict[str, dict],
    metric: str,
    k: int,
    filter_unknown: bool = False,
    query_variants: Optional[List[str]] = None,
    rerank: bool = False,
    rerank_alpha: float = 0.6,
    rerank_beta: float = 0.4,
    rerank_gamma: float = 0.0,
    rerank_pool_size: int = 50,
    cross_encoder: bool = False,
    ce_model=None,
    ce_tokenizer=None,
    ce_model_name: str = "biomedclip",
    ce_device: str = "cpu",
    ce_batch_size: int = 32,
    cross_encoder_pool: int = 100,
) -> dict:
    t0 = time.perf_counter()
    if cross_encoder:
        fetch_k = cross_encoder_pool
    elif rerank:
        fetch_k = rerank_pool_size
    else:
        fetch_k = k
    if query_variants:
        results = search_rrf(
            query_variants=query_variants,
            encoder=encoder, index=index,
            ids=ids, caption_map=caption_map, metric=metric, topk=fetch_k,
            filter_unknown=filter_unknown,
        )
    else:
        fn = search_filtered if filter_unknown else search
        results = fn(
            query=query_text, encoder=encoder, index=index,
            ids=ids, caption_map=caption_map, metric=metric, topk=fetch_k,
        )
    if rerank and results:
        intermediate_k = rerank_pool_size if cross_encoder else k
        results = rerank_results(
            query=query_text,
            candidates=results,
            expected_categories=None,
            alpha=rerank_alpha,
            beta=rerank_beta,
            gamma=rerank_gamma,
            topk=intermediate_k,
        )
    if cross_encoder and results:
        results = cross_encoder_rerank(
            query_text, results, ce_model, ce_tokenizer,
            model_name=ce_model_name,
            device=ce_device,
            batch_size=ce_batch_size,
            topk=k,
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "precision_at_k": precision_at_k(results, expected, k),
        "map_at_k":       average_precision_at_k(results, expected, k),
        "ndcg_at_k":      ndcg_at_k(results, expected, k),
        "latency_ms":     latency_ms,
        "results":        results,
    }


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def _print_rerank_debug(
    query_text: str,
    query_id: str,
    faiss_results: List[dict],
    reranked_results: List[dict],
    k: int,
) -> None:
    """Print compact rank-change table for one query."""
    faiss_ids = [r["id"] for r in faiss_results[:k]]
    rerank_ids = [r["id"] for r in reranked_results[:k]]
    id_to_faiss_rank = {rid: i for i, rid in enumerate(faiss_ids)}

    print(f"\n  [RERANK DEBUG] {query_id}: {query_text!r}")
    print(f"  {'Rerank #':<10} {'FAISS #':<10} {'D':<5} {'Pathology':<26} {'Rerank score'}")
    print("  " + "-" * 72)
    for rr_rank, r in enumerate(reranked_results[:k], start=1):
        faiss_rank = id_to_faiss_rank.get(r["id"])
        if faiss_rank is None:
            delta_str = "  NEW"
        else:
            delta = faiss_rank - (rr_rank - 1)  # positive = moved up
            delta_str = f"  +{delta}" if delta > 0 else (f"  -{-delta}" if delta < 0 else "  ==")
        path = (r.get("pathology") or "")[:25]
        score = r.get("ce_score", r.get("rerank_score", r.get("score", 0)))
        faiss_r_str = f"#{faiss_rank + 1}" if faiss_rank is not None else "  —"
        print(f"  #{rr_rank:<9} {faiss_r_str:<10} {delta_str:<5} {path:<26} {score:.4f}")


def run_benchmark(
    queries: List[dict],
    encoder,
    encoder_name: str,
    index,
    ids: List[str],
    caption_map: Dict[str, dict],
    metric: str,
    k: int,
    filter_unknown: bool = False,
    ensemble: bool = False,
    rerank: bool = False,
    rerank_alpha: float = 0.6,
    rerank_beta: float = 0.4,
    rerank_gamma: float = 0.0,
    rerank_pool_size: int = 50,
    rerank_debug_n: int = 0,
    cross_encoder: bool = False,
    ce_model=None,
    ce_tokenizer=None,
    ce_model_name: str = "biomedclip",
    ce_device: str = "cpu",
    ce_batch_size: int = 32,
    cross_encoder_pool: int = 100,
) -> List[dict]:
    rows = []
    n = len(queries)
    for qi, q in enumerate(queries):
        # Auto-detect ensemble format (has "queries" list) vs single-query format
        variants = q.get("queries") if ensemble else None
        display_query = variants[0] if variants else q.get("query", "")

        _ce_kwargs = dict(
            cross_encoder=cross_encoder, ce_model=ce_model,
            ce_tokenizer=ce_tokenizer, ce_model_name=ce_model_name,
            ce_device=ce_device, ce_batch_size=ce_batch_size,
            cross_encoder_pool=cross_encoder_pool,
        )
        # --- Optional debug: compare FAISS order vs re-ranked for first N queries ---
        if (rerank or cross_encoder) and rerank_debug_n > 0 and qi < rerank_debug_n:
            m_base = eval_query(
                query_text=display_query, expected=q["expected"],
                encoder=encoder, index=index, ids=ids,
                caption_map=caption_map, metric=metric, k=k,
                filter_unknown=filter_unknown, query_variants=variants,
                rerank=False,
            )
            m = eval_query(
                query_text=display_query, expected=q["expected"],
                encoder=encoder, index=index, ids=ids,
                caption_map=caption_map, metric=metric, k=k,
                filter_unknown=filter_unknown, query_variants=variants,
                rerank=rerank,
                rerank_alpha=rerank_alpha, rerank_beta=rerank_beta,
                rerank_gamma=rerank_gamma, rerank_pool_size=rerank_pool_size,
                **_ce_kwargs,
            )
            _print_rerank_debug(
                display_query, q["id"],
                m_base["results"], m["results"], k,
            )
        else:
            m = eval_query(
                query_text=display_query, expected=q["expected"],
                encoder=encoder, index=index, ids=ids,
                caption_map=caption_map, metric=metric, k=k,
                filter_unknown=filter_unknown, query_variants=variants,
                rerank=rerank,
                rerank_alpha=rerank_alpha, rerank_beta=rerank_beta,
                rerank_gamma=rerank_gamma, rerank_pool_size=rerank_pool_size,
                **_ce_kwargs,
            )

        rows.append({
            "id": q["id"], "query": display_query, "expected": q["expected"],
            "encoder": encoder_name,
            **{key: m[key] for key in ("precision_at_k", "map_at_k", "ndcg_at_k", "latency_ms")},
            "top_results": m["results"],
        })
        bar_done = int((qi + 1) / n * 30)
        bar = "#" * bar_done + "-" * (30 - bar_done)
        print(f"\r  [{bar}] {qi+1:3d}/{n}  {q['id']}  lat={m['latency_ms']:.1f}ms   ",
              end="", flush=True)
    print()
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _nanmean(values: List[float]) -> float:
    v = [x for x in values if not math.isnan(x)]
    return float(np.mean(v)) if v else math.nan


def aggregate_rows(rows: List[dict]) -> dict:
    return {
        "precision_at_k": _nanmean([r["precision_at_k"] for r in rows]),
        "map_at_k":       _nanmean([r["map_at_k"]       for r in rows]),
        "ndcg_at_k":      _nanmean([r["ndcg_at_k"]      for r in rows]),
        "latency_ms":     _nanmean([r["latency_ms"]      for r in rows]),
        "n_queries":      sum(1 for r in rows if r["expected"]),
    }


def aggregate_by_category(rows: List[dict]) -> Dict[str, dict]:
    cats: Dict[str, List[dict]] = {}
    for r in rows:
        for cat in r["expected"]:
            cats.setdefault(cat, []).append(r)
    return {cat: aggregate_rows(cat_rows) for cat, cat_rows in cats.items()}


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_summary(
    clip_agg: dict, quilt_agg: dict,
    clip_cats: Dict[str, dict], quilt_cats: Dict[str, dict],
    k: int,
) -> None:
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  BENCHMARK SUMMARY  (K={k}, labelled queries only)")
    print(sep)
    print(f"  {'Metric':<22}  {'CLIP':>8}  {'QuiltNet':>8}  {'D':>8}  {'D%':>8}")
    print("  " + "-" * 66)

    for label, key, higher_better in [
        ("Precision@K",  "precision_at_k", True),
        ("mAP@K",        "map_at_k",       True),
        ("nDCG@K",       "ndcg_at_k",      True),
        ("Latency (ms)", "latency_ms",     False),
    ]:
        c, q = clip_agg[key], quilt_agg[key]
        delta = q - c if not (math.isnan(q) or math.isnan(c)) else math.nan
        if math.isnan(delta):
            d_str, pct_str = "  n/a", "  n/a"
        else:
            sign = "+" if delta > 0 else ""
            d_str = f"{sign}{delta:.3f}"
            pct = delta / c * 100 if c != 0 else 0.0
            pct_str = f"{sign}{pct:.1f}%"
        print(f"  {label:<22}  {c:>8.3f}  {q:>8.3f}  {d_str:>8}  {pct_str:>8}")

    print(f"\n{sep}")
    print(f"  PER-CATEGORY BREAKDOWN  (Precision@{k})")
    print(sep)
    print(f"  {'Category':<28}  {'CLIP':>6}  {'QuiltNet':>8}  {'D':>7}")
    print("  " + "-" * 54)

    all_cats = sorted(
        set(list(clip_cats.keys()) + list(quilt_cats.keys())),
        key=lambda c: -(quilt_cats.get(c, {}).get("precision_at_k", 0) or 0),
    )
    for cat in all_cats:
        c_v = clip_cats.get(cat, {}).get("precision_at_k", math.nan)
        q_v = quilt_cats.get(cat, {}).get("precision_at_k", math.nan)
        delta = q_v - c_v if not (math.isnan(q_v) or math.isnan(c_v)) else math.nan
        if math.isnan(delta):
            d_str = "  n/a"
        else:
            sign = "+" if delta > 0 else ""
            d_str = f"{sign}{delta:.3f}"
        print(f"  {cat:<28}  {c_v:>6.3f}  {q_v:>8.3f}  {d_str:>7}")
    print(sep)


# ---------------------------------------------------------------------------
# Progression table
# ---------------------------------------------------------------------------

# Tracks all runs in-memory across calls within a session.
_PROGRESSION: List[dict] = []

_V1_LABEL = "V1: Baseline (original queries)"


def _print_progression(
    current_quilt_p: float,
    current_clip_p: float,
    config_name: Optional[str],
    baseline_path: Path,
) -> None:
    """Print cumulative progression table vs V1 baseline."""
    # Seed from saved V1 JSON if progression is empty
    if not _PROGRESSION and baseline_path.exists():
        try:
            saved = json.loads(baseline_path.read_text(encoding="utf-8"))
            v1_p = saved["quilt"]["aggregate"]["precision_at_k"]
            v1_c = saved["clip"]["aggregate"]["precision_at_k"]
            _PROGRESSION.append({"config": _V1_LABEL, "quilt": v1_p, "clip": v1_c})
        except Exception:
            pass

    label = config_name or f"Run {len(_PROGRESSION) + 1}"
    _PROGRESSION.append({"config": label, "quilt": current_quilt_p, "clip": current_clip_p})

    v1_quilt = _PROGRESSION[0]["quilt"] if _PROGRESSION else current_quilt_p
    v1_clip  = _PROGRESSION[0]["clip"]  if _PROGRESSION else current_clip_p

    sep = "-" * 72
    print(f"\n{sep}")
    print("  PROGRESSION TABLE  (QuiltNet Precision@K)")
    print(sep)
    print(f"  {'Configuration':<42}  {'QuiltNet':>8}  {'CLIP':>6}  {'D from V1':>10}")
    print("  " + "-" * 68)
    for row in _PROGRESSION:
        qp = row["quilt"]
        cp = row["clip"]
        delta = qp - v1_quilt
        sign = "+" if delta >= 0 else ""
        d_str = f"{sign}{delta:.3f}" if row["config"] != _V1_LABEL else "  baseline"
        print(f"  {row['config']:<42}  {qp:>8.3f}  {cp:>6.3f}  {d_str:>10}")
    print(sep)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f1f5f9; color: #1e293b; line-height: 1.5;
}
.container { max-width: 1400px; margin: 0 auto; padding: 32px 24px; }

/* Header */
.page-header { margin-bottom: 36px; }
.page-header h1 { font-size: 1.7rem; font-weight: 700; color: #0f172a; }
.page-header .subtitle { color: #64748b; font-size: 0.9rem; margin-top: 4px; }

/* Metric cards */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }
.metric-card {
  background: #fff; border-radius: 12px; padding: 20px 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,.07), 0 1px 2px rgba(0,0,0,.05);
}
.metric-label {
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: .08em;
  color: #94a3b8; font-weight: 600; margin-bottom: 10px;
}
.metric-row { display: flex; align-items: baseline; gap: 10px; }
.val-clip  { font-size: 1.6rem; font-weight: 800; color: #3b82f6; }
.val-quilt { font-size: 1.6rem; font-weight: 800; color: #16a34a; }
.vs { color: #cbd5e1; font-size: 0.85rem; }
.metric-delta { margin-top: 6px; font-size: 0.82rem; color: #64748b; }
.delta-pos { color: #16a34a; font-weight: 600; }
.delta-neg { color: #ef4444; font-weight: 600; }
.delta-neu { color: #64748b; }

/* Legend */
.legend { display: flex; gap: 24px; margin-bottom: 20px; font-size: 0.82rem; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; vertical-align: middle; }

/* Sections */
.section {
  background: #fff; border-radius: 12px; padding: 24px 28px;
  box-shadow: 0 1px 3px rgba(0,0,0,.07); margin-bottom: 28px;
}
.section h2 { font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 20px; }

/* Category table */
.cat-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
.cat-table th {
  text-align: left; padding: 8px 12px; color: #64748b;
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em;
  border-bottom: 2px solid #e2e8f0; font-weight: 600;
}
.cat-table td { padding: 10px 12px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
.cat-table tr:last-child td { border-bottom: none; }
.cat-table tr:hover td { background: #fafafa; }
.bar-cell { display: flex; align-items: center; gap: 8px; }
.bar-track { width: 160px; height: 8px; background: #f1f5f9; border-radius: 4px; position: relative; flex-shrink: 0; }
.bar-fill  { height: 8px; border-radius: 4px; position: absolute; top: 0; left: 0; }
.bar-clip  { background: #93c5fd; }
.bar-quilt { background: #86efac; }
.num-clip  { color: #3b82f6; font-weight: 700; min-width: 40px; }
.num-quilt { color: #16a34a; font-weight: 700; min-width: 40px; }
.num-delta { font-weight: 600; min-width: 55px; }

/* Sample query blocks */
.query-block {
  border: 1px solid #e2e8f0; border-radius: 10px;
  margin-bottom: 20px; overflow: hidden;
}
.query-head {
  background: #f8fafc; padding: 14px 20px;
  border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; gap: 12px;
}
.query-id { font-size: 0.72rem; color: #94a3b8; font-weight: 600; }
.query-text { font-weight: 600; font-size: 1rem; color: #0f172a; }
.query-cats { display: flex; gap: 6px; flex-wrap: wrap; }
.cat-badge {
  font-size: 0.68rem; padding: 2px 8px; border-radius: 20px;
  background: #f0fdf4; color: #15803d; font-weight: 600; border: 1px solid #bbf7d0;
}
.query-metrics {
  font-size: 0.75rem; color: #64748b; margin-left: auto; white-space: nowrap;
}

.result-grid { display: grid; grid-template-columns: 1fr 1fr; }
.result-col { padding: 16px 20px; }
.result-col:first-child { border-right: 1px solid #e2e8f0; }
.col-title {
  font-size: 0.78rem; font-weight: 700; padding: 3px 10px;
  border-radius: 20px; display: inline-block; margin-bottom: 12px;
}
.clip-title  { background: #dbeafe; color: #1d4ed8; }
.quilt-title { background: #dcfce7; color: #166534; }

.cards-row { display: flex; gap: 8px; flex-wrap: wrap; }
.img-card {
  width: 130px; border-radius: 8px; overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,.1); background: #fff;
  transition: transform .15s;
}
.img-card:hover { transform: translateY(-2px); }
.img-card.hit { outline: 2.5px solid #16a34a; }
.img-card img { width: 130px; height: 130px; object-fit: cover; display: block; }
.no-img {
  width: 130px; height: 130px; background: #f8fafc;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.62rem; color: #94a3b8; text-align: center; padding: 8px;
}
.card-info { padding: 6px 8px; }
.card-rank  { font-size: 0.65rem; color: #94a3b8; }
.card-score { font-size: 0.65rem; color: #64748b; margin-left: 4px; }
.card-path  {
  font-size: 0.65rem; margin-top: 3px; padding: 1px 5px; border-radius: 3px;
  background: #f0fdf4; color: #15803d;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 116px;
}
.card-path.miss { background: #fef9f0; color: #92400e; }

/* Notes */
.notes { font-size: 0.875rem; color: #475569; line-height: 1.7; }
.notes li { margin-left: 20px; margin-bottom: 4px; }

/* Footer */
footer { text-align: center; color: #94a3b8; font-size: 0.78rem; margin-top: 40px; padding-bottom: 32px; }
"""


def _find_image(img_id: str, img_dir: Path) -> Optional[Path]:
    direct = img_dir / img_id
    if direct.exists():
        return direct
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        cand = img_dir / (Path(img_id).stem + ext)
        if cand.exists():
            return cand
    return None


def _to_b64(img_path: Path, size: int = 130) -> Optional[str]:
    try:
        from PIL import Image as PILImage
        img = PILImage.open(img_path).convert("RGB")
        img.thumbnail((size, size), PILImage.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _img_card_html(rank: int, result: dict, img_dir: Path, expected: List[str]) -> str:
    is_hit = _relevant(result, expected)
    hit_cls = " hit" if is_hit else ""
    img_path = _find_image(result["id"], img_dir)
    b64 = _to_b64(img_path) if img_path else None

    img_html = (
        f'<img src="data:image/jpeg;base64,{b64}" alt="rank {rank}">'
        if b64 else '<div class="no-img">Image not found</div>'
    )
    path = _html.escape(result.get("pathology", "") or "")
    path_cls = "card-path" if is_hit else "card-path miss"
    path_html = f'<div class="{path_cls}">{path}</div>' if path else ""

    return f"""<div class="img-card{hit_cls}">
      {img_html}
      <div class="card-info">
        <span class="card-rank">#{rank}</span>
        <span class="card-score">{result["score"]:.3f}</span>
        {path_html}
      </div>
    </div>"""


def _query_block_html(clip_row: dict, quilt_row: dict, k: int, img_dir: Path) -> str:
    q_text   = _html.escape(clip_row["query"])
    expected = clip_row["expected"]
    cat_badges = "".join(f'<span class="cat-badge">{_html.escape(c)}</span>' for c in expected)
    if not cat_badges:
        cat_badges = '<span class="cat-badge" style="background:#f8fafc;color:#94a3b8;border-color:#e2e8f0">general</span>'

    cp = clip_row["precision_at_k"]
    qp = quilt_row["precision_at_k"]
    delta = qp - cp if not (math.isnan(qp) or math.isnan(cp)) else math.nan
    sign = "+" if delta >= 0 else ""
    d_str = f"{sign}{delta:.2f}" if not math.isnan(delta) else "n/a"
    metrics_html = f'P@{k}: <b style="color:#3b82f6">{cp:.2f}</b> vs <b style="color:#16a34a">{qp:.2f}</b> ({d_str})'

    show = min(k, 5)
    clip_cards  = "".join(_img_card_html(i+1, r, img_dir, expected) for i, r in enumerate(clip_row["top_results"][:show]))
    quilt_cards = "".join(_img_card_html(i+1, r, img_dir, expected) for i, r in enumerate(quilt_row["top_results"][:show]))

    return f"""<div class="query-block">
  <div class="query-head">
    <span class="query-id">{clip_row["id"]}</span>
    <span class="query-text">{q_text}</span>
    <div class="query-cats">{cat_badges}</div>
    <div class="query-metrics">{metrics_html}</div>
  </div>
  <div class="result-grid">
    <div class="result-col">
      <div class="col-title clip-title">Generic CLIP</div>
      <div class="cards-row">{clip_cards}</div>
    </div>
    <div class="result-col">
      <div class="col-title quilt-title">QuiltNet</div>
      <div class="cards-row">{quilt_cards}</div>
    </div>
  </div>
</div>"""


def _cat_table_html(clip_cats: Dict[str, dict], quilt_cats: Dict[str, dict]) -> str:
    all_cats = sorted(
        set(list(clip_cats.keys()) + list(quilt_cats.keys())),
        key=lambda c: -(quilt_cats.get(c, {}).get("precision_at_k", 0) or 0),
    )
    rows_html = ""
    for cat in all_cats:
        cv = clip_cats.get(cat, {}).get("precision_at_k", 0.0) or 0.0
        qv = quilt_cats.get(cat, {}).get("precision_at_k", 0.0) or 0.0
        delta = qv - cv
        sign = "+" if delta >= 0 else ""
        dc = "delta-pos" if delta > 0 else ("delta-neg" if delta < 0 else "delta-neu")

        cb_w = int(cv * 160)
        qb_w = int(qv * 160)

        rows_html += f"""<tr>
          <td>{_html.escape(cat)}</td>
          <td>
            <div class="bar-cell">
              <div class="bar-track"><div class="bar-fill bar-clip" style="width:{cb_w}px"></div></div>
              <span class="num-clip">{cv:.3f}</span>
            </div>
          </td>
          <td>
            <div class="bar-cell">
              <div class="bar-track"><div class="bar-fill bar-quilt" style="width:{qb_w}px"></div></div>
              <span class="num-quilt">{qv:.3f}</span>
            </div>
          </td>
          <td class="num-delta {dc}">{sign}{delta:.3f}</td>
        </tr>"""

    return f"""<table class="cat-table">
  <thead>
    <tr>
      <th>Category</th>
      <th><span style="color:#3b82f6">CLIP</span> Precision@K</th>
      <th><span style="color:#16a34a">QuiltNet</span> Precision@K</th>
      <th>Delta</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>"""


def _metric_card_html(label: str, clip_v: float, quilt_v: float,
                      fmt: str = ".3f", higher_better: bool = True) -> str:
    if math.isnan(clip_v) or math.isnan(quilt_v):
        return f'<div class="metric-card"><div class="metric-label">{label}</div><div>n/a</div></div>'
    delta = quilt_v - clip_v
    pct   = delta / clip_v * 100 if clip_v != 0 else 0.0
    sign  = "+" if delta >= 0 else ""
    improvement = (delta > 0) == higher_better
    dc = "delta-pos" if improvement else ("delta-neg" if delta != 0 else "delta-neu")
    return f"""<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-row">
    <span class="val-clip">{clip_v:{fmt}}</span>
    <span class="vs">vs</span>
    <span class="val-quilt">{quilt_v:{fmt}}</span>
  </div>
  <div class="metric-delta <{dc}">{sign}{delta:{fmt}} &nbsp;<span class="{dc}">({sign}{pct:.1f}%)</span></div>
</div>"""


def write_html(
    clip_rows: List[dict],
    quilt_rows: List[dict],
    clip_agg: dict,
    quilt_agg: dict,
    clip_cats: Dict[str, dict],
    quilt_cats: Dict[str, dict],
    k: int,
    out_path: Path,
    image_dir: Path,
    rerank_tag: str = "",
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_labelled = clip_agg["n_queries"]

    # --- Metric cards ---
    cards_html = (
        _metric_card_html("Precision@K",    clip_agg["precision_at_k"], quilt_agg["precision_at_k"])
        + _metric_card_html("mAP@K",        clip_agg["map_at_k"],       quilt_agg["map_at_k"])
        + _metric_card_html("nDCG@K",       clip_agg["ndcg_at_k"],      quilt_agg["ndcg_at_k"])
        + _metric_card_html("Latency (ms)", clip_agg["latency_ms"],     quilt_agg["latency_ms"],
                            fmt=".1f", higher_better=False)
    )

    # --- Category table ---
    cat_table = _cat_table_html(clip_cats, quilt_cats)

    # --- Select sample queries ---
    clip_by_id  = {r["id"]: r for r in clip_rows}
    quilt_by_id = {r["id"]: r for r in quilt_rows}

    labelled = [r for r in quilt_rows if r["expected"] and r["id"] in clip_by_id]
    labelled.sort(
        key=lambda r: r["precision_at_k"] - clip_by_id[r["id"]]["precision_at_k"],
        reverse=True,
    )

    selected_ids: set = set()
    samples: List[tuple] = []

    # Top 5 QuiltNet improvement
    for qr in labelled:
        if len(samples) >= 5:
            break
        samples.append((clip_by_id[qr["id"]], qr))
        selected_ids.add(qr["id"])

    # 1 where CLIP wins
    for qr in reversed(labelled):
        if qr["id"] not in selected_ids:
            cr = clip_by_id[qr["id"]]
            if cr["precision_at_k"] > qr["precision_at_k"]:
                samples.append((cr, qr))
                selected_ids.add(qr["id"])
                break

    # 1 where both zero or low
    for qr in reversed(labelled):
        if qr["id"] not in selected_ids and qr["precision_at_k"] == 0:
            samples.append((clip_by_id[qr["id"]], qr))
            break

    print(f"  [HTML] Rendering {len(samples)} sample queries (loading images from {image_dir})...")
    query_blocks = "".join(_query_block_html(cr, qr, k, image_dir) for cr, qr in samples)

    # --- Notes ---
    p_delta  = quilt_agg["precision_at_k"] - clip_agg["precision_at_k"]
    pct_p    = p_delta / clip_agg["precision_at_k"] * 100 if clip_agg["precision_at_k"] != 0 else 0
    lat_clip = clip_agg["latency_ms"]
    lat_quilt= quilt_agg["latency_ms"]

    best_cat  = max(quilt_cats, key=lambda c: quilt_cats[c]["precision_at_k"])
    worst_cat = min(quilt_cats, key=lambda c: quilt_cats[c]["precision_at_k"])
    clip_wins = [c for c in clip_cats if clip_cats[c].get("precision_at_k", 0) > quilt_cats.get(c, {}).get("precision_at_k", 0)]

    notes_items = [
        f"QuiltNet outperforms CLIP on Precision@{k} by <b>{p_delta:.3f}</b> ({pct_p:+.1f}%) averaged over {n_labelled} labelled queries.",
        f"Best QuiltNet category: <b>{_html.escape(best_cat)}</b> ({quilt_cats[best_cat]['precision_at_k']:.3f}). "
        f"Lowest: <b>{_html.escape(worst_cat)}</b> ({quilt_cats[worst_cat]['precision_at_k']:.3f}).",
        f"CLIP shows higher Precision@{k} in: <b>{', '.join(_html.escape(c) for c in clip_wins) if clip_wins else 'none'}</b>.",
        f"Latency: CLIP {lat_clip:.1f} ms vs QuiltNet {lat_quilt:.1f} ms per query (CPU, encode + FAISS). "
        f"QuiltNet uses open_clip tokenizer, adding ~{lat_quilt - lat_clip:.1f} ms overhead.",
        f"Evaluation is category-level (pathology field from Quilt-1M metadata). A result is <em>relevant</em> "
        f"if its pathology tag matches the query&apos;s expected category list. The 3 general/staining queries "
        f"(H&amp;E, IHC) are excluded from metric aggregation.",
        "Green outline on image cards = relevant (pathology matches expected category). "
        "Orange badge = irrelevant category.",
    ]
    notes_html = "<ul class='notes'>" + "".join(f"<li>{n}</li>" for n in notes_items) + "</ul>"

    # --- Assemble ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark: CLIP vs QuiltNet — Histopathology</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="container">

  <div class="page-header">
    <h1>Histopathology Multimodal Search Benchmark</h1>
    <div class="subtitle">
      CLIP (ViT-B/32) vs QuiltNet-B-32 &nbsp;·&nbsp;
      {n_labelled} labelled queries &nbsp;·&nbsp;
      114K image corpus &nbsp;·&nbsp;
      K = {k} &nbsp;·&nbsp;
      {ts}{(" &nbsp;·&nbsp; " + rerank_tag) if rerank_tag else ""}
    </div>
  </div>

  <div class="legend">
    <span><span class="legend-dot" style="background:#3b82f6"></span>Generic CLIP (openai/clip-vit-base-patch32)</span>
    <span><span class="legend-dot" style="background:#16a34a"></span>QuiltNet-B-32 (wisdomik/QuiltNet-B-32)</span>
  </div>

  <div class="metric-grid">{cards_html}</div>

  <div class="section">
    <h2>Per-Category Precision@{k}</h2>
    {cat_table}
  </div>

  <div class="section">
    <h2>Sample Query Comparisons (top-5 results per encoder)</h2>
    <p style="font-size:0.82rem;color:#64748b;margin-bottom:18px">
      <span style="display:inline-block;width:12px;height:12px;border-radius:2px;
            outline:2.5px solid #16a34a;margin-right:5px;vertical-align:middle"></span>
      Green outline = relevant (pathology matches expected category)
    </p>
    {query_blocks}
  </div>

  <div class="section">
    <h2>Discussion</h2>
    {notes_html}
  </div>

  <footer>
    Generated by compare_models.py &nbsp;·&nbsp; {ts} &nbsp;·&nbsp;
    Quilt-1M dataset &nbsp;·&nbsp; Bitirme Projesi
  </footer>

</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"[HTML] Report saved: {out_path}  ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CLIP vs QuiltNet")
    parser.add_argument("--queries",   default="artifacts/eval_queries.json")
    parser.add_argument("--topk",      type=int, default=10)
    parser.add_argument("--captions",  default="artifacts/captions_quilt.jsonl")
    parser.add_argument("--clip_dir",  default="artifacts/clip")
    parser.add_argument("--quilt_dir", default="artifacts/quilt")
    parser.add_argument("--image_dir",      default="data/images")
    parser.add_argument("--out",            default=None,
                        help="HTML output path (e.g. reports/comparison_64k.html)")
    parser.add_argument("--filter_unknown", action="store_true",
                        help="Filter out results with unknown/empty pathology (top-K*3 then slice)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use multi-query RRF ensemble (query file must have 'queries' list per entry)")
    parser.add_argument("--config_name",    default=None,
                        help="Label for this run in the progression table (e.g. 'V2: Better queries + filter')")
    parser.add_argument("--baseline_json",  default="artifacts/benchmark_results.json",
                        help="Path to V1 baseline results JSON for progression table")
    # Re-ranking (Jaccard)
    parser.add_argument("--rerank",           action="store_true",
                        help="Enable caption-based re-ranking after FAISS retrieval")
    parser.add_argument("--rerank_alpha",     type=float, default=0.6,
                        help="Re-rank weight for normalised cosine score (default 0.6)")
    parser.add_argument("--rerank_beta",      type=float, default=0.4,
                        help="Re-rank weight for caption Jaccard overlap (default 0.4)")
    parser.add_argument("--rerank_gamma",     type=float, default=0.0,
                        help="Re-rank weight for pathology bonus — keep 0.0 for fair eval (default 0.0)")
    parser.add_argument("--rerank_pool_size", type=int,   default=50,
                        help="Candidate pool size fetched from FAISS before re-ranking (default 50)")
    parser.add_argument("--rerank_debug",     type=int,   default=0, metavar="N",
                        help="Print FAISS-vs-reranked rank comparison for first N queries (default 0=off)")
    # Cross-encoder re-ranking
    parser.add_argument("--cross_encoder",       action="store_true",
                        help="Enable cross-encoder re-ranking (BiomedCLIP or ms-marco)")
    parser.add_argument("--cross_encoder_pool",  type=int, default=100,
                        help="FAISS candidate pool size for cross-encoder (default 100)")
    parser.add_argument("--cross_encoder_model", default="biomedclip",
                        choices=["biomedclip", "ms-marco"],
                        help="Cross-encoder model: biomedclip (default) or ms-marco fallback")
    parser.add_argument("--cross_encoder_batch", type=int, default=32,
                        help="Batch size for cross-encoder encoding (default 32)")
    args = parser.parse_args()

    K = args.topk

    queries: List[dict] = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    print(f"[INFO] Loaded {len(queries)} queries from {args.queries}")

    caption_map = load_captions(Path(args.captions))
    print(f"[INFO] Caption map: {len(caption_map)} entries")

    print("\n[CLIP] Loading index + encoder...")
    clip_index, clip_metric = load_index(args.clip_dir)
    _, clip_ids = load_embeddings(args.clip_dir)
    clip_enc = get_encoder("clip")

    print("\n[QUILT] Loading index + encoder...")
    quilt_index, quilt_metric = load_index(args.quilt_dir)
    _, quilt_ids = load_embeddings(args.quilt_dir)
    quilt_enc = get_encoder("quilt")

    filter_flag        = args.filter_unknown
    ensemble_flag      = args.ensemble
    rerank_flag        = args.rerank
    cross_encoder_flag = args.cross_encoder

    # Load cross-encoder once if requested
    ce_model = ce_tokenizer = ce_model_name_loaded = None
    ce_device = "cpu"
    if cross_encoder_flag:
        ce_device = "cuda" if torch.cuda.is_available() else "cpu"
        ce_model, ce_tokenizer, ce_model_name_loaded = load_cross_encoder(
            device=ce_device, model_name=args.cross_encoder_model
        )
        print(f"[INFO] Cross-encoder: {ce_model_name_loaded} on {ce_device}")

    mode_tag = ""
    if ensemble_flag:      mode_tag += " [RRF ensemble]"
    if filter_flag:        mode_tag += " [unknown filtered]"
    if rerank_flag:        mode_tag += f" [reranked a={args.rerank_alpha} b={args.rerank_beta} pool={args.rerank_pool_size}]"
    if cross_encoder_flag: mode_tag += f" [CE: {ce_model_name_loaded} pool={args.cross_encoder_pool}]"

    rerank_tag_parts = []
    if rerank_flag:
        rerank_tag_parts.append(
            f"caption re-rank a={args.rerank_alpha} b={args.rerank_beta} g={args.rerank_gamma} "
            f"pool={args.rerank_pool_size}"
        )
    if cross_encoder_flag:
        rerank_tag_parts.append(f"CE: {ce_model_name_loaded} pool={args.cross_encoder_pool}")
    rerank_tag = " → ".join(rerank_tag_parts)

    _rerank_kwargs = dict(
        rerank=rerank_flag,
        rerank_alpha=args.rerank_alpha,
        rerank_beta=args.rerank_beta,
        rerank_gamma=args.rerank_gamma,
        rerank_pool_size=args.rerank_pool_size,
        rerank_debug_n=args.rerank_debug,
        cross_encoder=cross_encoder_flag,
        ce_model=ce_model,
        ce_tokenizer=ce_tokenizer,
        ce_model_name=ce_model_name_loaded or args.cross_encoder_model,
        ce_device=ce_device,
        ce_batch_size=args.cross_encoder_batch,
        cross_encoder_pool=args.cross_encoder_pool,
    )

    print(f"\n[CLIP]  Evaluating {len(queries)} queries (K={K}){mode_tag}...")
    clip_rows = run_benchmark(
        queries, clip_enc, "clip",
        clip_index, clip_ids, caption_map, clip_metric, K,
        filter_unknown=filter_flag, ensemble=ensemble_flag,
        **_rerank_kwargs,
    )

    print(f"\n[QUILT] Evaluating {len(queries)} queries (K={K}){mode_tag}...")
    quilt_rows = run_benchmark(
        queries, quilt_enc, "quilt",
        quilt_index, quilt_ids, caption_map, quilt_metric, K,
        filter_unknown=filter_flag, ensemble=ensemble_flag,
        **_rerank_kwargs,
    )

    clip_labelled  = [r for r in clip_rows  if r["expected"]]
    quilt_labelled = [r for r in quilt_rows if r["expected"]]

    clip_agg   = aggregate_rows(clip_labelled)
    quilt_agg  = aggregate_rows(quilt_labelled)
    clip_cats  = aggregate_by_category(clip_labelled)
    quilt_cats = aggregate_by_category(quilt_labelled)

    print_summary(clip_agg, quilt_agg, clip_cats, quilt_cats, K)

    # Load baseline BEFORE saving so the file is not overwritten first
    baseline_path = Path(args.baseline_json)
    _print_progression(
        current_quilt_p=quilt_agg["precision_at_k"],
        current_clip_p=clip_agg["precision_at_k"],
        config_name=args.config_name,
        baseline_path=baseline_path,
    )

    # Save: V1 always goes to benchmark_results.json; subsequent runs get their own file
    is_v1 = not args.config_name or "V1" in (args.config_name or "")
    if is_v1:
        results_path = Path("artifacts/benchmark_results.json")
    else:
        safe_name = (args.config_name or "run").replace(" ", "_").replace(":", "").replace("+", "plus")
        results_path = Path(f"artifacts/benchmark_{safe_name}.json")

    results_path.write_text(
        json.dumps({
            "k": K, "config": args.config_name,
            "filter_unknown": filter_flag,
            "queries_file": args.queries,
            "clip":  {"aggregate": clip_agg,  "per_category": clip_cats},
            "quilt": {"aggregate": quilt_agg, "per_category": quilt_cats},
        }, indent=2, ensure_ascii=False, default=lambda x: None if math.isnan(x) else x),
        encoding="utf-8",
    )
    print(f"\n[INFO] Results saved: {results_path}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_html(
            clip_rows, quilt_rows,
            clip_agg, quilt_agg,
            clip_cats, quilt_cats,
            K, out_path, Path(args.image_dir),
            rerank_tag=rerank_tag,
        )


if __name__ == "__main__":
    main()
