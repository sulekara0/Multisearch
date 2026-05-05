"""
faiss_search.py — FAISS index build / search / benchmark

Subcommands
-----------
build   Build FAISS index from embeddings.npy in --art_dir
search  Encode a text query and return top-K results
bench   Latency benchmark using random query vectors from embeddings.npy

Usage examples
--------------
python scripts/faiss_search.py build --art_dir artifacts/clip
python scripts/faiss_search.py build --art_dir artifacts/quilt

python scripts/faiss_search.py search --art_dir artifacts/clip  --encoder clip  --query "lung cancer"
python scripts/faiss_search.py search --art_dir artifacts/quilt --encoder quilt --query "lung cancer"
python scripts/faiss_search.py search --art_dir artifacts/clip  --encoder clip  --query "invasive ductal carcinoma" --json

python scripts/faiss_search.py bench --art_dir artifacts/clip
python scripts/faiss_search.py bench --art_dir artifacts/quilt --n_queries 200 --topk 10
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
import tracemalloc
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

np.set_printoptions(suppress=True, precision=4)

try:
    import faiss
except ImportError:
    raise SystemExit("[ERROR] FAISS not installed. Run: pip install faiss-cpu")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _paths(art_dir: str):
    d = Path(art_dir)
    return {
        "emb":   d / "embeddings.npy",
        "ids":   d / "ids.json",
        "index": d / "faiss_index.bin",
        "meta":  d / "faiss_meta.json",
    }


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------

def load_embeddings(art_dir: str) -> Tuple[np.ndarray, List[str]]:
    p = _paths(art_dir)
    if not p["emb"].exists():
        raise FileNotFoundError(f"Embeddings not found: {p['emb']}")
    if not p["ids"].exists():
        raise FileNotFoundError(f"IDs not found: {p['ids']}")

    embs = np.load(p["emb"]).astype(np.float32)
    ids: List[str] = json.loads(p["ids"].read_text(encoding="utf-8"))

    if len(ids) != embs.shape[0]:
        raise ValueError(
            f"ids length ({len(ids)}) != embeddings rows ({embs.shape[0]})"
        )
    return embs, ids


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def _build_index(embs: np.ndarray, metric: str = "ip") -> faiss.Index:
    d = embs.shape[1]
    if metric == "l2":
        idx = faiss.IndexFlatL2(d)
        idx.add(embs)
    elif metric == "ip":
        idx = faiss.IndexFlatIP(d)
        idx.add(_l2_normalize(embs))
    else:
        raise ValueError("metric must be 'l2' or 'ip'")
    return idx


def _save_index(idx: faiss.Index, art_dir: str, metric: str, n: int, d: int):
    p = _paths(art_dir)
    faiss.write_index(idx, str(p["index"]))
    p["meta"].write_text(
        json.dumps({"metric": metric, "n": n, "d": d}, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] Index saved: {p['index']}  meta: {p['meta']}")


def load_index(art_dir: str) -> Tuple[faiss.Index, str]:
    p = _paths(art_dir)
    if not p["index"].exists():
        raise FileNotFoundError(
            f"Index not found: {p['index']} — run 'build' first."
        )
    # Use relpath to avoid FAISS C-library failures on non-ASCII absolute paths (Windows).
    try:
        index_path = os.path.relpath(str(p["index"]))
    except ValueError:
        index_path = str(p["index"])
    idx = faiss.read_index(index_path)
    meta = json.loads(p["meta"].read_text(encoding="utf-8"))
    return idx, meta["metric"]


# ---------------------------------------------------------------------------
# Caption map
# ---------------------------------------------------------------------------

def load_captions(jsonl_path: Path) -> Dict[str, dict]:
    """Load captions_quilt.jsonl → {filename: {caption, pathology}}."""
    cap_map: Dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
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
    return cap_map


def _resolve_captions(art_dir: str, captions_arg: str | None) -> Dict[str, dict]:
    if captions_arg:
        cap_path = Path(captions_arg)
    else:
        cap_path = Path(art_dir).parent / "captions_quilt.jsonl"

    if cap_path.exists():
        cap_map = load_captions(cap_path)
        print(f"[INFO] Captions loaded: {len(cap_map)} entries from {cap_path}")
        return cap_map

    # Silently skip — captions are optional
    return {}


# ---------------------------------------------------------------------------
# Core search function (programmatic API)
# ---------------------------------------------------------------------------

def search(
    query: str,
    encoder,
    index: faiss.Index,
    ids: List[str],
    caption_map: Dict[str, dict],
    metric: str = "ip",
    topk: int = 10,
) -> List[dict]:
    """
    Encode query text and search the FAISS index.

    Returns
    -------
    list of dicts: [{"id", "score", "caption", "pathology"}, ...]
    Importable by compare_models.py: from faiss_search import search
    """
    q = encoder.encode_texts([query])          # (1, D) float32, L2-normalized
    if metric == "ip":
        q = _l2_normalize(q)
        scores, I = index.search(q, topk)
        raw_scores = scores[0].tolist()
    else:
        dists, I = index.search(q, topk)
        raw_scores = (-dists[0]).tolist()       # negate for display

    results = []
    for rank_i, idx in enumerate(I[0].tolist()):
        img_id = ids[idx]
        cap_rec = caption_map.get(img_id, {})
        results.append({
            "id":        img_id,
            "score":     float(raw_scores[rank_i]),
            "caption":   cap_rec.get("caption", ""),
            "pathology": cap_rec.get("pathology", ""),
        })
    return results


# ---------------------------------------------------------------------------
# CLI output formatter
# ---------------------------------------------------------------------------

def format_results_for_cli(results: List[dict]) -> str:
    lines = []
    for rank, r in enumerate(results, start=1):
        score_str = f"{r['score']:.4f}"
        line = f"#{rank:02d}  score={score_str}  {r['id']}"
        if r["pathology"]:
            line += f"\n      pathology: {r['pathology']}"
        if r["caption"]:
            preview = r["caption"][:120].replace("\n", " ")
            line += f"\n      caption:   {preview}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_build(args):
    embs, ids = load_embeddings(args.art_dir)
    print(f"[INFO] Embeddings: {embs.shape}  ids: {len(ids)}")
    idx = _build_index(embs, metric=args.metric)
    _save_index(idx, args.art_dir, metric=args.metric,
                n=embs.shape[0], d=embs.shape[1])


def cmd_search(args):
    sys.path.insert(0, str(Path(__file__).parent))
    from encoders import get_encoder  # noqa: PLC0415

    index, metric = load_index(args.art_dir)
    _, ids = load_embeddings(args.art_dir)

    encoder = get_encoder(args.encoder)
    caption_map = _resolve_captions(args.art_dir, args.captions)

    results = search(
        query=args.query,
        encoder=encoder,
        index=index,
        ids=ids,
        caption_map=caption_map,
        metric=metric,
        topk=args.topk,
    )

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(format_results_for_cli(results))


def cmd_bench(args):
    # --- Load index and embeddings ---
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    index, metric = load_index(args.art_dir)
    embs, _ = load_embeddings(args.art_dir)

    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    ram_mb = sum(s.size for s in snap_after.compare_to(snap_before, "lineno")) / 1024 / 1024

    # --- Build query vectors: random sample from stored embeddings ---
    n_q = min(args.n_queries, len(embs))
    rng = np.random.default_rng(42)
    query_vecs = embs[rng.choice(len(embs), size=n_q, replace=False)]  # already normalized

    # --- Measure latency ---
    latencies = []
    for qv in query_vecs:
        qv = qv[np.newaxis, :].astype(np.float32)
        t0 = time.perf_counter()
        index.search(qv, args.topk)
        latencies.append(time.perf_counter() - t0)

    lat = np.array(latencies) * 1000  # ms
    index_size_mb = _paths(args.art_dir)["index"].stat().st_size / 1024 / 1024

    report = {
        "art_dir":         args.art_dir,
        "n_vectors":       int(embs.shape[0]),
        "dim":             int(embs.shape[1]),
        "n_queries":       n_q,
        "topk":            args.topk,
        "latency_mean_ms": round(float(lat.mean()), 3),
        "latency_p50_ms":  round(float(np.percentile(lat, 50)), 3),
        "latency_p95_ms":  round(float(np.percentile(lat, 95)), 3),
        "latency_p99_ms":  round(float(np.percentile(lat, 99)), 3),
        "index_size_mb":   round(index_size_mb, 3),
        "ram_delta_mb":    round(ram_mb, 3),
    }

    report_path = Path(args.art_dir) / "bench_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n[BENCH RESULTS]")
    for k, v in report.items():
        print(f"  {k:<22} {v}")
    print(f"\n[INFO] Report saved: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FAISS text-to-image search (encoder-agnostic)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- build ---
    p_build = sub.add_parser("build", help="Build FAISS index from embeddings.npy")
    p_build.add_argument("--art_dir", default="artifacts",
                         help="Artifacts directory (contains embeddings.npy)")
    p_build.add_argument("--metric", choices=["l2", "ip"], default="ip")
    p_build.set_defaults(func=cmd_build)

    # --- search ---
    p_search = sub.add_parser("search", help="Encode text query and retrieve images")
    p_search.add_argument("--art_dir", default="artifacts")
    p_search.add_argument("--encoder",
                          default=os.environ.get("ENCODER", "clip"),
                          help="Encoder name: clip | quilt | ... (or set ENCODER env var)")
    p_search.add_argument("--query", required=True, help="Text query")
    p_search.add_argument("--topk", type=int, default=10)
    p_search.add_argument("--captions", default=None,
                          help="Path to captions JSONL (default: auto-detect)")
    p_search.add_argument("--json", action="store_true",
                          help="Print results as JSON instead of human-readable text")
    p_search.set_defaults(func=cmd_search)

    # --- bench ---
    p_bench = sub.add_parser("bench", help="Latency benchmark")
    p_bench.add_argument("--art_dir", default="artifacts")
    p_bench.add_argument("--n_queries", type=int, default=100,
                         help="Number of random query vectors to benchmark")
    p_bench.add_argument("--topk", type=int, default=10)
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
