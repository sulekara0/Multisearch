"""
test_thresholds.py — Calibration harness for the two-stage domain/quality
thresholds (scripts/thresholds.py).

Runs a fixed set of in-domain / out-of-domain / borderline queries through both
the Generic CLIP and QuiltNet indexes, prints the full top-10 cosine
distribution per query, and shows the apply_thresholds() status under the
current default thresholds. Ends with a per-encoder summary table comparing the
observed status against the expected one.

Usage:
    python scripts/test_thresholds.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- UTF-8 console (box-drawing + ✓/⚠ on Windows) ---
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = ROOT / "scripts"
ART_ROOT = ROOT / "artifacts"

# CWD = project root so FAISS relpath avoids non-ASCII absolute-path failures.
os.chdir(ROOT)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from encoders import get_encoder
from faiss_search import load_index, load_embeddings, load_captions
from thresholds import apply_thresholds

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DOMAIN_THRESHOLD = 0.18
QUALITY_THRESHOLD = 0.22
TOPK = 10
ENCODERS = ["clip", "quilt"]
ENCODER_LABELS = {"clip": "Generic CLIP (ViT-B/32)", "quilt": "QuiltNet (ViT-B/32)"}

# (query, accepted_statuses, short_expected_label)
QUERIES = [
    ("lung cancer biopsy",          {"ok", "partial"},                "ok"),
    ("invasive ductal carcinoma",   {"ok", "partial"},                "ok"),
    ("lymph node biopsy histology", {"ok", "partial"},                "ok"),
    ("araba motoru",                {"domain_out"},                   "dom_out"),
    ("yemek tarifi tarihi",         {"domain_out"},                   "dom_out"),
    ("chocolate cake recipe",       {"domain_out"},                   "dom_out"),
    ("random gibberish xkcd",       {"domain_out", "low_quality"},    "dom/low"),
    ("the quick brown fox",         {"domain_out"},                   "dom_out"),
    ("computer screen",             {"domain_out"},                   "dom_out"),
    ("medical image",               {"ok", "partial", "low_quality"}, "borderline"),
]


def _find_captions() -> Path:
    for name in ("captions_quilt.jsonl", "captions_quilt_test.jsonl"):
        p = ART_ROOT / name
        if p.exists():
            return p
    return ART_ROOT / "captions_quilt.jsonl"


def _search_scores(query, encoder, index, metric, ids, cap_map):
    """Return the top-K result dicts for a text query (score = cosine)."""
    from faiss_search import search
    return search(query, encoder, index, ids, cap_map, metric=metric, topk=TOPK)


def run_for_encoder(enc_name: str, cap_map):
    bar = "=" * 68
    print(f"\n{bar}\n CALIBRATION TEST — {ENCODER_LABELS[enc_name]}\n{bar}")

    encoder = get_encoder(enc_name)
    index, metric = load_index(str(ART_ROOT / enc_name))

    rows = []  # (query, top1, status, expected_label, ok_flag)
    for i, (query, accepted, expected_label) in enumerate(QUERIES, start=1):
        results = _search_scores(query, encoder, index, metric, IDS, cap_map)
        scores = [r["score"] for r in results]
        top1 = max(scores) if scores else 0.0
        mean = sum(scores) / len(scores) if scores else 0.0
        n_above_q = sum(1 for s in scores if s >= QUALITY_THRESHOLD)

        verdict = apply_thresholds(results, DOMAIN_THRESHOLD, QUALITY_THRESHOLD)
        status = verdict["status"]

        score_str = "[" + ", ".join(f"{s:.2f}" for s in scores) + "]"
        print(f'\nQ{i}: "{query}"')
        print(f"  Top-{TOPK} scores: {score_str}")
        print(f"  Top-1: {top1:.2f}  Top-K mean: {mean:.2f}")
        extra = f"  ({n_above_q}/{len(scores)} above quality)" if status in ("ok", "partial", "low_quality") else ""
        print(f"  Status (d={DOMAIN_THRESHOLD}, q={QUALITY_THRESHOLD}): {status}{extra}")

        ok_flag = status in accepted
        rows.append((query, top1, status, expected_label, ok_flag))

    # --- Summary table ---
    print(f"\n{bar}")
    print(f" SUMMARY — {ENCODER_LABELS[enc_name]} "
          f"(default thresholds: domain={DOMAIN_THRESHOLD}, quality={QUALITY_THRESHOLD})")
    print(bar)
    print(f"{'Query':<32}| {'Top-1':<6}| {'Status':<12}| {'Expected':<11}|")
    print(f"{'-'*32}|{'-'*7}|{'-'*13}|{'-'*12}|")
    for query, top1, status, expected_label, ok_flag in rows:
        mark = "✓" if ok_flag else "⚠"
        print(f"{query[:31]:<32}| {top1:<6.2f}| {status:<12}| {expected_label:<11}| {mark}")

    n_ok = sum(1 for *_ , f in rows if f)
    print(f"\n  Matched expectation: {n_ok}/{len(rows)}")
    return rows


if __name__ == "__main__":
    CAPTIONS_PATH = _find_captions()
    cap_map = load_captions(CAPTIONS_PATH) if CAPTIONS_PATH.exists() else {}

    # ids are shared across encoders only if built from same image set; load per encoder
    # to be safe, but the score-only calibration just needs the index, not captions.
    for enc in ENCODERS:
        _, IDS = load_embeddings(str(ART_ROOT / enc))
        run_for_encoder(enc, cap_map)

    print("\n[done] Calibration complete. Adjust DOMAIN_THRESHOLD / QUALITY_THRESHOLD "
          "above based on the score distributions.")
