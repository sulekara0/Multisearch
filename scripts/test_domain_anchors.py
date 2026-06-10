"""
test_domain_anchors.py — Probe a TEXT-to-TEXT domain gate.

Calibration showed absolute text->image cosine does NOT separate in/out-of-domain
queries (QuiltNet collapses everything to ~0.32 due to the modality gap). This
script tests an alternative: embed the query with the TEXT tower and compare it
to a small set of curated histopathology "anchor" phrases (text<->text cosine,
no modality gap). Text<->text similarity is far more discriminative.

For each query we report:
    d_max   — max cosine to any anchor
    d_top3  — mean of the top-3 anchor cosines (more robust)

Usage:
    python scripts/test_domain_anchors.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = ROOT / "scripts"
os.chdir(ROOT)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
from encoders import get_encoder

ENCODERS = ["clip", "quilt"]
ENCODER_LABELS = {"clip": "Generic CLIP", "quilt": "QuiltNet"}

# Histopathology concept anchors (text-side domain definition)
ANCHORS = [
    "histopathology image of tissue",
    "tissue biopsy under the microscope",
    "hematoxylin and eosin stained section",
    "carcinoma cells in tissue",
    "tumor histology",
    "lymph node biopsy",
    "microscopic view of cells",
    "pathology specimen of an organ",
    "cancer diagnosis from a biopsy",
    "glandular tissue architecture",
    "inflammation in a tissue section",
    "malignant neoplasm histology",
]

# (query, in_domain?)
QUERIES = [
    ("lung cancer biopsy",          True),
    ("invasive ductal carcinoma",   True),
    ("lymph node biopsy histology", True),
    ("araba motoru",                False),
    ("yemek tarifi tarihi",         False),
    ("chocolate cake recipe",       False),
    ("random gibberish xkcd",       False),
    ("the quick brown fox",         False),
    ("computer screen",             False),
    ("medical image",               None),   # borderline
]


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def run_for_encoder(enc_name: str):
    bar = "=" * 70
    print(f"\n{bar}\n DOMAIN-ANCHOR PROBE — {ENCODER_LABELS[enc_name]}\n{bar}")

    enc = get_encoder(enc_name)
    A = _norm(enc.encode_texts(ANCHORS))          # (n_anchor, D)

    rows = []
    for query, in_domain in QUERIES:
        q = _norm(enc.encode_texts([query]))[0]    # (D,)
        sims = A @ q                                # (n_anchor,)
        sims_sorted = np.sort(sims)[::-1]
        d_max = float(sims_sorted[0])
        d_top3 = float(sims_sorted[:3].mean())
        tag = {True: "in ", False: "OUT", None: "brd"}[in_domain]
        print(f"  [{tag}] {query:<30} d_max={d_max:.2f}  d_top3={d_top3:.2f}")
        rows.append((query, in_domain, d_max, d_top3))

    # Separation summary
    ins = [r[3] for r in rows if r[1] is True]
    outs = [r[3] for r in rows if r[1] is False]
    print(f"\n  in-domain  d_top3: min={min(ins):.2f}  max={max(ins):.2f}")
    print(f"  out-domain d_top3: min={min(outs):.2f}  max={max(outs):.2f}")
    gap = min(ins) - max(outs)
    midpoint = (min(ins) + max(outs)) / 2
    verdict = f"SEPARABLE — suggest threshold ~{midpoint:.2f}" if gap > 0 else "OVERLAP — no clean split"
    print(f"  gap (min_in - max_out) = {gap:+.2f}  →  {verdict}")
    return rows


if __name__ == "__main__":
    for enc in ENCODERS:
        run_for_encoder(enc)
    print("\n[done] If d_top3 separates in vs out, the text-anchor gate replaces "
          "the broken absolute-cosine domain check.")
