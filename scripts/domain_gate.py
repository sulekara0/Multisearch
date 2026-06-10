"""
domain_gate.py — Text-to-text histopathology domain gate.

Why this exists
---------------
Calibration (scripts/test_thresholds.py) proved that absolute text->image cosine
does NOT separate in-domain from out-of-domain queries: the vision-language
modality gap compresses every query to ~0.32, and QuiltNet (domain-adapted)
collapses even harder — "yemek tarifi tarihi" scores the same as "lymph node
biopsy". So domain detection cannot live on the retrieval score.

Instead we detect domain in TEXT space, which has no modality gap. We embed the
query with a general-purpose CLIP text tower and compare it (text<->text cosine)
to a curated set of histopathology concept anchors. d_top3 (mean of the 3 closest
anchors) separates in vs out by a usable margin (~0.80 boundary on CLIP).

Encoder independence
--------------------
The gate always uses CLIP's text tower regardless of which encoder performs
retrieval, because QuiltNet's own text tower is also domain-collapsed and cannot
gate. Demo narrative: "domain detection on a general-purpose text encoder,
retrieval on the domain-specialised encoder."
"""

from __future__ import annotations

from typing import List

import numpy as np

# Histopathology concept anchors (text-side definition of the domain).
ANCHORS: List[str] = [
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

DEFAULT_THRESHOLD = 0.80
DEFAULT_TOPN = 3


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class DomainGate:
    """Scores how 'histopathology-like' a text query is, in text space.

    Build once with a CLIP encoder (anchor matrix is precomputed and cached on
    the instance); call .score(query) per query — one extra text encode (~ms).
    """

    def __init__(self, encoder, anchors: List[str] = ANCHORS, topn: int = DEFAULT_TOPN):
        self.encoder = encoder
        self.anchors = anchors
        self.topn = topn
        self._A = _l2_normalize(encoder.encode_texts(anchors))  # (n_anchor, D)

    def score(self, query: str) -> float:
        """Domain score = mean cosine of the top-N closest anchors (d_top3)."""
        q = _l2_normalize(self.encoder.encode_texts([query]))[0]  # (D,)
        sims = self._A @ q                                        # (n_anchor,)
        topn = np.sort(sims)[::-1][: self.topn]
        return float(topn.mean())

    def is_in_domain(self, query: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
        return self.score(query) >= threshold


# ---------------------------------------------------------------------------
# Self-test / validation (re-locks the threshold against the 10 calib queries)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ROOT = Path(__file__).parent.parent
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "scripts"))

    from encoders import get_encoder
    from thresholds import apply_domain_gate

    THRESHOLD = DEFAULT_THRESHOLD
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
        ("medical image",               None),   # borderline -> either is fine
    ]

    gate = DomainGate(get_encoder("clip"))
    dummy = [{"score": 0.32}]  # stand-in for FAISS results (real ones come later)

    print(f"\n{'='*64}\n DOMAIN GATE VALIDATION (CLIP text-anchor, threshold={THRESHOLD})\n{'='*64}")
    print(f"{'Query':<30}| {'d_top3':<7}| {'status':<11}| {'expected':<9}|")
    print(f"{'-'*30}|{'-'*8}|{'-'*12}|{'-'*10}|")
    n_ok = 0
    for query, in_domain in QUERIES:
        score = gate.score(query)
        verdict = apply_domain_gate(dummy, score, THRESHOLD)
        status = verdict["status"]
        if in_domain is None:
            exp, ok = "border", True
        else:
            exp = "in" if in_domain else "out"
            ok = (status == "ok") == in_domain
        n_ok += ok
        mark = "OK" if ok else "XX"
        print(f"{query[:29]:<30}| {score:<7.2f}| {status:<11}| {exp:<9}| {mark}")
    print(f"\n  Matched: {n_ok}/{len(QUERIES)}")
