"""
reranker.py — Caption-based re-ranking for histopathology image search.

Re-ranking score (three components):
    final = alpha * cosine + beta * weighted_caption_overlap + gamma * pathology_bonus

Components:
    cosine                  — Raw FAISS cosine similarity (already in [0,1] for
                              L2-normalised vectors; no min-max normalisation needed)
    weighted_caption_overlap— Weighted Jaccard: high-signal tokens (anatomy, disease
                              names) weight 1.0x; low-signal tokens (cells, year, old)
                              weight 0.3x; hard stopwords removed entirely
    pathology_bonus         — +1.0 if candidate pathology in expected_categories, else 0.0

Default weights (alpha=0.75, beta=0.25, gamma=0.0):
    gamma=0.0 keeps evaluation honest — no ground-truth leakage.
    gamma>0 is valid only in production when categories come from user intent,
    not from eval ground-truth labels.

Usage (standalone test):
    python scripts/reranker.py
"""

from __future__ import annotations

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Hard stopwords — removed entirely from tokenisation
# ---------------------------------------------------------------------------

_EN_STOPWORDS = {
    "a", "an", "the", "of", "in", "with", "and", "or", "is", "are",
    "was", "were", "to", "from", "at", "by", "for", "on", "this", "that",
    "these", "those", "it", "its", "be", "has", "have", "had", "as", "not",
    "but", "so", "if", "up", "do", "no", "we", "he", "she", "they", "our",
    "also", "which", "within", "between", "into", "than", "more", "such",
    "can", "may", "show", "showing", "shows", "seen", "note", "present",
    "consistent", "well", "large", "small", "high", "low", "number",
    "area", "areas", "region", "regions",
}

_MEDICAL_STOPWORDS = {
    "image", "images", "histology", "histological", "histopathology",
    "histopathological", "stain", "staining", "stained", "he", "hande",
    "section", "sections", "tissue", "tissues", "specimen", "specimens",
    "slide", "slides", "microscopy", "microscopic", "photomicrograph",
    "pathology", "pathological", "biopsy", "biopsies",
    "magnification", "objective", "field", "view",
}

_STOPWORDS = _EN_STOPWORDS | _MEDICAL_STOPWORDS

# ---------------------------------------------------------------------------
# Soft low-signal tokens — kept in tokenisation but weighted 0.3x in Jaccard
# Tokens that survive hard filtering but are too common to be discriminative
# ---------------------------------------------------------------------------

_LOW_SIGNAL_TOKENS = {
    "cells", "cell", "year", "years", "old", "male", "female",
    "case", "cases", "patient", "patients",
    "normal", "abnormal", "chronic", "acute",
    "type", "grade", "stage", "level",
    "right", "left", "bilateral", "adjacent", "surrounding",
    "one", "two", "three", "multiple", "several", "many", "few",
}

_LOW_SIGNAL_WEIGHT = 0.3


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alpha chars, remove hard stopwords, min length 2.

    Returns a list (not set) to preserve duplicates for weighted counting.
    """
    tokens = re.split(r"[^a-zA-Z]+", text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Weighted overlap metric
# ---------------------------------------------------------------------------

def _weighted_caption_overlap(query_tokens: list[str], caption_tokens: list[str]) -> float:
    """Weighted Jaccard: low-signal tokens count 0.3x, others count 1.0x.

    Uses set intersection/union on unique tokens, then applies per-token
    weights to the numerator and denominator.
    """
    q_set = set(query_tokens)
    c_set = set(caption_tokens)

    union = q_set | c_set
    if not union:
        return 0.0

    intersection = q_set & c_set

    def _w(token: str) -> float:
        return _LOW_SIGNAL_WEIGHT if token in _LOW_SIGNAL_TOKENS else 1.0

    weighted_intersection = sum(_w(t) for t in intersection)
    weighted_union        = sum(_w(t) for t in union)

    return weighted_intersection / weighted_union if weighted_union > 0 else 0.0


# ---------------------------------------------------------------------------
# Core re-ranking function
# ---------------------------------------------------------------------------

def rerank_results(
    query: str,
    candidates: List[dict],
    expected_categories: Optional[List[str]] = None,
    alpha: float = 0.75,
    beta: float = 0.25,
    gamma: float = 0.0,
    topk: int = 10,
) -> List[dict]:
    """Re-rank candidate list and return top-K.

    Parameters
    ----------
    query : str
        Original text query.
    candidates : list[dict]
        Each dict must have keys: "id", "score", "caption", "pathology".
        Typically the top-K*pool_size results from FAISS search.
    expected_categories : list[str] | None
        Known-relevant pathology labels. Pass None (default) during evaluation
        to avoid ground-truth leakage. Pass user-specified category list in
        production (e.g. from a UI filter or query-classification module).
    alpha : float
        Weight for raw cosine similarity (FAISS score, no normalisation).
    beta : float
        Weight for weighted caption overlap (weighted Jaccard).
    gamma : float
        Weight for pathology bonus. Effective only when expected_categories
        is not None. Keep at 0.0 for fair benchmark evaluation.
    topk : int
        Number of results to return.

    Returns
    -------
    list[dict]
        Re-ranked top-K dicts, each with an added "rerank_score" field.
    """
    if not candidates:
        return []

    # --- Tokenise query once ---
    q_tokens = _tokenize(query)

    # --- Expected categories as a set for O(1) lookup ---
    expected_set: set[str] = set(expected_categories) if expected_categories else set()

    # --- Score each candidate ---
    scored: List[tuple[float, dict]] = []
    for c in candidates:
        cosine = c["score"]                                          # raw, no min-max
        cap    = _weighted_caption_overlap(q_tokens, _tokenize(c.get("caption", "")))
        bonus  = 1.0 if (expected_set and c.get("pathology", "") in expected_set) else 0.0

        rerank_score = alpha * cosine + beta * cap + gamma * bonus
        scored.append((rerank_score, {**c, "rerank_score": round(rerank_score, 5)}))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:topk]]


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _FAKE_CANDIDATES = [
        {"id": "img_001.jpg", "score": 0.82, "pathology": "lung carcinoma",
         "caption": "H&E section showing invasive lung carcinoma with squamous differentiation"},
        {"id": "img_002.jpg", "score": 0.81, "pathology": "breast carcinoma",
         "caption": "Breast tissue section with ductal carcinoma in situ"},
        {"id": "img_003.jpg", "score": 0.79, "pathology": "lung carcinoma",
         "caption": "Lung adenocarcinoma with lepidic growth pattern"},
        {"id": "img_004.jpg", "score": 0.77, "pathology": "unknown",
         "caption": "Tissue section stained with hematoxylin and eosin"},
        {"id": "img_005.jpg", "score": 0.76, "pathology": "lung carcinoma",
         "caption": "Small cell lung carcinoma with necrosis and high mitotic rate"},
    ]

    QUERY = "invasive lung carcinoma squamous"
    q_tok = _tokenize(QUERY)
    print(f"Query: {QUERY!r}")
    print(f"Query tokens: {q_tok}\n")

    print("--- Before re-ranking (FAISS order) ---")
    for i, c in enumerate(_FAKE_CANDIDATES, 1):
        cap_tok = _tokenize(c["caption"])
        overlap = _weighted_caption_overlap(q_tok, cap_tok)
        print(f"  #{i}  cos={c['score']:.3f}  overlap={overlap:.3f}  "
              f"{c['pathology']:<22}  {c['caption'][:55]}")

    reranked = rerank_results(QUERY, _FAKE_CANDIDATES, topk=5)
    print("\n--- After re-ranking (alpha=0.75, beta=0.25, gamma=0.0) ---")
    for i, c in enumerate(reranked, 1):
        print(f"  #{i}  rerank={c['rerank_score']:.4f}  {c['pathology']:<22}  {c['caption'][:55]}")
