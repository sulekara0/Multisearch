"""
thresholds.py — Domain-gate decision / status packaging.

Design history: the original two-stage absolute-cosine threshold (domain +
quality) was discarded after calibration showed text->image cosine cannot
separate domain (see scripts/domain_gate.py docstring). Domain detection now
happens in text space (DomainGate); this module only packages the decision into
a UI-friendly status dict.

The per-result quality filter was dropped intentionally: in-domain top-10
cosines are nearly flat (~0.32), so there is no meaningful absolute-cosine
quality gradient to filter on — within-domain differentiation is the job of the
caption re-ranker, not a threshold.

Status values: "ok" | "domain_out" | "empty".
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class GateResult(TypedDict):
    status: str            # "ok" | "domain_out" | "empty"
    results: List[dict]
    message: Optional[str]


def apply_domain_gate(
    results: List[dict],
    domain_score: float,
    domain_threshold: float = 0.80,
) -> GateResult:
    """Decide whether a query is in-domain and package the result list.

    Parameters
    ----------
    results : list[dict]
        Retrieval results (already re-ranked / truncated by the caller).
    domain_score : float
        Text-space domain score for the query (DomainGate.score). NOT the
        retrieval cosine — that scale cannot gate domain.
    domain_threshold : float
        Minimum domain score to treat the query as in-domain.

    Returns
    -------
    GateResult
        status + (possibly empty) results + an optional user-facing message.
    """
    if domain_score < domain_threshold:
        return {
            "status": "domain_out",
            "results": [],
            "message": (
                f"Bu sorgu histopatoloji alanıyla uyumlu görünmüyor "
                f"(alan benzerliği: {domain_score:.2f}, eşik: {domain_threshold:.2f}). "
                f"Daha spesifik histopatolojik terimler deneyin."
            ),
        }

    if not results:
        return {"status": "empty", "results": [], "message": "Sonuç bulunamadı"}

    return {"status": "ok", "results": results, "message": None}
