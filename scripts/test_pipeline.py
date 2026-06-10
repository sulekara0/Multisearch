"""
test_pipeline.py — End-to-end check of the app's search_pipeline logic
(headless, no Streamlit runtime). Mirrors app.search_pipeline exactly:
domain-gate (CLIP) → early-exit OR FAISS retrieval (QuiltNet) → optional V5b
rerank → apply_domain_gate. Confirms statuses + that domain_out skips FAISS.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from encoders import get_encoder
from faiss_search import load_index, load_embeddings, load_captions
from domain_gate import DomainGate
from reranker import rerank_results
from thresholds import apply_domain_gate

ART_ROOT = ROOT / "artifacts"
RETRIEVAL_ENC = "quilt"
TOPK = 8
DOMAIN_THRESHOLD = 0.80


def _l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _raw_faiss(index, metric, ids, cap_map, q_vec, topk):
    """Copy of app._raw_faiss — basename caption lookup (so rerank gets signal)."""
    if metric == "ip":
        scores, I = index.search(_l2(q_vec), topk)
        raw = scores[0].tolist()
    else:
        dists, I = index.search(q_vec, topk)
        raw = (-dists[0]).tolist()
    out = []
    for rank_i, idx in enumerate(I[0].tolist()):
        img_id = ids[idx]
        rec = cap_map.get(Path(img_id).name, cap_map.get(img_id, {}))
        out.append({"id": img_id, "score": float(raw[rank_i]),
                    "caption": rec.get("caption", ""), "pathology": rec.get("pathology", "")})
    return out


def pipeline(query, encoder, index, metric, ids, cap_map, gate, topk, enable_rerank, domain_threshold):
    domain_score = gate.score(query)
    faiss_called = False
    if domain_score < domain_threshold:
        out = apply_domain_gate([], domain_score, domain_threshold)
        out.update(domain_score=domain_score, faiss_called=faiss_called)
        return out
    fetch_k = topk * 5 if enable_rerank else topk
    q_vec = encoder.encode_texts([query])
    results = _raw_faiss(index, metric, ids, cap_map, q_vec, fetch_k)
    faiss_called = True
    if enable_rerank:
        results = rerank_results(query, results, topk=topk, alpha=0.75, beta=0.25)
    else:
        results = results[:topk]
    out = apply_domain_gate(results, domain_score, domain_threshold)
    out.update(domain_score=domain_score, faiss_called=faiss_called)
    return out


if __name__ == "__main__":
    gate = DomainGate(get_encoder("clip"))
    enc = get_encoder(RETRIEVAL_ENC)
    index, metric = load_index(str(ART_ROOT / RETRIEVAL_ENC))
    _, ids = load_embeddings(str(ART_ROOT / RETRIEVAL_ENC))
    cap_map = load_captions(ART_ROOT / "captions_quilt.jsonl") if (ART_ROOT / "captions_quilt.jsonl").exists() else {}

    CASES = [
        ("lung cancer biopsy",     True,  "ok"),
        ("invasive ductal carcinoma", True, "ok"),
        ("medical image",          True,  "ok"),
        ("araba motoru",           True,  "domain_out"),
        ("chocolate cake recipe",  True,  "domain_out"),
    ]

    print(f"\n{'='*70}\n PIPELINE TEST — gate=CLIP, retrieval={RETRIEVAL_ENC}, "
          f"topk={TOPK}, d={DOMAIN_THRESHOLD}\n{'='*70}")
    for query, rerank_on, expected in CASES:
        res = pipeline(query, enc, index, metric, ids, cap_map, gate, TOPK, rerank_on, DOMAIN_THRESHOLD)
        match = "OK" if res["status"] == expected else "XX"
        print(f'\n[{match}] "{query}"  rerank={rerank_on}')
        print(f"     status={res['status']}  domain={res['domain_score']:.2f}  "
              f"faiss_called={res['faiss_called']}  n={len(res['results'])}")
        for r in res["results"][:3]:
            rs = r.get("rerank_score")
            rs_str = f" rerank={rs:.4f}" if rs is not None else ""
            print(f"       cos={r['score']:.3f}{rs_str}  {r['pathology'][:18]:<18} {Path(r['id']).name}")

    # explicit early-exit assertion
    out = pipeline("araba motoru", enc, index, metric, ids, cap_map, gate, TOPK, True, DOMAIN_THRESHOLD)
    assert out["status"] == "domain_out" and out["faiss_called"] is False, "early-exit broken!"
    print("\n[assert] domain_out short-circuits FAISS ✓")
