"""
cross_encoder_reranker.py — BiomedCLIP (veya ms-marco fallback) ile cross-encoder re-ranking.

Akış:
    query + captions → text encoder → cosine similarity → sıralı top-K

Model seçimi:
    BiomedCLIP  — open_clip, histopatoloji PubMed metni üzerinde eğitilmiş (varsayılan)
    ms-marco    — sentence-transformers CrossEncoder, BiomedCLIP yüklenemezse fallback

Kullanım:
    from cross_encoder_reranker import load_cross_encoder, cross_encoder_rerank

    model, tokenizer, model_name = load_cross_encoder(device="cpu")
    reranked = cross_encoder_rerank(query, candidates, model, tokenizer, model_name)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Module-level singleton cache
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None
_MODEL_NAME: Optional[str] = None

_BIOMEDCLIP_HUB = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
_MIN_CAPTION_LEN = 5  # bu uzunluktan kısa caption'lar güvenilmez


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_cross_encoder(
    device: str = "cpu",
    model_name: str = "biomedclip",
) -> Tuple[object, Optional[object], str]:
    """Model singleton'ı döndürür. İlk çağrıda yükler, sonraki çağrılarda cache.

    Returns
    -------
    (model, tokenizer, model_name_str)
        BiomedCLIP: (open_clip model, open_clip tokenizer, "biomedclip")
        ms-marco  : (CrossEncoder, None, "ms-marco")
    """
    global _MODEL, _TOKENIZER, _MODEL_NAME

    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _MODEL_NAME

    if model_name == "biomedclip":
        try:
            import open_clip

            print(f"[INFO] BiomedCLIP yükleniyor: {_BIOMEDCLIP_HUB}")
            model, _, _ = open_clip.create_model_and_transforms(
                _BIOMEDCLIP_HUB, device=device
            )
            tokenizer = open_clip.get_tokenizer(_BIOMEDCLIP_HUB)
            model = model.to(device).eval()

            _MODEL, _TOKENIZER, _MODEL_NAME = model, tokenizer, "biomedclip"
            print(f"[OK] BiomedCLIP yüklendi (device={device})")
            return _MODEL, _TOKENIZER, _MODEL_NAME

        except Exception as exc:
            print(f"[WARN] BiomedCLIP yüklenemedi: {exc}")
            print("[INFO] Fallback: ms-marco MiniLM kullanılacak")
            model_name = "ms-marco"

    if model_name == "ms-marco":
        from sentence_transformers import CrossEncoder

        print("[INFO] ms-marco MiniLM yükleniyor...")
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device=device)
        _MODEL, _TOKENIZER, _MODEL_NAME = model, None, "ms-marco"
        print(f"[OK] ms-marco MiniLM yüklendi (device={device})")
        return _MODEL, _TOKENIZER, _MODEL_NAME

    raise ValueError(f"Bilinmeyen model_name: {model_name!r}. 'biomedclip' veya 'ms-marco' olmalı.")


# ---------------------------------------------------------------------------
# Core re-ranking function
# ---------------------------------------------------------------------------

def cross_encoder_rerank(
    query: str,
    candidates: List[dict],
    model,
    tokenizer,
    model_name: str = "biomedclip",
    device: str = "cpu",
    batch_size: int = 32,
    topk: int = 10,
) -> List[dict]:
    """Cross-encoder ile candidate listesini yeniden skorla, top-K döndür.

    Parameters
    ----------
    query      : Orijinal metin sorgusu.
    candidates : Her dict en az {"id", "score", "caption"} key'lerini içermeli.
    model      : load_cross_encoder() çıktısı (BiomedCLIP veya CrossEncoder).
    tokenizer  : BiomedCLIP tokenizer ya da None (ms-marco için).
    model_name : "biomedclip" veya "ms-marco".
    device     : torch device string.
    batch_size : Caption batch boyutu (BiomedCLIP).
    topk       : Döndürülecek sonuç sayısı.

    Returns
    -------
    list[dict]
        Yeniden sıralanmış top-K dict, her birinde "ce_score" field'ı eklenmiş.
    """
    if not candidates:
        return []

    captions = [c.get("caption") or "" for c in candidates]

    if model_name == "biomedclip":
        scores = _score_biomedclip(query, captions, model, tokenizer, device, batch_size)
    else:
        scores = _score_msmarco(query, captions, model, batch_size)

    # Boş/kısa caption'lara fallback: FAISS cosine'in yarısı
    for i, cap in enumerate(captions):
        if len(cap.strip()) < _MIN_CAPTION_LEN:
            scores[i] = candidates[i].get("score", 0.0) * 0.5

    scored = sorted(
        zip(scores, candidates),
        key=lambda x: -x[0],
    )
    return [
        {**cand, "ce_score": round(float(sc), 5)}
        for sc, cand in scored[:topk]
    ]


# ---------------------------------------------------------------------------
# BiomedCLIP scoring (bi-encoder cosine)
# ---------------------------------------------------------------------------

def _score_biomedclip(
    query: str,
    captions: List[str],
    model,
    tokenizer,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Query ve caption'ları BiomedCLIP text encoder ile encode eder, cosine döndürür."""

    # Query'yi 1 kez encode et
    with torch.no_grad():
        q_tokens = tokenizer([query]).to(device)
        q_emb = model.encode_text(q_tokens)
        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

    # Caption'ları batch'ler halinde encode et
    cap_emb_parts: list[torch.Tensor] = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i : i + batch_size]
        with torch.no_grad():
            c_tokens = tokenizer(batch).to(device)
            c_embs = model.encode_text(c_tokens)
            c_embs = c_embs / c_embs.norm(dim=-1, keepdim=True)
        cap_emb_parts.append(c_embs)

    caption_embs = torch.cat(cap_emb_parts, dim=0)  # (N, D)

    # Cosine = dot product (her ikisi L2-normalized)
    scores = (q_emb @ caption_embs.T).squeeze(0).cpu().numpy()
    return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# ms-marco scoring (CrossEncoder)
# ---------------------------------------------------------------------------

def _score_msmarco(
    query: str,
    captions: List[str],
    model,
    batch_size: int,
) -> np.ndarray:
    """CrossEncoder.predict() ile (query, caption) çiftlerini skorlar."""
    pairs = [(query, cap) for cap in captions]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return np.array(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _FAKE_CANDIDATES = [
        {
            "id": "img_001.jpg",
            "score": 0.82,
            "pathology": "lung adenocarcinoma",
            "caption": "lung adenocarcinoma acinar pattern with mucin production and pleural invasion",
        },
        {
            "id": "img_002.jpg",
            "score": 0.80,
            "pathology": "breast carcinoma",
            "caption": "invasive breast carcinoma with ductal differentiation and lymph node metastasis",
        },
        {
            "id": "img_003.jpg",
            "score": 0.78,
            "pathology": "kidney",
            "caption": "kidney glomerulonephritis with mesangial expansion and crescents",
        },
        {
            "id": "img_004.jpg",
            "score": 0.76,
            "pathology": "unknown",
            "caption": "generic image",
        },
        {
            "id": "img_005.jpg",
            "score": 0.74,
            "pathology": "unknown",
            "caption": "",
        },
    ]

    QUERY = "lung adenocarcinoma"
    print(f"Query: {QUERY!r}\n")
    print("--- Kandidatlar (FAISS sırası) ---")
    for i, c in enumerate(_FAKE_CANDIDATES, 1):
        cap_preview = (c["caption"][:60] or "(bos)") + ("..." if len(c["caption"]) > 60 else "")
        print(f"  #{i}  faiss={c['score']:.3f}  {c['pathology']:<22}  {cap_preview!r}")

    model, tokenizer, model_name = load_cross_encoder(device="cpu", model_name="biomedclip")
    print(f"\n[Aktif model: {model_name}]\n")

    reranked = cross_encoder_rerank(
        QUERY, _FAKE_CANDIDATES, model, tokenizer,
        model_name=model_name, device="cpu", batch_size=32, topk=5,
    )

    print("--- CE Rerank sonrası (ce_score sırası) ---")
    for i, c in enumerate(reranked, 1):
        cap_preview = (c.get("caption", "")[:55] or "(bos)") + ("..." if len(c.get("caption", "")) > 55 else "")
        print(
            f"  #{i}  ce_score={c['ce_score']:.4f}  "
            f"faiss={c['score']:.3f}  "
            f"{c.get('pathology', ''):<22}  {cap_preview!r}"
        )
