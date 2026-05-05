# app.py — Histopatoloji Multimodal Arama Demo (Streamlit)
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
ART_ROOT = ROOT / "artifacts"

def _find_img_dir() -> Path:
    for name in ("data/images", "data/images_test"):
        p = ROOT / name
        if p.exists() and any(p.iterdir()):
            return p
    return ROOT / "data" / "images"

def _find_captions() -> Path:
    for name in ("captions_quilt.jsonl", "captions_quilt_test.jsonl"):
        p = ART_ROOT / name
        if p.exists():
            return p
    return ART_ROOT / "captions_quilt.jsonl"

IMG_DIR = _find_img_dir()
CAPTIONS_PATH = _find_captions()
SCRIPTS_DIR = ROOT / "scripts"

os.chdir(ROOT)  # guarantee CWD = project root so FAISS relpath avoids non-ASCII path issues

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ENCODERS = ["clip", "quilt"]
ENCODER_LABELS = {
    "clip":  "Generic CLIP (ViT-B/32)",
    "quilt": "QuiltNet (ViT-B/32)",
}
MODES = ["Tek Model", "Karşılaştırma", "Görsel→Görsel"]

# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------
def check_artifacts(encoder_name: str) -> Tuple[bool, str]:
    art_dir = ART_ROOT / encoder_name
    required = ["embeddings.npy", "ids.json", "faiss_index.bin"]
    missing = [f for f in required if not (art_dir / f).exists()]
    if not missing:
        return True, ""
    cmd1 = f"python scripts/01_build_embeddings.py --encoder {encoder_name} --out_dir artifacts/{encoder_name}"
    cmd2 = f"python scripts/faiss_search.py build --art_dir artifacts/{encoder_name}"
    msg = (
        f"**{encoder_name}** için eksik dosyalar: `{', '.join(missing)}`\n\n"
        f"Çalıştır:\n```\n{cmd1}\n{cmd2}\n```"
    )
    return False, msg

# ---------------------------------------------------------------------------
# Cached resources (one entry per (name) or (art_dir) key)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Model yükleniyor…")
def load_encoder(name: str):
    from encoders import get_encoder
    return get_encoder(name)

@st.cache_resource(show_spinner="Index yükleniyor…")
def load_index_data(art_dir: str) -> Tuple:
    from faiss_search import load_index, load_embeddings, load_captions
    index, metric = load_index(art_dir)
    _, ids = load_embeddings(art_dir)
    cap_map: Dict[str, dict] = {}
    if CAPTIONS_PATH.exists():
        cap_map = load_captions(CAPTIONS_PATH)
    return index, metric, ids, cap_map

# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------
def _raw_faiss(index, metric: str, ids: List[str], cap_map: Dict[str, dict],
               q_vec: np.ndarray, topk: int) -> List[dict]:
    if metric == "ip":
        scores, I = index.search(q_vec, topk)
        raw_scores = scores[0].tolist()
    else:
        dists, I = index.search(q_vec, topk)
        raw_scores = (-dists[0]).tolist()
    results = []
    for rank_i, idx in enumerate(I[0].tolist()):
        img_id = ids[idx]
        cap_rec = cap_map.get(Path(img_id).name, cap_map.get(img_id, {}))
        results.append({
            "id":        img_id,
            "score":     float(raw_scores[rank_i]),
            "caption":   cap_rec.get("caption", ""),
            "pathology": cap_rec.get("pathology", ""),
        })
    return results

def run_text_search(encoder_name: str, query: str, topk: int) -> Tuple[List[dict], float, float]:
    encoder = load_encoder(encoder_name)
    index, metric, ids, cap_map = load_index_data(str(ART_ROOT / encoder_name))
    t0 = time.perf_counter()
    q_vec = encoder.encode_texts([query])
    t1 = time.perf_counter()
    results = _raw_faiss(index, metric, ids, cap_map, q_vec, topk)
    t2 = time.perf_counter()
    return results, (t1 - t0) * 1000, (t2 - t1) * 1000

def run_image_search(encoder_name: str, img: Image.Image, topk: int) -> Tuple[List[dict], float, float]:
    encoder = load_encoder(encoder_name)
    index, metric, ids, cap_map = load_index_data(str(ART_ROOT / encoder_name))
    t0 = time.perf_counter()
    q_vec = encoder.encode_images([img])
    t1 = time.perf_counter()
    results = _raw_faiss(index, metric, ids, cap_map, q_vec, topk)
    t2 = time.perf_counter()
    return results, (t1 - t0) * 1000, (t2 - t1) * 1000

# ---------------------------------------------------------------------------
# Card renderer
# ---------------------------------------------------------------------------
def draw_card(col, result: dict, is_overlap: bool = False) -> None:
    img_path = IMG_DIR / Path(result["id"]).name
    with col:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            st.warning(f"Açılamadı: {Path(result['id']).name}")
            return
        overlap_tag = "↔  " if is_overlap else ""
        st.image(img, caption=f"{overlap_tag}{result['score']:.3f}", use_container_width=True)
        if result["pathology"]:
            st.caption(f"🔬 {result['pathology']}")
        if result["caption"]:
            preview = result["caption"][:80] + ("…" if len(result["caption"]) > 80 else "")
            st.caption(preview)
            with st.expander("Tam caption"):
                st.write(result["caption"])

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Histopatoloji Multimodal Arama",
    page_icon="🔬",
    layout="wide",
)
st.title("🔬 Histopatoloji Multimodal Arama")
st.caption("Generic CLIP vs QuiltNet — Alan adaptasyonu karşılaştırması | Quilt-1M + FAISS")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("search_history", [])
st.session_state.setdefault("query_input", "")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Ayarlar")

    mode = st.radio("Mod", MODES, index=0)

    if mode in ("Tek Model", "Görsel→Görsel"):
        encoder_name = st.selectbox(
            "Model",
            ENCODERS,
            format_func=lambda x: ENCODER_LABELS[x],
            key="encoder_select",
        )
    else:
        encoder_name = None  # karşılaştırma: her iki encoder kullanılır

    topk = st.slider("Top-K", min_value=4, max_value=20, value=8, step=4)

    st.divider()

    # Artifact health check — sidebar'da uyarı, ana alan etkilenir
    if mode == "Karşılaştırma":
        checks_ok = True
        for enc in ENCODERS:
            ok, err_msg = check_artifacts(enc)
            if not ok:
                st.error(err_msg)
                checks_ok = False
    else:
        checks_ok, err_msg = check_artifacts(encoder_name)
        if not checks_ok:
            st.error(err_msg)

    st.divider()

    # Son aramalar
    if st.session_state.search_history:
        st.subheader("Son Aramalar")
        for past_q in st.session_state.search_history:
            if st.button(past_q, key=f"hist_{past_q}", use_container_width=True):
                st.session_state["tq_single"] = past_q
                st.session_state["tq_compare"] = past_q
                st.rerun()

# ---------------------------------------------------------------------------
# Main: Tek Model
# ---------------------------------------------------------------------------
if mode == "Tek Model":
    query = st.text_input(
        "Metin sorgusu",
        placeholder="lung tissue with cancer cells",
        disabled=not checks_ok,
        key="tq_single",
    )

    if query.strip() and checks_ok:
        results, enc_ms, srch_ms = run_text_search(encoder_name, query.strip(), topk)

        st.caption(
            f"⏱ Arama: **{enc_ms + srch_ms:.1f}ms** "
            f"(encode: {enc_ms:.1f}ms, search: {srch_ms:.1f}ms) "
            f"| {len(results)} sonuç"
        )

        # history
        if query.strip() not in st.session_state.search_history:
            st.session_state.search_history.insert(0, query.strip())
            st.session_state.search_history = st.session_state.search_history[:5]

        cols = st.columns(4)
        for i, r in enumerate(results):
            draw_card(cols[i % 4], r)

# ---------------------------------------------------------------------------
# Main: Karşılaştırma
# ---------------------------------------------------------------------------
elif mode == "Karşılaştırma":
    query = st.text_input(
        "Metin sorgusu",
        placeholder="invasive ductal carcinoma",
        disabled=not checks_ok,
        key="tq_compare",
    )

    if query.strip() and checks_ok:
        clip_results,  clip_enc_ms,  clip_srch_ms  = run_text_search("clip",  query.strip(), topk)
        quilt_results, quilt_enc_ms, quilt_srch_ms = run_text_search("quilt", query.strip(), topk)

        # history
        if query.strip() not in st.session_state.search_history:
            st.session_state.search_history.insert(0, query.strip())
            st.session_state.search_history = st.session_state.search_history[:5]

        # Overlap counter
        clip_ids  = {r["id"] for r in clip_results}
        quilt_ids = {r["id"] for r in quilt_results}
        overlap   = clip_ids & quilt_ids
        overlap_count = len(overlap)
        overlap_pct   = (overlap_count / topk) * 100
        clip_only     = len(clip_ids - quilt_ids)
        quilt_only    = len(quilt_ids - clip_ids)

        st.info(
            f"🟢 Top-{topk}'te **{overlap_count}** ortak görsel (%{overlap_pct:.1f}) "
            f"| CLIP'e özel: **{clip_only}**, QuiltNet'e özel: **{quilt_only}**"
        )

        # Latency row
        lat_l, lat_r = st.columns(2)
        with lat_l:
            st.caption(f"⏱ CLIP — encode: {clip_enc_ms:.1f}ms, search: {clip_srch_ms:.1f}ms")
        with lat_r:
            st.caption(f"⏱ QuiltNet — encode: {quilt_enc_ms:.1f}ms, search: {quilt_srch_ms:.1f}ms")

        # Column headers
        hdr_l, hdr_r = st.columns(2)
        with hdr_l:
            st.subheader("Generic CLIP")
        with hdr_r:
            st.subheader("QuiltNet")

        # Results: rank-aligned rows
        for i in range(max(len(clip_results), len(quilt_results))):
            row_l, row_r = st.columns(2)
            if i < len(clip_results):
                r = clip_results[i]
                draw_card(row_l, r, is_overlap=(r["id"] in overlap))
            if i < len(quilt_results):
                r = quilt_results[i]
                draw_card(row_r, r, is_overlap=(r["id"] in overlap))

# ---------------------------------------------------------------------------
# Main: Görsel→Görsel
# ---------------------------------------------------------------------------
elif mode == "Görsel→Görsel":
    uploaded = st.file_uploader(
        "Sorgu görseli yükle (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        disabled=not checks_ok,
    )

    if not checks_ok:
        st.error("Seçili model için artifact eksik. Sidebar'daki komutları çalıştırın.")

    if uploaded and checks_ok:
        query_img = Image.open(uploaded).convert("RGB")

        prev_col, info_col = st.columns([1, 3])
        with prev_col:
            st.image(query_img, caption="Sorgu görseli", use_container_width=True)
        with info_col:
            st.markdown(f"**Boyut:** {query_img.size[0]}×{query_img.size[1]} px")
            st.markdown(f"**Model:** {ENCODER_LABELS[encoder_name]}")

        results, enc_ms, srch_ms = run_image_search(encoder_name, query_img, topk)

        st.caption(
            f"⏱ Arama: **{enc_ms + srch_ms:.1f}ms** "
            f"(encode: {enc_ms:.1f}ms, search: {srch_ms:.1f}ms)"
        )

        st.subheader("Benzer Görseller")
        cols = st.columns(4)
        for i, r in enumerate(results):
            draw_card(cols[i % 4], r)
