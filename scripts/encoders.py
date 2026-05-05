"""
encoders.py — Model-agnostic embedding encoder katmanı.

Bu modül, generic CLIP (HuggingFace) ve domain-specific modeller (QuiltNet vb.,
open_clip ile yüklenir) için ortak bir arayüz sağlar. Pipeline'ın geri kalanı
hangi modeli kullandığını bilmek zorunda değildir.

Kullanım:
    from encoders import get_encoder

    encoder = get_encoder("quilt")           # veya "clip"
    img_embeds = encoder.encode_images(pil_images)   # (N, D) float32, L2-normalized
    txt_embeds = encoder.encode_texts(["sample text"])  # (1, D) float32, L2-normalized

Yeni model eklemek için:
    1. BaseEncoder'dan miras alan yeni bir sınıf yazın
    2. ENCODER_REGISTRY'e ekleyin
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _pick_device(device: str | None = None) -> str:
    """Otomatik cihaz seçimi: cuda > mps > cpu."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """L2 normalize on last dim (cosine similarity için)."""
    return torch.nn.functional.normalize(x, p=2, dim=-1)


# =============================================================================
# Abstract Base Class
# =============================================================================

class BaseEncoder(ABC):
    """Tüm encoder'ların implement etmesi gereken arayüz."""

    name: str = "base"
    embed_dim: int = 0  # Subclass'lar set eder

    @abstractmethod
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """PIL görsel listesini (N, D) float32 normalized embedding'e çevirir."""
        ...

    @abstractmethod
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Metin listesini (N, D) float32 normalized embedding'e çevirir."""
        ...

    def info(self) -> dict:
        """Model meta-bilgisi (raporlama için)."""
        return {
            "name": self.name,
            "embed_dim": self.embed_dim,
            "device": getattr(self, "device", "unknown"),
        }


# =============================================================================
# Generic CLIP (HuggingFace transformers)
# =============================================================================

class CLIPEncoder(BaseEncoder):
    """OpenAI CLIP veya HuggingFace üzerinde transformers API'siyle yüklenebilen
    her CLIP modeli için. Mevcut projedeki davranışla uyumlu."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32",
                 device: str | None = None):
        from transformers import CLIPModel, CLIPProcessor

        self.name = f"clip:{model_id}"
        self.device = _pick_device(device)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.embed_dim = self.model.config.projection_dim
        print(f"[INFO] Loaded {self.name} on {self.device} (dim={self.embed_dim})")

    @torch.inference_mode()
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        # transformers >=5 returns BaseModelOutputWithPooling instead of a tensor
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output
        feats = _l2_normalize(feats)
        return feats.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt",
                                padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output
        feats = _l2_normalize(feats)
        return feats.cpu().numpy().astype(np.float32)


# =============================================================================
# OpenCLIP (QuiltNet, BiomedCLIP, PLIP, vs.)
# =============================================================================

class OpenCLIPEncoder(BaseEncoder):
    """open_clip kütüphanesi ile yüklenen modeller için.
    QuiltNet ve birçok domain-specific medical CLIP modeli bu kategoride."""

    def __init__(self, model_name: str = "hf-hub:wisdomik/QuiltNet-B-32",
                 device: str | None = None):
        import open_clip

        self.name = f"openclip:{model_name}"
        self.device = _pick_device(device)

        # open_clip create_model_and_transforms 3 şey döndürür:
        # model, training_preprocess, eval_preprocess
        # Inference için 3.üyü (eval_preprocess) kullanıyoruz.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Embedding boyutunu küçük bir dummy ile çıkar
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            self.embed_dim = self.model.encode_image(dummy).shape[-1]

        print(f"[INFO] Loaded {self.name} on {self.device} (dim={self.embed_dim})")

    @torch.inference_mode()
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        # open_clip preprocess tek görsel alıyor, batch için manuel stack
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        feats = self.model.encode_image(batch)
        feats = _l2_normalize(feats)
        return feats.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = _l2_normalize(feats)
        return feats.cpu().numpy().astype(np.float32)


# =============================================================================
# Factory
# =============================================================================

ENCODER_REGISTRY = {
    # Generic CLIP varyantları (HuggingFace transformers ile)
    "clip":       lambda **kw: CLIPEncoder("openai/clip-vit-base-patch32", **kw),
    "clip-b32":   lambda **kw: CLIPEncoder("openai/clip-vit-base-patch32", **kw),
    "clip-b16":   lambda **kw: CLIPEncoder("openai/clip-vit-base-patch16", **kw),
    "clip-l14":   lambda **kw: CLIPEncoder("openai/clip-vit-large-patch14", **kw),

    # Histopatoloji modelleri (open_clip ile)
    "quilt":      lambda **kw: OpenCLIPEncoder("hf-hub:wisdomik/QuiltNet-B-32", **kw),
    "quilt-b32":  lambda **kw: OpenCLIPEncoder("hf-hub:wisdomik/QuiltNet-B-32", **kw),
    "quilt-b16":  lambda **kw: OpenCLIPEncoder("hf-hub:wisdomik/QuiltNet-B-16", **kw),
    "quilt-pmb":  lambda **kw: OpenCLIPEncoder("hf-hub:wisdomik/QuiltNet-B-16-PMB", **kw),
}


def get_encoder(name: str = "clip", device: str | None = None) -> BaseEncoder:
    """Encoder factory. İsim verilirse uygun encoder'ı oluşturup döndürür.

    Args:
        name: ENCODER_REGISTRY'deki anahtarlardan biri ("clip", "quilt", vs.)
              veya doğrudan bir model_id (string) — heuristic ile tip tahmin
              edilir.
        device: "cuda" / "mps" / "cpu" / None (otomatik)

    Returns:
        BaseEncoder instance
    """
    name_lower = name.lower().strip()

    # Registry'de varsa direkt döndür
    if name_lower in ENCODER_REGISTRY:
        return ENCODER_REGISTRY[name_lower](device=device)

    # Heuristic: "hf-hub:" prefix'i veya quilt/biomedclip içeriyorsa open_clip
    if name.startswith("hf-hub:") or "quilt" in name_lower or "biomed" in name_lower:
        return OpenCLIPEncoder(name, device=device)

    # Aksi halde HuggingFace CLIPModel formatında varsay
    return CLIPEncoder(name, device=device)


def list_encoders() -> List[str]:
    """Mevcut encoder isimlerini listeler."""
    return list(ENCODER_REGISTRY.keys())


# =============================================================================
# Hızlı sanity test (modülü direkt çalıştırırsanız)
# =============================================================================

if __name__ == "__main__":
    import sys

    enc_name = sys.argv[1] if len(sys.argv) > 1 else "clip"
    print(f"\n=== Testing encoder: {enc_name} ===\n")

    enc = get_encoder(enc_name)
    print("Info:", enc.info())

    # Dummy beyaz görsel ile test
    dummy_img = Image.new("RGB", (224, 224), color="white")
    img_emb = enc.encode_images([dummy_img])
    print(f"Image embedding shape: {img_emb.shape}, "
          f"norm: {np.linalg.norm(img_emb[0]):.4f}")

    txt_emb = enc.encode_texts(["a histopathology slide"])
    print(f"Text embedding shape: {txt_emb.shape}, "
          f"norm: {np.linalg.norm(txt_emb[0]):.4f}")

    # Cosine similarity (normalized vektörler için dot product)
    sim = float(img_emb[0] @ txt_emb[0])
    print(f"Image-text cosine similarity: {sim:.4f}")
