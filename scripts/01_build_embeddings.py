import os, sys, json, time, argparse
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime
from tqdm import tqdm
import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def iter_images(img_dir, allowed=None):
    for root, _, files in os.walk(img_dir):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTS:
                if allowed is None or f in allowed:
                    yield os.path.join(root, f)


def load_allowed_filenames(jsonl_path):
    allowed = set()
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Sırasıyla dene: bizim format (path), sonra alternatifler
            raw = rec.get("path") or rec.get("file_name") or rec.get("filename")
            if raw:
                allowed.add(os.path.basename(raw))
    return allowed


def process_batch(encoder, batch_pils, batch_paths, all_embs, all_ids):
    try:
        embs = encoder.encode_images(batch_pils)
        all_embs.append(embs)
        all_ids.extend([os.path.basename(p) for p in batch_paths])
    except torch.cuda.OutOfMemoryError:
        sys.exit(
            "\n[HATA] GPU belleği yetersiz (OutOfMemoryError).\n"
            "  → --batch_size değerini düşürün (örn. --batch_size 16 veya --batch_size 8)\n"
            "  → Çıkış yapılıyor."
        )
    except Exception:
        for img, path in zip(batch_pils, batch_paths):
            try:
                emb = encoder.encode_images([img])
                all_embs.append(emb)
                all_ids.append(os.path.basename(path))
            except Exception as e:
                print(f"[WARN] Atlandi: {path} - {e}")


def main():
    parser = argparse.ArgumentParser(description="Quilt-1M görsel embedding üretici")
    parser.add_argument("--img_dir", default="data/images")
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument(
        "--encoder",
        default=os.environ.get("ENCODER", "clip"),
        help="Encoder adı: clip | clip-b16 | quilt  (veya ENCODER env var)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--captions_jsonl",
        default=None,
        help="Opsiyonel: sadece bu JSONL'da geçen görselleri işle (filename match)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from encoders import get_encoder  # noqa: PLC0415

    os.makedirs(args.out_dir, exist_ok=True)

    allowed = None
    if args.captions_jsonl:
        allowed = load_allowed_filenames(args.captions_jsonl)
        print(f"Filtre aktif: {len(allowed)} benzersiz dosya adı yüklendi.")

    encoder = get_encoder(args.encoder, device=None)
    print(f"Encoder: {encoder.name} | dim={encoder.embed_dim} | device={encoder.device}")

    paths = list(iter_images(args.img_dir, allowed))
    if not paths:
        sys.exit(f"[HATA] {args.img_dir} altında görsel bulunamadı.")
    print(f"İşlenecek görsel sayısı: {len(paths)}")

    all_embs, all_ids = [], []
    t0 = time.time()

    batch_pils, batch_paths = [], []
    for p in tqdm(paths, desc="Embedding"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[WARN] Acilamadi: {p} - {e}")
            continue
        batch_pils.append(img)
        batch_paths.append(p)
        if len(batch_pils) == args.batch_size:
            process_batch(encoder, batch_pils, batch_paths, all_embs, all_ids)
            batch_pils, batch_paths = [], []

    if batch_pils:
        process_batch(encoder, batch_pils, batch_paths, all_embs, all_ids)

    if not all_embs:
        sys.exit("[HATA] Hiçbir embedding üretilemedi.")

    duration = time.time() - t0
    embeddings = np.vstack(all_embs).astype("float32")

    np.save(os.path.join(args.out_dir, "embeddings.npy"), embeddings)

    with open(os.path.join(args.out_dir, "ids.json"), "w", encoding="utf-8") as f:
        json.dump(all_ids, f, ensure_ascii=False, indent=2)

    model_info = {
        "name": encoder.name,
        "embed_dim": encoder.embed_dim,
        "device": encoder.device,
        "n_images": len(all_ids),
        "batch_size": args.batch_size,
        "duration_seconds": round(duration, 2),
        "creation_time": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.out_dir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print(
        f"\n[OK] Kaydedildi: {embeddings.shape} | "
        f"{len(all_ids)} gorsel | {duration:.1f}s | {args.out_dir}/"
    )


if __name__ == "__main__":
    main()
