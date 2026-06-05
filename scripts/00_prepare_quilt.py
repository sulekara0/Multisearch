"""
00_prepare_quilt.py — Quilt-1M subset hazırlama

Adımlar:
  1. Görsel kaynağını indexle  -> {filename: path} dict
       ZIP modu   : images_part_*.zip dosyaları taranır
       Klasör modu: extract edilmiş klasör(ler) os.walk ile taranır
                    Virgülle ayrılmış birden fazla kaynak kabul edilir.
  2. CSV'yi chunk'larla tara, filtrele, kaynakta olan + split eşleşenleri tut
  3. Sampling:
       stratified (default): primary_pathology'ye göre orantılı örnekleme
       balanced (yeni)      : BALANCED_TARGETS dict ile az temsil edilenlere ağırlık ver
  4. Görselleri hedefe yaz
       ZIP modu   : ZipFile.read + write (lazy handle cache + ExitStack)
       Klasör modu: os.link (hard link) → başarısız olursa shutil.copy2 fallback
  5. E:\Multisearch_data\ altına captions_quilt.jsonl ve subset_stats.json yaz

Kullanım:
  python scripts/00_prepare_quilt.py --zip_dir data/zips --n 100
  python scripts/00_prepare_quilt.py \
      --csv "C:\\data\\quilt_1M_lookup.csv" \
      --img_source_dir "C:\\path1,E:\\path2,E:\\path3" \
      --sampling_mode balanced --clean
"""

import argparse
import ast
import contextlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CSV     = "data/quilt_1M_lookup.csv"
DEFAULT_ZIP_DIR = "data/zips"
DEFAULT_OUT_DIR = r"E:\Multisearch_data\images"
DEFAULT_JSONL   = r"E:\Multisearch_data\captions_quilt.jsonl"
DEFAULT_STATS   = r"E:\Multisearch_data\subset_stats.json"
DEFAULT_N       = 10_000
DEFAULT_SEED    = 42
DEFAULT_SPLIT   = "train"
CHUNK_SIZE      = 50_000
NEEDED_COLS     = ["image_path", "caption", "pathology", "not_histology", "split"]

# ---------------------------------------------------------------------------
# Balanced sampling hedefleri
# ---------------------------------------------------------------------------
BALANCED_TARGETS = {
    "Dermatopathology": 20000,
    "unknown":          25000,
    "Gastrointestinal": 18000,
    "Pulmonary":        14000,
    "Gynecologic":      14000,
    "Soft tissue":      12000,
    "Breast":           12000,
    "Renal":            10000,
    "Hematopathology":  10000,
    "Genitourinary":     8000,
    "Bone":              7000,
    "Neuropathology":    6000,
    "Cardiac":           5000,
    "Hepatopathology":   5000,
    "Endocrine":         5000,
    "Head and Neck":     5000,
    "Cytopathology":     4000,
    "Ophthalmic":        3000,
    "Pediatric":         3000,
    "Unknown":           4000,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_pathology(raw) -> str:
    """Stringified liste -> birincil etiket. NaN / boş -> 'unknown'."""
    if pd.isna(raw) or str(raw).strip() in ("", "[]", "nan"):
        return "unknown"
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    except Exception:
        pass
    return str(raw).strip()


# ---------------------------------------------------------------------------
# Adım 1 — Kaynak index (ZIP veya klasör)
# ---------------------------------------------------------------------------

def build_zip_index(zip_dir: str) -> dict:
    """
    zip_dir içindeki tüm images_part_*.zip dosyalarını tara.
    Dönüş: {dosya_adı: zip_yolu_str}
    """
    zip_paths = sorted(Path(zip_dir).glob("images_part_*.zip"))
    if not zip_paths:
        raise FileNotFoundError(
            f"'{zip_dir}' içinde images_part_*.zip bulunamadı."
        )

    print(f"[1/4] ZIP indexleniyor — {len(zip_paths)} dosya bulundu:")
    index: dict[str, str] = {}

    for zp in zip_paths:
        with zipfile.ZipFile(zp, "r") as z:
            names = z.namelist()
        count = 0
        for n in names:
            if "/" in n and not n.endswith("/"):
                fname = n.split("/", 1)[-1]
                index[fname] = str(zp)
                count += 1
        print(f"      {zp.name}: {count:>7,} dosya")

    print(f"      {'TOPLAM':<30} {len(index):>7,} dosya — {len(zip_paths)} ZIP")
    return index


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def build_folder_index(img_source_dirs: list[str]) -> dict:
    """
    Bir veya birden fazla extract edilmiş klasörü os.walk ile tara.
    Aynı dosya adı birden fazla klasörde varsa ilk bulunan kazanır (uyarı verilir).
    Dönüş: {dosya_adı: tam_yol_str}
    """
    print(f"[1/4] Klasör(ler) indexleniyor — {len(img_source_dirs)} kaynak:")
    index: dict[str, str] = {}
    duplicates = 0

    for src_dir in img_source_dirs:
        dir_count = 0
        print(f"      Taraniyor: {src_dir}")
        for root, _, files in os.walk(src_dir):
            for fname in files:
                if Path(fname).suffix.lower() in IMAGE_EXTS:
                    if fname in index:
                        duplicates += 1
                        if duplicates <= 5:
                            print(f"  [WARN] Duplicate, ilk kaynak korundu: {fname}")
                        elif duplicates == 6:
                            print("  [WARN] Daha fazla duplicate uyarısı bastırılıyor...")
                    else:
                        index[fname] = str(Path(root) / fname)
                        dir_count += 1
        print(f"             -> {dir_count:>7,} yeni dosya eklendi")

    if duplicates > 0:
        print(f"  [WARN] Toplam {duplicates:,} duplicate dosya atlandı (ilk kaynak korundu).")
    print(f"      {'TOPLAM':<30} {len(index):>7,} dosya — {len(img_source_dirs)} klasör")
    return index


# ---------------------------------------------------------------------------
# Adım 2 — CSV tarama
# ---------------------------------------------------------------------------

def collect_candidates(csv_path: str, zip_index: dict, split: str) -> pd.DataFrame:
    """
    CSV'yi chunk'larla tara:
      - not_histology == 0
      - caption dolu
      - split eşleşme
      - image_path kaynakta mevcut
    """
    print(f"\n[2/4] CSV taranıyor (split='{split}', chunk={CHUNK_SIZE:,}): {csv_path}")
    chunks = []
    total_read = 0
    zip_fileset = set(zip_index.keys())

    reader = pd.read_csv(
        csv_path,
        usecols=NEEDED_COLS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
    for chunk in tqdm(reader, desc="  chunk", unit="chunk"):
        total_read += len(chunk)
        mask = (
            (chunk["not_histology"] == 0)
            & (chunk["split"] == split)
            & chunk["caption"].notna()
            & (chunk["caption"].str.strip() != "")
            & chunk["image_path"].isin(zip_fileset)
        )
        chunks.append(chunk[mask])

    df = pd.concat(chunks, ignore_index=True)
    print(f"      Toplam okunan: {total_read:,}  |  Aday: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Adım 3a — Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """primary_pathology'ye göre orantılı örnekleme."""
    print(f"\n[3/4] Stratified sampling: n={n:,}, seed={seed}")

    df = df.copy()
    df["primary_pathology"] = df["pathology"].apply(parse_pathology)

    counts = df["primary_pathology"].value_counts()
    total  = len(df)
    n_cats = counts.shape[0]
    print(f"      Kategoriler: {n_cats} adet  |  Kaynak havuz: {total:,}")

    sampled_parts = []
    for label, count in counts.items():
        quota = max(1, round(n * count / total))
        part  = df[df["primary_pathology"] == label]
        take  = min(quota, len(part))
        sampled_parts.append(part.sample(take, random_state=seed))

    result = pd.concat(sampled_parts, ignore_index=True)

    if len(result) > n:
        result = result.sample(n, random_state=seed)
    elif len(result) < n:
        remaining = df[~df.index.isin(result.index)]
        extra = min(n - len(result), len(remaining))
        if extra > 0:
            result = pd.concat(
                [result, remaining.sample(extra, random_state=seed)],
                ignore_index=True,
            )

    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"      Seçilen: {len(result):,} görsel, {result['primary_pathology'].nunique()} kategori")
    return result


# ---------------------------------------------------------------------------
# Adım 3b — Balanced sampling
# ---------------------------------------------------------------------------

def balanced_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    BALANCED_TARGETS dict'e göre az temsil edilen kategorilere ağırlıklı örnekleme.
    --n ignore edilir; toplam ~sum(BALANCED_TARGETS) civarı olur.
    """
    print(f"\n[3/4] Balanced sampling: seed={seed}")
    print(f"      Hedef toplam: {sum(BALANCED_TARGETS.values()):,} görsel, "
          f"{len(BALANCED_TARGETS)} kategori")

    df = df.copy()
    df["primary_pathology"] = df["pathology"].apply(parse_pathology)

    available_counts = df["primary_pathology"].value_counts().to_dict()

    sampled_parts = []
    summary_rows = []

    for label, target in BALANCED_TARGETS.items():
        pool = df[df["primary_pathology"] == label]
        available = len(pool)

        if available == 0:
            print(f"  [WARN] Havuzda yok, atlandı: '{label}'")
            summary_rows.append((label, target, 0, "HAVUZDA YOK"))
            continue

        take = min(target, available)
        sampled_parts.append(pool.sample(take, random_state=seed))

        if take < target:
            summary_rows.append((label, target, take, "yetersiz"))
        else:
            summary_rows.append((label, target, take, "OK"))

    # BALANCED_TARGETS'ta olmayan ama havuzda olan kategoriler (istatistik için)
    covered = set(BALANCED_TARGETS.keys())
    uncovered = set(available_counts.keys()) - covered
    if uncovered:
        print(f"  [INFO] {len(uncovered)} kategori BALANCED_TARGETS'ta yok, atlandı: "
              f"{', '.join(sorted(uncovered)[:5])}{'...' if len(uncovered) > 5 else ''}")

    result = pd.concat(sampled_parts, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    # --- Özet tablosu ---
    print(f"\n  {'Kategori':<22} {'Hedef':>8} {'Havuz':>8} {'Alınan':>8}  Durum")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}  {'-'*12}")
    for label, target, taken, status in sorted(summary_rows, key=lambda x: -x[1]):
        pool_size = available_counts.get(label, 0)
        flag = "" if status == "OK" else f"  <- {status}"
        print(f"  {label:<22} {target:>8,} {pool_size:>8,} {taken:>8,}{flag}")
    print(f"  {'TOPLAM':<22} {sum(BALANCED_TARGETS.values()):>8,} "
          f"{'':>8} {len(result):>8,}")

    return result


# ---------------------------------------------------------------------------
# Adım 4 — Extract + JSONL
# ---------------------------------------------------------------------------

def extract_and_write(
    df: pd.DataFrame,
    image_index: dict,
    out_dir: Path,
    jsonl_path: Path,
    split: str,
    clean: bool = False,
    source_mode: str = "zip",
) -> None:
    """
    Her görseli hedefe yaz ve JSONL metadata üret.

    source_mode="zip"   : Lazy ZipFile cache + ExitStack (eski davranış).
    source_mode="folder": os.link (hard link) → OSError'da shutil.copy2 fallback.
                          hardlinked / copied sayaçları ayrı tutulur.
    --clean verilmişse out_dir önce silinip yeniden oluşturulur.
    """
    print(f"\n[4/4] {'Hard-link/kopyalama' if source_mode == 'folder' else 'Extract'} -> {out_dir}")

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"      --clean: '{out_dir}' silindi, yeniden oluşturulacak.")

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    ok = skip = fail = 0

    if source_mode == "folder":
        hardlinked = copied = 0

        with open(jsonl_path, "w", encoding="utf-8") as meta:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="  link/copy", unit="img"):
                fname = row["image_path"]
                src   = image_index.get(fname)
                dst   = out_dir / fname

                try:
                    already_exists = dst.exists()
                except OSError as e:
                    print(f"\n  [WARN] dst.exists() OSError (sürücü erişim sorunu?): {dst} — {e}")
                    already_exists = False

                if already_exists:
                    skip += 1
                else:
                    if src is None:
                        print(f"\n  [WARN] index'te yok: {fname}")
                        fail += 1
                        continue
                    try:
                        os.link(src, dst)
                        hardlinked += 1
                        ok += 1
                    except OSError:
                        try:
                            shutil.copy2(src, dst)
                            copied += 1
                            ok += 1
                        except Exception as e:
                            print(f"\n  [WARN] Kopyalama hatası: {fname} — {e}")
                            fail += 1
                            continue

                meta.write(json.dumps(
                    {
                        "path":      str(dst),
                        "caption":   row["caption"].strip(),
                        "pathology": row["primary_pathology"],
                        "split":     split,
                    },
                    ensure_ascii=False,
                ) + "\n")

        print(f"      Hard link: {hardlinked:,}  Kopya: {copied:,}  "
              f"Skip(zaten_var): {skip:,}  Fail: {fail:,}")

    else:
        # ZIP modu — orijinal lazy cache + ExitStack mantığı
        zip_cache: dict[str, dict] = {}

        with contextlib.ExitStack() as stack, \
             open(jsonl_path, "w", encoding="utf-8") as meta:

            for _, row in tqdm(df.iterrows(), total=len(df), desc="  extract", unit="img"):
                fname    = row["image_path"]
                zip_path = image_index.get(fname)
                dst      = out_dir / fname

                if dst.exists():
                    skip += 1
                else:
                    if zip_path is None:
                        print(f"\n  [WARN] index'te yok: {fname}")
                        fail += 1
                        continue

                    if zip_path not in zip_cache:
                        handle = stack.enter_context(zipfile.ZipFile(zip_path, "r"))
                        members = {
                            info.filename.split("/", 1)[-1]: info
                            for info in handle.infolist()
                            if "/" in info.filename and not info.filename.endswith("/")
                        }
                        zip_cache[zip_path] = {"handle": handle, "members": members}

                    entry  = zip_cache[zip_path]
                    member = entry["members"].get(fname)

                    if member is None:
                        print(f"\n  [WARN] ZIP içinde bulunamadı: {fname} @ {Path(zip_path).name}")
                        fail += 1
                        continue

                    try:
                        data = entry["handle"].read(member)
                        dst.write_bytes(data)
                        ok += 1
                    except Exception as e:
                        print(f"\n  [WARN] Okuma hatası: {fname} @ {Path(zip_path).name} — {e}")
                        fail += 1
                        continue

                meta.write(json.dumps(
                    {
                        "path":      str(dst),
                        "caption":   row["caption"].strip(),
                        "pathology": row["primary_pathology"],
                        "split":     split,
                    },
                    ensure_ascii=False,
                ) + "\n")

        print(f"      OK={ok:,}  Skip(zaten_var)={skip:,}  Fail={fail:,}")


# ---------------------------------------------------------------------------
# Adım 5 — Stats JSON + JSONL doğrulama
# ---------------------------------------------------------------------------

def write_stats(df: pd.DataFrame, stats_path: Path, n: int, split: str) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    dist = df["primary_pathology"].value_counts().to_dict()
    payload = {
        "subset_size":           len(df),
        "requested_n":           n,
        "split":                 split,
        "n_categories":          len(dist),
        "category_distribution": dist,
    }
    stats_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"      Stats  -> {stats_path}")


def validate_jsonl(jsonl_path: Path) -> int:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())
    print(f"      JSONL doğrulandı: {count:,} satır — {jsonl_path}")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Quilt-1M subset hazırla")
    p.add_argument("--csv",            default=DEFAULT_CSV,     help="quilt_1M_lookup.csv yolu")
    p.add_argument("--zip_dir",        default=DEFAULT_ZIP_DIR, help="images_part_*.zip klasörü (ZIP modu)")
    p.add_argument("--img_source_dir", default=None,
                   help="Extract edilmiş görsel klasörü/klasörleri — virgülle ayrılmış birden fazla "
                        "kabul edilir (klasör modu; verilirse ZIP'e göre öncelikli). "
                        r'Örnek: "C:\part1,E:\part2,E:\part3"')
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Görsel çıktı klasörü")
    p.add_argument("--jsonl",   default=DEFAULT_JSONL,   help="Çıktı JSONL yolu")
    p.add_argument("--stats",   default=DEFAULT_STATS,   help="İstatistik JSON yolu")
    p.add_argument("--n",    type=int, default=DEFAULT_N,
                   help="Subset boyutu (sadece --sampling_mode stratified için kullanılır)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    p.add_argument("--split",   default=DEFAULT_SPLIT,
                   choices=["train", "val", "test"], help="Kullanılacak split")
    p.add_argument("--sampling_mode", default="stratified",
                   choices=["stratified", "balanced"],
                   help="stratified: orantılı örnekleme (--n gerekli); "
                        "balanced: BALANCED_TARGETS dict ile dengeli örnekleme (--n ignore edilir)")
    p.add_argument("--clean", action="store_true",
                   help="Extract öncesi out_dir'i sil ve yeniden oluştur")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir    = Path(args.out_dir)
    jsonl_path = Path(args.jsonl)
    stats_path = Path(args.stats)

    if args.img_source_dir:
        dirs = [d.strip() for d in args.img_source_dir.split(",") if d.strip()]
        image_index = build_folder_index(dirs)
        source_mode = "folder"
    else:
        image_index = build_zip_index(args.zip_dir)
        source_mode = "zip"

    candidates = collect_candidates(args.csv, image_index, split=args.split)

    if len(candidates) == 0:
        print("[ERROR] Filtre sonrası hiç aday kalmadı.")
        print("        CSV yolu, görsel kaynağı ve --split değerini kontrol et.")
        return

    if args.sampling_mode == "balanced":
        subset = balanced_sample(candidates, seed=args.seed)
    else:
        subset = stratified_sample(candidates, n=args.n, seed=args.seed)

    extract_and_write(
        subset, image_index, out_dir, jsonl_path,
        split=args.split, clean=args.clean, source_mode=source_mode,
    )
    write_stats(subset, stats_path, n=len(subset), split=args.split)
    validate_jsonl(jsonl_path)

    print("\n=== ÖZET ===")
    print(f"  Görsel klasörü  : {out_dir}")
    print(f"  Meta JSONL      : {jsonl_path}")
    print(f"  Stats JSON      : {stats_path}")
    print(f"  Sampling modu   : {args.sampling_mode}")
    print(f"  Toplam işlenen  : {len(subset):,} görsel")
    print(f"\n  Kategori dağılımı (top-10):")
    print(subset["primary_pathology"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
