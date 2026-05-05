"""
00_prepare_quilt.py — Quilt-1M subset hazırlama

Adımlar:
  1. ZIP dizinindeki tüm ZIP'leri indexle  -> {filename: zip_path} dict
  2. CSV'yi chunk'larla tara, filtrele, ZIP'te olan + split eşleşenleri tut
  3. primary_pathology'ye göre stratified N sampling (seed=42)
  4. Görselleri doğru ZIP'ten extract et (lazy handle cache + ExitStack)
  5. artifacts/captions_quilt.jsonl  ve  artifacts/subset_stats.json yaz

Kullanım:
  python scripts/00_prepare_quilt.py --zip_dir data/zips --n 100
  python scripts/00_prepare_quilt.py --zip_dir data/zips
  python scripts/00_prepare_quilt.py \\
      --csv "C:\\data\\quilt_1M_lookup.csv" \\
      --zip_dir "C:\\data\\quilt_zips" \\
      --split train --n 10000 --clean
"""

import argparse
import ast
import contextlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults (proje-relative; CLI ile override edilebilir)
# ---------------------------------------------------------------------------
DEFAULT_CSV     = "data/quilt_1M_lookup.csv"
DEFAULT_ZIP_DIR = "data/zips"
DEFAULT_OUT_DIR = "data/images"
DEFAULT_JSONL   = "artifacts/captions_quilt.jsonl"
DEFAULT_STATS   = "artifacts/subset_stats.json"
DEFAULT_N       = 10_000
DEFAULT_SEED    = 42
DEFAULT_SPLIT   = "train"
CHUNK_SIZE      = 50_000
NEEDED_COLS     = ["image_path", "caption", "pathology", "not_histology", "split"]


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
# Adım 1 — ZIP index
# ---------------------------------------------------------------------------

def build_zip_index(zip_dir: str) -> dict:
    """
    zip_dir içindeki tüm images_part_*.zip dosyalarını tara.
    Dönüş: {dosya_adı: zip_yolu_str}
    Her ZIP'teki dosya sayısını ve toplamı yazdırır.
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


# ---------------------------------------------------------------------------
# Adım 2 — CSV tarama
# ---------------------------------------------------------------------------

def collect_candidates(csv_path: str, zip_index: dict, split: str) -> pd.DataFrame:
    """
    CSV'yi chunk'larla tara:
      - not_histology == 0
      - caption dolu
      - split eşleşme
      - image_path ZIP'te mevcut
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
# Adım 3 — Stratified sampling
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

    # Quota yuvarlamadan kaynaklanan fazla/eksik düzeltme
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
# Adım 4 — Extract + JSONL
# ---------------------------------------------------------------------------

def extract_and_write(
    df: pd.DataFrame,
    zip_index: dict,
    out_dir: Path,
    jsonl_path: Path,
    split: str,
    clean: bool = False,
) -> None:
    """
    Her görseli doğru ZIP'ten çek (lazy cache), JSONL'a yaz.
    contextlib.ExitStack ile ZIP handle'ları exception'da da temiz kapanır.
    --clean verilmişse out_dir önce silinip yeniden oluşturulur.
    """
    print(f"\n[4/4] Extract -> {out_dir}")

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"      --clean: '{out_dir}' silindi, yeniden oluşturulacak.")

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # zip_yolu -> {"handle": ZipFile, "members": {dosya_adı: ZipInfo}}
    zip_cache: dict[str, dict] = {}

    ok = skip = fail = 0

    with contextlib.ExitStack() as stack, \
         open(jsonl_path, "w", encoding="utf-8") as meta:

        for _, row in tqdm(df.iterrows(), total=len(df), desc="  extract", unit="img"):
            fname    = row["image_path"]
            zip_path = zip_index.get(fname)
            dst      = out_dir / fname

            if dst.exists():
                skip += 1
            else:
                if zip_path is None:
                    print(f"\n  [WARN] index'te yok: {fname}")
                    fail += 1
                    continue

                # Lazy ZIP açma — ilk kez görülen ZIP'i açıp cache'e al
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

    print(f"      OK={ok}  Skip(zaten_var)={skip}  Fail={fail}")


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
    """JSONL dosyasını açıp satır sayısını doğrular."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())
    print(f"      JSONL doğrulandı: {count} satır — {jsonl_path}")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Quilt-1M subset hazırla")
    p.add_argument("--csv",     default=DEFAULT_CSV,     help="quilt_1M_lookup.csv yolu")
    p.add_argument("--zip_dir", default=DEFAULT_ZIP_DIR, help="images_part_*.zip klasörü")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Görsel çıktı klasörü")
    p.add_argument("--jsonl",   default=DEFAULT_JSONL,   help="Çıktı JSONL yolu")
    p.add_argument("--stats",   default=DEFAULT_STATS,   help="İstatistik JSON yolu")
    p.add_argument("--n",    type=int, default=DEFAULT_N,    help="Subset boyutu")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    p.add_argument("--split",   default=DEFAULT_SPLIT,
                   choices=["train", "val", "test"], help="Kullanılacak split")
    p.add_argument("--clean", action="store_true",
                   help="Extract öncesi out_dir'i sil ve yeniden oluştur")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir    = Path(args.out_dir)
    jsonl_path = Path(args.jsonl)
    stats_path = Path(args.stats)

    zip_index  = build_zip_index(args.zip_dir)
    candidates = collect_candidates(args.csv, zip_index, split=args.split)

    if len(candidates) == 0:
        print("[ERROR] Filtre sonrası hiç aday kalmadı.")
        print("        CSV yolu, ZIP dizini ve --split değerini kontrol et.")
        return

    subset = stratified_sample(candidates, n=args.n, seed=args.seed)
    extract_and_write(
        subset, zip_index, out_dir, jsonl_path,
        split=args.split, clean=args.clean,
    )
    write_stats(subset, stats_path, n=args.n, split=args.split)
    validate_jsonl(jsonl_path)

    print("\n=== ÖZET ===")
    print(f"  Görsel klasörü : {out_dir}")
    print(f"  Meta JSONL     : {jsonl_path}")
    print(f"  Stats JSON     : {stats_path}")
    print(f"\n  Kategori dağılımı (top-10):")
    print(subset["primary_pathology"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
