# scripts/00_fetch_from_csv.py  (v3 - solid)
import csv, os, re, json, hashlib, sys
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

CSV_PATH = "data/train.csv"                         # CSV burada
OUT_DIR = Path("data/images")                       # Görseller buraya inecek
META_OUT = Path("artifacts/captions_coco.jsonl")    # path+caption logu
MAX_NUM = 50000

OUT_DIR.mkdir(parents=True, exist_ok=True)
META_OUT.parent.mkdir(parents=True, exist_ok=True)

def safe_name(url:str)->str:
    m = re.search(r'/(\d+)\.(jpg|jpeg|png|webp)$', url.lower())
    if m: return f"coco_{m.group(1)}.jpg"
    h = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"img_{h}.jpg"

def download_and_save(url, cap, sess):
    url = url.strip()
    name = safe_name(url)
    out_path = OUT_DIR / name
    if out_path.exists():
        return out_path, "skip_exists"
    resp = try_fetch(sess, url)
    im = Image.open(BytesIO(resp.content)).convert("RGB")
    im.save(out_path, "JPEG", quality=90, optimize=True)
    return out_path, "ok"

def try_fetch(sess, url: str):
    def _get(u, verify=True):
        resp = sess.get(u, timeout=20, verify=verify, allow_redirects=True)
        resp.raise_for_status()
        return resp
    # 1) olduğu gibi
    try:
        return _get(url)
    except Exception:
        pass
    # 2) http/https değiştir
    if url.startswith("http://"):
        alt = "https://" + url[len("http://"):]
    elif url.startswith("https://"):
        alt = "http://" + url[len("https://"):]
    else:
        alt = "http://" + url
    try:
        return _get(alt)
    except Exception:
        pass
    # 3) HTTPS verify kapalı fallback
    if not alt.startswith("https://"):
        alt2 = "https://" + alt.split("://",1)[1]
    else:
        alt2 = alt
    return _get(alt2, verify=False)

def main():
    if not Path(CSV_PATH).exists():
        print(f"❌ CSV yok: {CSV_PATH}")
        sys.exit(1)

    # 1) Önce UTF-8-SIG + excel (virgül) dene
    try:
        f = open(CSV_PATH, 'r', encoding='utf-8-sig', newline='')
        reader = csv.DictReader(f)  # excel dialect -> delimiter=','
        header = [h.strip().lower() for h in reader.fieldnames]
        delim_used = ','
    except Exception as e:
        print(f"⚠️ excel/utf-8-sig ile okuyamadı: {e}")
        # 2) Alternatif: noktalı virgül
        f = open(CSV_PATH, 'r', encoding='utf-8-sig', newline='')
        reader = csv.DictReader(f, delimiter=';')
        header = [h.strip().lower() for h in reader.fieldnames]
        delim_used = ';'

    # Sütunlar
    url_col = next((c for c in ["url","image_url","img_url","href","link"] if c in header), None)
    cap_col = next((c for c in ["caption","text","alt","description","desc"] if c in header), None)
    if not url_col:
        print(f"❌ URL sütunu bulunamadı. Header: {header}")
        sys.exit(1)
    print(f"📄 Header: {header} | 🔹 Delimiter: {repr(delim_used)} | url_col: {url_col} | cap_col: {cap_col or '(yok)'}")

    rows = list(reader)
    total = len(rows) if MAX_NUM is None else min(MAX_NUM, len(rows))
    print(f"🧮 Toplam satır: {len(rows)} | İşlenecek: {total}")

    sess = requests.Session()
    sess.headers["User-Agent"] = "multisearch-downloader/1.0"

    ok = skip = fail = 0
    with open(META_OUT, "w", encoding="utf-8") as meta:
        for i, row in enumerate(rows[:total], 1):
            url = (row.get(url_col) or "").strip()
            cap = (row.get(cap_col) or "").strip()
            if not url:
                fail += 1
                continue
            try:
                path, status = download_and_save(url, cap, sess)
                if status == "ok":
                    ok += 1
                    if ok % 25 == 0:
                        print(f"  ✓ {ok} indirildi...")
                else:
                    skip += 1
                meta.write(json.dumps({"path": str(path), "caption": cap}) + "\n")
            except Exception as e:
                fail += 1
                if fail < 10 or fail % 50 == 0:
                    print(f"  ⚠️ {i}: {url} -> {e}")

    f.close()
    print(f"✅ Bitti | OK: {ok}  Skip: {skip}  Fail: {fail}  -> {OUT_DIR}")

if __name__ == "__main__":
    main()
