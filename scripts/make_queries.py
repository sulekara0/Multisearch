import json, random
from pathlib import Path

CAP_PATH = Path("artifacts/captions_coco.jsonl")
OUT_PATH = Path("artifacts/queries.json")

N = 200          # 200 sorgu iyi bir benchmark
SEED = 42

rows = []
with open(CAP_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        # faiss_search.py ids.json genelde "coco_123.jpg" veya göreli yol tutuyor.
        # obj["path"] örn: "data/images/coco_123.jpg" olabilir.
        # Biz sadece dosya adını alıyoruz (ids.json ile aynıysa çalışır)
        expected = Path(obj["path"]).name
        cap = (obj.get("caption") or "").strip()
        if cap:
            rows.append({"q": cap, "expected_id": expected})

random.seed(SEED)
random.shuffle(rows)
rows = rows[:N]

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print("Wrote", len(rows), "queries to", OUT_PATH)
