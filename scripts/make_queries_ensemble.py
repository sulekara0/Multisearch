"""
make_queries_ensemble.py — Multi-query ensemble set (V3).

Each concept has 3 query variants:
  [0] General      — broad, matches varied caption styles
  [1] Medium       — standard pathology report language
  [2] High         — detailed histological terminology (like V2)

RRF fuses all three result lists: score(d) = sum(1 / (60 + rank_i))

Usage:
    python scripts/make_queries_ensemble.py
    python scripts/make_queries_ensemble.py --out artifacts/eval_queries_ensemble.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Format: (general, medium, high_specificity), expected_categories
CONCEPTS = [
    # ------------------------------------------------------------------ Dermatopathology (7)
    (("skin cancer biopsy",
      "basal cell carcinoma skin",
      "basal cell carcinoma nodular pattern H&E"),
     ["Dermatopathology"]),

    (("skin melanoma pigmented",
      "melanoma histology",
      "malignant melanoma Clark level dermis invasion"),
     ["Dermatopathology"]),

    (("skin squamous cancer",
      "squamous cell carcinoma skin",
      "cutaneous squamous cell carcinoma keratinization"),
     ["Dermatopathology"]),

    (("skin inflammation biopsy",
      "psoriasis skin biopsy",
      "psoriasis acanthosis parakeratosis Munro microabscesses"),
     ["Dermatopathology"]),

    (("skin fibrous tumor dermis",
      "dermatofibrosarcoma DFSP",
      "dermatofibrosarcoma protuberans storiform pattern"),
     ["Dermatopathology"]),

    (("skin autoimmune blister",
      "pemphigus vulgaris",
      "pemphigus vulgaris acantholysis suprabasal split"),
     ["Dermatopathology"]),

    (("skin lymphoma infiltrate",
      "cutaneous T-cell lymphoma",
      "mycosis fungoides Pautrier microabscesses epidermotropism"),
     ["Dermatopathology"]),

    # ------------------------------------------------------------------ Gastrointestinal (5)
    (("colon cancer histology",
      "colon adenocarcinoma",
      "colorectal adenocarcinoma glandular pattern H&E"),
     ["Gastrointestinal"]),

    (("stomach cancer microscopy",
      "gastric cancer histology",
      "gastric adenocarcinoma intestinal type Lauren classification"),
     ["Gastrointestinal"]),

    (("colon polyp biopsy",
      "colorectal polyp adenoma",
      "tubular adenoma colonic mucosa low grade dysplasia"),
     ["Gastrointestinal"]),

    (("bowel inflammation biopsy",
      "Crohn disease intestinal biopsy",
      "Crohn disease transmural granulomatous inflammation"),
     ["Gastrointestinal"]),

    (("pancreas cancer",
      "pancreatic ductal adenocarcinoma",
      "pancreatic adenocarcinoma desmoplastic stroma glands"),
     ["Gastrointestinal"]),

    # ------------------------------------------------------------------ Pulmonary (5)
    (("lung cancer",
      "lung adenocarcinoma",
      "pulmonary adenocarcinoma acinar lepidic pattern H&E"),
     ["Pulmonary"]),

    (("lung squamous cancer",
      "lung squamous cell carcinoma",
      "squamous cell carcinoma lung keratinization histopathology"),
     ["Pulmonary"]),

    (("lung small cell cancer",
      "small cell lung cancer",
      "small cell lung carcinoma neuroendocrine"),
     ["Pulmonary"]),

    (("lung disease fibrosis",
      "pulmonary fibrosis histology",
      "usual interstitial pneumonia UIP fibrosis histology"),
     ["Pulmonary"]),

    (("pleura tumor",
      "mesothelioma pleural",
      "pleural mesothelioma epithelioid histopathology"),
     ["Pulmonary"]),

    # ------------------------------------------------------------------ Gynecologic (5)
    (("uterus cancer",
      "endometrial carcinoma",
      "endometrial adenocarcinoma uterus H&E histopathology"),
     ["Gynecologic"]),

    (("cervix cancer",
      "cervical squamous cell carcinoma",
      "cervical intraepithelial neoplasia CIN histology"),
     ["Gynecologic"]),

    (("ovary cancer",
      "ovarian serous carcinoma",
      "high grade serous carcinoma ovary histopathology"),
     ["Gynecologic"]),

    (("uterus fibroid myoma",
      "uterine leiomyoma fibroid",
      "uterine smooth muscle leiomyoma myometrium"),
     ["Gynecologic"]),

    (("placenta pregnancy tissue",
      "placental histology",
      "placental chorionic villi trophoblast histopathology"),
     ["Gynecologic"]),

    # ------------------------------------------------------------------ Soft tissue (4)
    (("soft tissue tumor",
      "soft tissue sarcoma",
      "soft tissue sarcoma H&E histopathology"),
     ["Soft tissue"]),

    (("smooth muscle tumor",
      "leiomyosarcoma",
      "leiomyosarcoma smooth muscle tumor spindle cells"),
     ["Soft tissue"]),

    (("fat tissue tumor",
      "liposarcoma adipose",
      "liposarcoma adipose tissue lipoblasts"),
     ["Soft tissue"]),

    (("muscle tumor children",
      "rhabdomyosarcoma",
      "rhabdomyosarcoma skeletal muscle rhabdomyoblasts"),
     ["Soft tissue"]),

    # ------------------------------------------------------------------ Genitourinary (4)
    (("prostate cancer",
      "prostate adenocarcinoma Gleason",
      "prostate adenocarcinoma acinar Gleason score histopathology"),
     ["Genitourinary"]),

    (("bladder cancer",
      "bladder urothelial carcinoma",
      "bladder urothelial transitional cell carcinoma high grade"),
     ["Genitourinary"]),

    (("testis cancer germ cell",
      "testicular seminoma",
      "testicular seminoma germ cell fibrous septa"),
     ["Genitourinary"]),

    (("urinary tract cancer",
      "urothelial carcinoma",
      "urothelial carcinoma papillary fronds"),
     ["Genitourinary"]),

    # ------------------------------------------------------------------ Hematopathology (4)
    (("lymph node lymphoma",
      "follicular lymphoma lymph node",
      "follicular lymphoma lymph node germinal centers"),
     ["Hematopathology"]),

    (("Hodgkin disease lymphoma",
      "Hodgkin lymphoma Reed-Sternberg",
      "Hodgkin lymphoma Reed-Sternberg cells mixed cellularity"),
     ["Hematopathology"]),

    (("bone marrow biopsy",
      "bone marrow biopsy hematopoietic",
      "bone marrow biopsy cellularity hematopoietic cells"),
     ["Hematopathology"]),

    (("plasma cell tumor myeloma",
      "multiple myeloma bone marrow",
      "multiple myeloma plasma cell bone marrow"),
     ["Hematopathology"]),

    # ------------------------------------------------------------------ Renal (3)
    (("kidney cancer",
      "renal cell carcinoma",
      "clear cell renal cell carcinoma"),
     ["Renal"]),

    (("kidney inflammation",
      "glomerulonephritis kidney biopsy",
      "glomerulonephritis mesangial immune deposits kidney"),
     ["Renal"]),

    (("kidney damage necrosis",
      "kidney tubular necrosis",
      "acute tubular necrosis kidney cortex"),
     ["Renal"]),

    # ------------------------------------------------------------------ Bone (3)
    (("bone cancer",
      "osteosarcoma bone",
      "osteosarcoma bone tumor osteoid matrix"),
     ["Bone"]),

    (("bone cartilage tumor",
      "chondrosarcoma cartilage",
      "chondrosarcoma hyaline cartilage lobular"),
     ["Bone"]),

    (("bone giant cell tumor",
      "giant cell tumor bone",
      "giant cell tumor of bone osteoclast"),
     ["Bone"]),

    # ------------------------------------------------------------------ Neuropathology (3)
    (("brain tumor",
      "glioblastoma brain",
      "glioblastoma pseudopalisading necrosis microvascular proliferation"),
     ["Neuropathology"]),

    (("brain meninges tumor",
      "meningioma histology",
      "meningioma whorls psammoma bodies histopathology"),
     ["Neuropathology"]),

    (("brain glioma",
      "astrocytoma glioma",
      "diffuse astrocytoma IDH mutant glioma"),
     ["Neuropathology"]),

    # ------------------------------------------------------------------ Breast (3)
    (("breast cancer",
      "invasive ductal carcinoma breast",
      "invasive ductal carcinoma breast histopathology"),
     ["Breast"]),

    (("breast in situ cancer",
      "ductal carcinoma in situ DCIS",
      "ductal carcinoma in situ DCIS comedo necrosis"),
     ["Breast"]),

    (("breast lobular cancer",
      "lobular carcinoma breast",
      "invasive lobular carcinoma breast single file"),
     ["Breast"]),

    # ------------------------------------------------------------------ Endocrine (2)
    (("thyroid cancer",
      "papillary thyroid carcinoma",
      "papillary thyroid carcinoma nuclear features ground glass"),
     ["Endocrine"]),

    (("adrenal tumor",
      "adrenal pheochromocytoma",
      "adrenal pheochromocytoma chromaffin cells zellballen"),
     ["Endocrine"]),

    # ------------------------------------------------------------------ Head and Neck (2)
    (("oral cancer",
      "squamous cell carcinoma head neck",
      "oral cavity squamous cell carcinoma tongue histology"),
     ["Head and Neck"]),

    (("salivary gland tumor",
      "pleomorphic adenoma salivary",
      "pleomorphic adenoma parotid salivary gland myoepithelial"),
     ["Head and Neck"]),

    # ------------------------------------------------------------------ Cardiac (2)
    (("heart attack necrosis",
      "myocardial infarction",
      "myocardial infarction coagulation necrosis cardiomyocyte"),
     ["Cardiac"]),

    (("heart muscle disease",
      "cardiomyopathy histology",
      "hypertrophic cardiomyopathy myocyte disarray"),
     ["Cardiac"]),

    # ------------------------------------------------------------------ Hepatopathology (2)
    (("liver cancer",
      "hepatocellular carcinoma",
      "hepatocellular carcinoma trabecular pattern"),
     ["Hepatopathology"]),

    (("liver cirrhosis fibrosis",
      "liver fibrosis histology",
      "liver cirrhosis bridging fibrosis portal"),
     ["Hepatopathology"]),

    # ------------------------------------------------------------------ Ophthalmic (1)
    (("eye tumor",
      "retinoblastoma",
      "retinoblastoma Flexner-Wintersteiner rosettes"),
     ["Ophthalmic"]),

    # ------------------------------------------------------------------ Cytopathology (1)
    (("cancer cells smear",
      "fine needle aspiration cytology",
      "fine needle aspiration cytology smear"),
     ["Cytopathology"]),
]


def main():
    parser = argparse.ArgumentParser(description="Write eval_queries_ensemble.json")
    parser.add_argument("--out", default="artifacts/eval_queries_ensemble.json")
    args = parser.parse_args()

    entries = []
    for i, (queries, expected) in enumerate(CONCEPTS):
        entries.append({
            "id":       f"q{i+1:03d}",
            "queries":  list(queries),
            "expected": expected,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cats: dict[str, int] = {}
    for e in entries:
        for c in e["expected"]:
            cats[c] = cats.get(c, 0) + 1

    v2_counts = {
        "Dermatopathology": 10, "Gastrointestinal": 8, "Pulmonary": 8,
        "Gynecologic": 7, "Soft tissue": 6, "Genitourinary": 6,
        "Hematopathology": 6, "Renal": 5, "Bone": 5, "Neuropathology": 5,
        "Breast": 5, "Endocrine": 4, "Head and Neck": 4, "Cardiac": 4,
        "Hepatopathology": 4, "Ophthalmic": 1, "Cytopathology": 2,
    }

    print(f"[OK] {len(entries)} concepts x 3 variants = "
          f"{len(entries)*3} FAISS searches per encoder -> {out_path}")
    print(f"\nCategory breakdown (concepts, each 3 queries):")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        v2 = v2_counts.get(cat, 0)
        print(f"  {cat:<35} {cnt} concepts  (V2 had {v2} single queries)")


if __name__ == "__main__":
    main()
