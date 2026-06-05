"""
make_queries_v2.py — Improved eval query set (V2).

Changes vs V1:
- Weak categories fully rewritten with precise pathology terminology
- Category coverage expanded (2-4 new queries in most weak categories)
- Total labelled queries: 90 (V1: 63)

Usage:
    python scripts/make_queries_v2.py
    python scripts/make_queries_v2.py --out artifacts/eval_queries_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QUERIES = [
    # --- Dermatopathology (10) — unchanged, already P@10=0.900 ---
    {"query": "basal cell carcinoma skin biopsy",                   "expected": ["Dermatopathology"]},
    {"query": "melanoma histology",                                  "expected": ["Dermatopathology"]},
    {"query": "squamous cell carcinoma skin",                        "expected": ["Dermatopathology"]},
    {"query": "psoriasis skin biopsy",                               "expected": ["Dermatopathology"]},
    {"query": "dermatofibrosarcoma protuberans DFSP",                "expected": ["Dermatopathology"]},
    {"query": "sebaceous carcinoma",                                 "expected": ["Dermatopathology"]},
    {"query": "actinic keratosis",                                   "expected": ["Dermatopathology"]},
    {"query": "pemphigus vulgaris",                                  "expected": ["Dermatopathology"]},
    {"query": "cutaneous T-cell lymphoma mycosis fungoides",         "expected": ["Dermatopathology"]},
    {"query": "lichen planus skin biopsy",                           "expected": ["Dermatopathology"]},

    # --- Gastrointestinal (8) — +1 new ---
    {"query": "colon adenocarcinoma",                                "expected": ["Gastrointestinal"]},
    {"query": "gastric cancer histology",                            "expected": ["Gastrointestinal"]},
    {"query": "colorectal polyp adenoma",                            "expected": ["Gastrointestinal"]},
    {"query": "Crohn disease intestinal biopsy",                     "expected": ["Gastrointestinal"]},
    {"query": "dysplasia in colonic mucosa",                         "expected": ["Gastrointestinal"]},
    {"query": "esophageal squamous cell carcinoma",                  "expected": ["Gastrointestinal"]},
    {"query": "pancreatic ductal adenocarcinoma",                    "expected": ["Gastrointestinal"]},
    {"query": "colorectal mucinous adenocarcinoma signet ring cell", "expected": ["Gastrointestinal"]},

    # --- Pulmonary (8) — all rewritten + 2 new ---
    {"query": "pulmonary adenocarcinoma acinar lepidic pattern H&E",   "expected": ["Pulmonary"]},
    {"query": "lung squamous cell carcinoma keratinization histopathology", "expected": ["Pulmonary"]},
    {"query": "small cell lung carcinoma neuroendocrine",              "expected": ["Pulmonary"]},
    {"query": "usual interstitial pneumonia UIP fibrosis histology",   "expected": ["Pulmonary"]},
    {"query": "pleural mesothelioma epithelioid histopathology",       "expected": ["Pulmonary"]},
    {"query": "pulmonary carcinoid neuroendocrine tumor",              "expected": ["Pulmonary"]},
    {"query": "non-small cell lung cancer NSCLC microscopy",           "expected": ["Pulmonary"]},
    {"query": "lung large cell carcinoma pleomorphic",                 "expected": ["Pulmonary"]},

    # --- Gynecologic (7) — all rewritten + 2 new ---
    {"query": "endometrial adenocarcinoma uterus H&E histopathology",   "expected": ["Gynecologic"]},
    {"query": "cervical intraepithelial neoplasia CIN histology",        "expected": ["Gynecologic"]},
    {"query": "high grade serous carcinoma ovary histopathology",        "expected": ["Gynecologic"]},
    {"query": "uterine smooth muscle leiomyoma myometrium",              "expected": ["Gynecologic"]},
    {"query": "vulvar squamous intraepithelial lesion VIN",              "expected": ["Gynecologic"]},
    {"query": "endometriosis ectopic endometrial glands stroma",         "expected": ["Gynecologic"]},
    {"query": "placental chorionic villi trophoblast histopathology",    "expected": ["Gynecologic"]},

    # --- Soft tissue (6) — +1 new ---
    {"query": "soft tissue sarcoma H&E histopathology",                 "expected": ["Soft tissue"]},
    {"query": "leiomyosarcoma smooth muscle tumor",                      "expected": ["Soft tissue"]},
    {"query": "liposarcoma adipose tissue",                              "expected": ["Soft tissue"]},
    {"query": "rhabdomyosarcoma skeletal muscle",                        "expected": ["Soft tissue"]},
    {"query": "synovial sarcoma biphasic histology",                     "expected": ["Soft tissue"]},
    {"query": "undifferentiated pleomorphic sarcoma spindle cells",      "expected": ["Soft tissue"]},

    # --- Genitourinary (6) — +2 new ---
    {"query": "prostate adenocarcinoma Gleason score histopathology",    "expected": ["Genitourinary"]},
    {"query": "bladder urothelial transitional cell carcinoma high grade","expected": ["Genitourinary"]},
    {"query": "testicular seminoma germ cell",                           "expected": ["Genitourinary"]},
    {"query": "urothelial carcinoma papillary",                          "expected": ["Genitourinary"]},
    {"query": "penile squamous cell carcinoma",                          "expected": ["Genitourinary"]},
    {"query": "renal pelvis urothelial carcinoma",                       "expected": ["Genitourinary"]},

    # --- Hematopathology (6) — all rewritten + 2 new ---
    {"query": "follicular lymphoma lymph node germinal centers",         "expected": ["Hematopathology"]},
    {"query": "Hodgkin lymphoma Reed-Sternberg cells mixed cellularity", "expected": ["Hematopathology"]},
    {"query": "bone marrow biopsy cellularity hematopoietic cells",     "expected": ["Hematopathology"]},
    {"query": "non-Hodgkin lymphoma lymph node architecture",           "expected": ["Hematopathology"]},
    {"query": "mantle cell lymphoma cyclin D1 histology",               "expected": ["Hematopathology"]},
    {"query": "multiple myeloma plasma cell bone marrow",               "expected": ["Hematopathology"]},

    # --- Renal (5) — +2 new ---
    {"query": "clear cell renal cell carcinoma",                         "expected": ["Renal"]},
    {"query": "glomerulonephritis kidney biopsy",                        "expected": ["Renal"]},
    {"query": "kidney tubular necrosis acute",                           "expected": ["Renal"]},
    {"query": "papillary renal cell carcinoma type 1",                   "expected": ["Renal"]},
    {"query": "IgA nephropathy mesangial immune deposits",               "expected": ["Renal"]},

    # --- Bone (5) — +2 new ---
    {"query": "osteosarcoma bone tumor osteoid matrix",                  "expected": ["Bone"]},
    {"query": "chondrosarcoma hyaline cartilage",                        "expected": ["Bone"]},
    {"query": "giant cell tumor of bone osteoclast",                     "expected": ["Bone"]},
    {"query": "Ewing sarcoma small round blue cells",                    "expected": ["Bone"]},
    {"query": "fibrous dysplasia bone Chinese letters",                  "expected": ["Bone"]},

    # --- Neuropathology (5) — all rewritten + 2 new ---
    {"query": "glioblastoma pseudopalisading necrosis microvascular proliferation", "expected": ["Neuropathology"]},
    {"query": "meningioma whorls psammoma bodies histopathology",        "expected": ["Neuropathology"]},
    {"query": "diffuse astrocytoma IDH mutant glioma",                   "expected": ["Neuropathology"]},
    {"query": "medulloblastoma cerebellum small round cells",            "expected": ["Neuropathology"]},
    {"query": "ependymoma perivascular pseudorosettes",                  "expected": ["Neuropathology"]},

    # --- Breast (5) — +2 new ---
    {"query": "invasive ductal carcinoma breast histopathology",         "expected": ["Breast"]},
    {"query": "ductal carcinoma in situ DCIS comedo",                    "expected": ["Breast"]},
    {"query": "invasive lobular carcinoma breast",                       "expected": ["Breast"]},
    {"query": "phyllodes tumor breast fibroepithelial",                  "expected": ["Breast"]},
    {"query": "triple negative breast cancer basal-like",                "expected": ["Breast"]},

    # --- Endocrine (4) — +2 new ---
    {"query": "papillary thyroid carcinoma nuclear features",            "expected": ["Endocrine"]},
    {"query": "adrenal pheochromocytoma",                                "expected": ["Endocrine"]},
    {"query": "papillary thyroid microcarcinoma",                        "expected": ["Endocrine"]},
    {"query": "parathyroid adenoma chief cells",                         "expected": ["Endocrine"]},

    # --- Head and Neck (4) — all rewritten + 2 new ---
    {"query": "oral cavity squamous cell carcinoma tongue histology",    "expected": ["Head and Neck"]},
    {"query": "nasopharyngeal carcinoma lymphoepithelioma",              "expected": ["Head and Neck"]},
    {"query": "pleomorphic adenoma parotid salivary gland",              "expected": ["Head and Neck"]},
    {"query": "laryngeal squamous cell carcinoma",                       "expected": ["Head and Neck"]},

    # --- Cardiac (4) — rewritten + 2 new ---
    {"query": "myocardial infarction coagulation necrosis cardiomyocyte","expected": ["Cardiac"]},
    {"query": "cardiac amyloidosis apple-green birefringence",           "expected": ["Cardiac"]},
    {"query": "hypertrophic cardiomyopathy myocyte disarray",            "expected": ["Cardiac"]},
    {"query": "cardiac myxoma stellate cells myxoid stroma",             "expected": ["Cardiac"]},

    # --- Hepatopathology (4) — rewritten + 2 new ---
    {"query": "hepatocellular carcinoma trabecular pattern",             "expected": ["Hepatopathology"]},
    {"query": "liver cirrhosis bridging fibrosis portal",                "expected": ["Hepatopathology"]},
    {"query": "hepatic steatosis fatty liver lipid droplets",            "expected": ["Hepatopathology"]},
    {"query": "cholangiocarcinoma bile duct adenocarcinoma",             "expected": ["Hepatopathology"]},

    # --- Ophthalmic (1) — slightly improved ---
    {"query": "retinoblastoma eye tumor Flexner-Wintersteiner rosettes", "expected": ["Ophthalmic"]},

    # --- Cytopathology (2) — +1 new ---
    {"query": "fine needle aspiration cytology smear",                   "expected": ["Cytopathology"]},
    {"query": "Pap smear cervical cytology squamous cells",              "expected": ["Cytopathology"]},

    # --- General / staining (3) — no expected category ---
    {"query": "H&E stained tissue section",                              "expected": []},
    {"query": "immunohistochemistry IHC staining",                       "expected": []},
    {"query": "hematoxylin eosin stained biopsy",                        "expected": []},
]


def main():
    parser = argparse.ArgumentParser(description="Write eval_queries_v2.json")
    parser.add_argument("--out", default="artifacts/eval_queries_v2.json")
    args = parser.parse_args()

    numbered = [
        {"id": f"q{i+1:03d}", "query": q["query"], "expected": q["expected"]}
        for i, q in enumerate(QUERIES)
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(numbered, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cats: dict[str, int] = {}
    for q in numbered:
        for c in q["expected"]:
            cats[c] = cats.get(c, 0) + 1
    no_cat = sum(1 for q in numbered if not q["expected"])

    labelled = sum(1 for q in numbered if q["expected"])
    print(f"[OK] {len(numbered)} queries ({labelled} labelled + {no_cat} general) -> {out_path}")
    print("\nCategory breakdown (queries per category):")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        change = ""
        v1_counts = {
            "Dermatopathology": 10, "Gastrointestinal": 7, "Pulmonary": 6,
            "Gynecologic": 5, "Soft tissue": 5, "Genitourinary": 4,
            "Hematopathology": 4, "Renal": 3, "Bone": 3, "Neuropathology": 3,
            "Breast": 3, "Endocrine": 2, "Head and Neck": 2, "Cardiac": 2,
            "Hepatopathology": 2, "Ophthalmic": 1, "Cytopathology": 1,
        }
        v1 = v1_counts.get(cat, 0)
        if cnt > v1:
            change = f"  (+{cnt - v1} from V1)"
        print(f"  {cat:<35} {cnt}{change}")
    print(f"  {'(no category)':<35} {no_cat}")


if __name__ == "__main__":
    main()
