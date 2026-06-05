"""
make_queries.py — Generate eval_queries.json for benchmark evaluation.

Usage:
    python scripts/make_queries.py
    python scripts/make_queries.py --out artifacts/eval_queries.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QUERIES = [
    # --- Dermatopathology (10) ---
    {"query": "basal cell carcinoma skin biopsy",                "expected": ["Dermatopathology"]},
    {"query": "melanoma histology",                              "expected": ["Dermatopathology"]},
    {"query": "squamous cell carcinoma skin",                    "expected": ["Dermatopathology"]},
    {"query": "psoriasis skin biopsy",                           "expected": ["Dermatopathology"]},
    {"query": "dermatofibrosarcoma protuberans DFSP",            "expected": ["Dermatopathology"]},
    {"query": "sebaceous carcinoma",                             "expected": ["Dermatopathology"]},
    {"query": "actinic keratosis",                               "expected": ["Dermatopathology"]},
    {"query": "pemphigus vulgaris",                              "expected": ["Dermatopathology"]},
    {"query": "cutaneous T-cell lymphoma mycosis fungoides",     "expected": ["Dermatopathology"]},
    {"query": "lichen planus skin biopsy",                       "expected": ["Dermatopathology"]},

    # --- Gastrointestinal (7) ---
    {"query": "colon adenocarcinoma",                            "expected": ["Gastrointestinal"]},
    {"query": "gastric cancer histology",                        "expected": ["Gastrointestinal"]},
    {"query": "colorectal polyp adenoma",                        "expected": ["Gastrointestinal"]},
    {"query": "Crohn disease intestinal biopsy",                 "expected": ["Gastrointestinal"]},
    {"query": "dysplasia in colonic mucosa",                     "expected": ["Gastrointestinal"]},
    {"query": "esophageal squamous cell carcinoma",              "expected": ["Gastrointestinal"]},
    {"query": "pancreatic ductal adenocarcinoma",                "expected": ["Gastrointestinal"]},

    # --- Pulmonary (6) ---
    {"query": "lung adenocarcinoma",                             "expected": ["Pulmonary"]},
    {"query": "lung squamous cell carcinoma",                    "expected": ["Pulmonary"]},
    {"query": "small cell lung cancer",                          "expected": ["Pulmonary"]},
    {"query": "pulmonary fibrosis",                              "expected": ["Pulmonary"]},
    {"query": "mesothelioma pleural",                            "expected": ["Pulmonary"]},
    {"query": "lung tissue with cancer cells",                   "expected": ["Pulmonary"]},

    # --- Gynecologic (5) ---
    {"query": "endometrial carcinoma",                           "expected": ["Gynecologic"]},
    {"query": "cervical squamous cell carcinoma",                "expected": ["Gynecologic"]},
    {"query": "ovarian serous carcinoma",                        "expected": ["Gynecologic"]},
    {"query": "uterine leiomyoma fibroid",                       "expected": ["Gynecologic"]},
    {"query": "vulvar intraepithelial neoplasia",                "expected": ["Gynecologic"]},

    # --- Soft tissue (5) ---
    {"query": "soft tissue sarcoma",                             "expected": ["Soft tissue"]},
    {"query": "leiomyosarcoma smooth muscle tumor",              "expected": ["Soft tissue"]},
    {"query": "liposarcoma adipose tissue",                      "expected": ["Soft tissue"]},
    {"query": "rhabdomyosarcoma skeletal muscle",                "expected": ["Soft tissue"]},
    {"query": "synovial sarcoma",                                "expected": ["Soft tissue"]},

    # --- Genitourinary (4) ---
    {"query": "prostate adenocarcinoma Gleason score",           "expected": ["Genitourinary"]},
    {"query": "bladder transitional cell carcinoma",             "expected": ["Genitourinary"]},
    {"query": "testicular seminoma",                             "expected": ["Genitourinary"]},
    {"query": "urothelial carcinoma",                            "expected": ["Genitourinary"]},

    # --- Hematopathology (4) ---
    {"query": "diffuse large B-cell lymphoma",                   "expected": ["Hematopathology"]},
    {"query": "lymph node with metastatic carcinoma",            "expected": ["Hematopathology"]},
    {"query": "chronic lymphocytic leukemia bone marrow",        "expected": ["Hematopathology"]},
    {"query": "Hodgkin lymphoma Reed-Sternberg cells",           "expected": ["Hematopathology"]},

    # --- Renal (3) ---
    {"query": "clear cell renal cell carcinoma",                 "expected": ["Renal"]},
    {"query": "glomerulonephritis kidney biopsy",                "expected": ["Renal"]},
    {"query": "kidney tubular necrosis",                         "expected": ["Renal"]},

    # --- Bone (3) ---
    {"query": "osteosarcoma bone tumor",                         "expected": ["Bone"]},
    {"query": "chondrosarcoma cartilage",                        "expected": ["Bone"]},
    {"query": "giant cell tumor of bone",                        "expected": ["Bone"]},

    # --- Neuropathology (3) ---
    {"query": "glioblastoma multiforme brain tumor",             "expected": ["Neuropathology"]},
    {"query": "meningioma brain",                                "expected": ["Neuropathology"]},
    {"query": "astrocytoma glioma",                              "expected": ["Neuropathology"]},

    # --- Breast (3) ---
    {"query": "invasive ductal carcinoma breast",                "expected": ["Breast"]},
    {"query": "ductal carcinoma in situ DCIS",                   "expected": ["Breast"]},
    {"query": "lobular carcinoma breast",                        "expected": ["Breast"]},

    # --- Endocrine (2) ---
    {"query": "thyroid papillary carcinoma",                     "expected": ["Endocrine"]},
    {"query": "adrenal pheochromocytoma",                        "expected": ["Endocrine"]},

    # --- Head and Neck (2) ---
    {"query": "squamous cell carcinoma head neck",               "expected": ["Head and Neck"]},
    {"query": "salivary gland pleomorphic adenoma",              "expected": ["Head and Neck"]},

    # --- Cardiac (2) ---
    {"query": "myocardial infarction cardiac fibrosis",          "expected": ["Cardiac"]},
    {"query": "cardiac myxoma",                                  "expected": ["Cardiac"]},

    # --- Hepatopathology (2) ---
    {"query": "hepatocellular carcinoma liver",                  "expected": ["Hepatopathology"]},
    {"query": "liver cirrhosis fibrosis",                        "expected": ["Hepatopathology"]},

    # --- Ophthalmic (1) ---
    {"query": "retinoblastoma eye tumor",                        "expected": ["Ophthalmic"]},

    # --- Cytopathology (1) ---
    {"query": "fine needle aspiration cytology smear",           "expected": ["Cytopathology"]},

    # --- General / staining (3) — no expected category ---
    {"query": "H&E stained tissue section",                      "expected": []},
    {"query": "immunohistochemistry IHC staining",               "expected": []},
    {"query": "hematoxylin eosin stained biopsy",                "expected": []},
]


def main():
    parser = argparse.ArgumentParser(description="Write eval_queries.json")
    parser.add_argument(
        "--out", default="artifacts/eval_queries.json",
        help="Output path (default: artifacts/eval_queries.json)"
    )
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
    cats["(no category)"] = sum(1 for q in numbered if not q["expected"])

    print(f"[OK] {len(numbered)} queries -> {out_path}")
    print("\nCategory breakdown:")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:<35} {cnt}")


if __name__ == "__main__":
    main()
