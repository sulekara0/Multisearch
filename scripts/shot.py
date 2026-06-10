"""
shot.py — Capture Streamlit UI screenshots via Playwright.

Drives the running app on :8501: types a query, waits for the result, and
screenshots. Captures the domain_out warning (out-of-domain query) and a normal
ok result (in-domain) for comparison.

Usage (server must already be running on :8501):
    python scripts/shot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).parent.parent
OUT = ROOT / "artifacts" / "shots"
OUT.mkdir(parents=True, exist_ok=True)

URL = "http://localhost:8501"
PLACEHOLDER = "lung tissue with cancer cells"


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.goto(URL, wait_until="networkidle", timeout=60_000)

        # Wait for the text input to be ready
        inp = page.get_by_placeholder(PLACEHOLDER)
        inp.wait_for(timeout=30_000)

        # --- 1) Out-of-domain query → domain_out warning ---
        inp.click()
        inp.fill("araba motoru")
        inp.press("Enter")
        page.wait_for_selector("text=histopatoloji alanıyla", timeout=90_000)
        page.wait_for_timeout(1200)  # let layout settle
        page.screenshot(path=str(OUT / "domain_out.png"), full_page=True)
        print(f"[shot] domain_out → {OUT / 'domain_out.png'}")

        # --- 2) In-domain query → ok results ---
        inp = page.get_by_placeholder(PLACEHOLDER)
        inp.click()
        inp.fill("lung cancer biopsy")
        inp.press("Enter")
        page.wait_for_selector("text=domain:", timeout=90_000)
        page.wait_for_timeout(2500)  # let images render
        page.screenshot(path=str(OUT / "ok_results.png"), full_page=True)
        print(f"[shot] ok_results → {OUT / 'ok_results.png'}")

        browser.close()


if __name__ == "__main__":
    run()
