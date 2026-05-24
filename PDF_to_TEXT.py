import os
import sys
import time
import fitz
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pdf2image import convert_from_path
import pytesseract

# ================= CONFIG =================
POPPLER_PATH = r"C:\Release-24.02.0-0\poppler-24.02.0\Library\bin"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INPUT_FOLDER = r"ESG_Reports Crawl_Manually_2"
OUTPUT_FOLDER = r"pdf_texts_3"
LOG_EXCEL = "progress_log.xlsx"

TEXT_SAMPLE_PAGES = 5
TEXT_MIN_LEN = 300
OCR_DPI = 150
OCR_LANG = "eng"

TEXT_WORKERS = 6       # lightweight
OCR_WORKERS = 3        # CPU heavy → keep small
# =========================================

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- UTIL ----------
def extract_text_pdf(pdf_path, pages=None):
    try:
        doc = fitz.open(pdf_path)
        total = doc.page_count if pages is None else min(pages, doc.page_count)
        texts = []
        for i in range(total):
            texts.append(doc.load_page(i).get_text("text") or "")
        doc.close()
        return "\n".join(texts)
    except Exception:
        return ""

def ocr_pdf_pagewise(pdf_path):
    texts = []
    try:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()

        for p in range(1, page_count + 1):
            images = convert_from_path(
                pdf_path,
                dpi=OCR_DPI,
                first_page=p,
                last_page=p,
                poppler_path=POPPLER_PATH
            )
            text = pytesseract.image_to_string(images[0], lang=OCR_LANG)
            texts.append(text)
        return "\n".join(texts)
    except Exception:
        return ""

# ---------- MAIN ----------
def process_single_pdf(pdf_path):
    fname = os.path.basename(pdf_path)
    out_txt = os.path.join(
        OUTPUT_FOLDER,
        os.path.splitext(fname)[0] + ".txt"
    )

    # ✅ RESUME CHECK
    if os.path.exists(out_txt):
        return fname, "SKIPPED", "already processed"

    t0 = time.time()

    # 1) Fast text probe
    text = extract_text_pdf(pdf_path, pages=TEXT_SAMPLE_PAGES)

    if len(text.strip()) >= TEXT_MIN_LEN:
        # 2) Full text extract
        text = extract_text_pdf(pdf_path, pages=None)
        method = "TEXT"
    else:
        # 3) OCR fallback
        text = ocr_pdf_pagewise(pdf_path)
        method = "OCR"

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text or "")

    elapsed = round(time.time() - t0, 2)
    return fname, method, f"{elapsed}s"

# ---------- DRIVER ----------
def run_batch(folder):
    pdfs = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ]

    results = []

    with ThreadPoolExecutor(max_workers=TEXT_WORKERS) as text_pool, \
         ProcessPoolExecutor(max_workers=OCR_WORKERS) as ocr_pool:

        futures = []
        for pdf in pdfs:
            futures.append(text_pool.submit(process_single_pdf, pdf))

        for fut in as_completed(futures):
            try:
                fname, method, note = fut.result()
                print(f"✔ {fname} [{method}] {note}")
                results.append({
                    "file": fname,
                    "method": method,
                    "status": "OK",
                    "note": note
                })
            except Exception as e:
                print(f"✖ ERROR {e}")
                results.append({
                    "file": pdf,
                    "method": "FAIL",
                    "status": "ERROR",
                    "note": str(e)
                })

    df = pd.DataFrame(results)
    if os.path.exists(LOG_EXCEL):
        old = pd.read_excel(LOG_EXCEL)
        df = pd.concat([old, df], ignore_index=True)

    df.to_excel(LOG_EXCEL, index=False)

# ---------- ENTRY ----------
if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else INPUT_FOLDER
    run_batch(folder)
