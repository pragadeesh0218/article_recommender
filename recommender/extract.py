"""STEP 1: open every PDF in articles/, read its text, save it to a CSV together with its label (the file name).

    python -m recommender.extract                       ->  data/articles.csv

CSV columns:  label, link, file, pages, words, text      (one row per PDF)
`link` is the article's web address (fill it in yourself). Running extract again KEEPS the links you already typed in
(matched by label = PDF file name), so renaming a PDF loses its link.
"""
import argparse
import csv
import io
import re
from pathlib import Path

from .config import ARTICLES_DIR, CSV_PATH
from .pdf_reader import display_name, read_folder

COLUMNS = ["label", "link", "file", "pages", "words", "text"]
_LINK_ALIASES = ("link", "url")


def clean_link(value):
    """Only http(s) addresses are accepted, so a CSV cell can never put a javascript: or file: link on the page."""
    v = (value or "").strip()
    return v if re.match(r"^https?://\S+$", v, re.I) else ""


def _read_rows(path):
    """CSV -> list of dicts with lower-case, trimmed headers. Excel on Windows may save as cp1252, so fall back to it."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    rows = []
    for r in csv.DictReader(io.StringIO(text, newline="")):
        rows.append({(k or "").strip().lower(): (v or "") for k, v in r.items()})
    return rows


def _link_of(row):
    for a in _LINK_ALIASES:
        if a in row:
            return row[a]
    return ""


def existing_links(csv_path):
    if not Path(csv_path).exists():
        return {}
    return {r["label"]: clean_link(_link_of(r)) for r in _read_rows(csv_path) if r.get("label") and clean_link(_link_of(r))}


def run(folder=ARTICLES_DIR, csv_path=CSV_PATH):
    folder, csv_path = Path(folder), Path(csv_path)
    prior = existing_links(csv_path)                       # read BEFORE overwriting the file
    docs, report = read_folder(folder)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for d in docs.values():
            w.writerow([d["label"], prior.get(d["label"], ""), Path(d["source"]).relative_to(folder).as_posix(),
                        d["pages"], len(d["text"].split()), d["text"]])
    report["n_rows"] = len(docs)
    report["links_kept"] = sum(1 for d in docs if d in prior)
    report["links_lost"] = sorted(set(prior) - set(docs))
    return docs, report


def load_csv(csv_path=CSV_PATH) -> dict:
    """CSV -> {label: {"label","name","text","link","pages","source","link_problem"}}"""
    docs = {}
    for row in _read_rows(csv_path):
        label, text = row.get("label", "").strip(), row.get("text", "")
        if label and text.strip():
            raw = _link_of(row).strip()
            docs[label] = {"label": label, "name": display_name(label), "text": text, "link": clean_link(raw),
                           "link_problem": raw if raw and not clean_link(raw) else "",
                           "pages": int(row.get("pages") or 0), "source": row.get("file", "")}
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(ARTICLES_DIR))
    ap.add_argument("--csv", default=str(CSV_PATH))
    a = ap.parse_args()
    docs, rep = run(a.folder, a.csv)
    print(f"PDFs found: {rep['n_pdfs']} | rows written: {rep['n_rows']} -> {a.csv}")
    total = sum(len(d["text"].split()) for d in docs.values())
    if docs:
        short = sorted(docs.values(), key=lambda d: len(d["text"].split()))[:3]
        print(f"total words: {total} | shortest: " + ", ".join(f"{d['label']} ({len(d['text'].split())})" for d in short))
    print(f"links kept from the old CSV: {rep['links_kept']} | rows without a link: {rep['n_rows'] - rep['links_kept']}")
    if rep["links_lost"]:
        print("WARNING these labels had a link but no PDF with that name any more:", rep["links_lost"])
    if rep["no_text"]:
        print("WARNING no text found (scanned image PDFs need OCR):", rep["no_text"])
    if rep["unreadable"]:
        print("WARNING unreadable PDFs:", rep["unreadable"])
    if rep["boilerplate_removed"]:
        print("removed repeated header/footer lines, e.g.:", rep["boilerplate_removed"][:3])


if __name__ == "__main__":
    main()
