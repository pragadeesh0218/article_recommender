"""STEP 1: open every PDF in articles/, read its text, save it to a CSV together with its label (the file name).

    python -m recommender.extract                       ->  data/articles.csv

CSV columns:  label, file, pages, words, text      (one row per PDF)
"""
import argparse
import csv
from pathlib import Path

from .config import ARTICLES_DIR, CSV_PATH
from .pdf_reader import display_name, read_folder

COLUMNS = ["label", "file", "pages", "words", "text"]


def run(folder=ARTICLES_DIR, csv_path=CSV_PATH):
    folder, csv_path = Path(folder), Path(csv_path)
    docs, report = read_folder(folder)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for d in docs.values():
            w.writerow([d["label"], Path(d["source"]).relative_to(folder).as_posix(), d["pages"], len(d["text"].split()), d["text"]])
    report["n_rows"] = len(docs)
    return docs, report


def load_csv(csv_path=CSV_PATH) -> dict:
    """Read the CSV back -> {label: {"label","name","text","pages","source"}}."""
    docs = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["label"] and row["text"].strip():
                docs[row["label"]] = {"label": row["label"], "name": display_name(row["label"]), "text": row["text"],
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
    if rep["no_text"]:
        print("WARNING no text found (scanned image PDFs need OCR):", rep["no_text"])
    if rep["unreadable"]:
        print("WARNING unreadable PDFs:", rep["unreadable"])
    if rep["boilerplate_removed"]:
        print("removed repeated header/footer lines, e.g.:", rep["boilerplate_removed"][:3])


if __name__ == "__main__":
    main()
