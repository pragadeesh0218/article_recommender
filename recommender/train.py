"""Train:  python -m recommender.train [--folder articles]   (reads the PDFs, writes models/model.joblib)"""
import argparse
import json

from .config import ARTICLES_DIR, MODEL_PATH
from .model import Recommender
from .pdf_reader import read_folder


def run(folder=ARTICLES_DIR, model_path=MODEL_PATH, k=4):
    docs, info = read_folder(folder)
    if len(docs) < 2:
        raise SystemExit(f"Found {info['n_pdfs']} PDFs in {folder} but only {len(docs)} had readable text. "
                         f"No text: {info['no_text']}  Unreadable: {info['unreadable']}")
    model = Recommender.train(docs, k=k)
    model.save(model_path)
    return model, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(ARTICLES_DIR))
    a = ap.parse_args()
    model, info = run(a.folder)
    r = model.report
    print(f"PDFs found: {info['n_pdfs']} | used as classes: {r['n_articles']} | training examples: {r['n_training_examples']}")
    if info["no_text"]:
        print("WARNING no text could be read (scanned image PDFs need OCR):", info["no_text"])
    if info["unreadable"]:
        print("WARNING unreadable PDFs:", info["unreadable"])
    if info["boilerplate_removed"]:
        print("removed repeated header/footer lines, e.g.:", info["boilerplate_removed"][:3])
    print("model:", r["selected"])
    print("held-out test (sentence groups the model never trained on):")
    for kind, label in (("span", "short fragment (4-8 words)"), ("chunk", "full sentence group")):
        if kind in r["held_out"]:
            h = r["held_out"][kind]
            print(f"  {label:28} top-1 {h['top1']:.2f}   top-3 {h['top3']:.2f}   (n={h['n']})")
    if r["most_confused"]:
        print("most confused article pairs:")
        for c in r["most_confused"]:
            print(f"  {c['article']}  ->  {c['predicted_instead']}  ({c['times']}x)")
    (MODEL_PATH.parent / "report.json").write_text(json.dumps(r, indent=1))
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
