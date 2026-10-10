"""STEP 2: train from the CSV:  python -m recommender.train   (reads data/articles.csv, writes models/model.joblib)"""
import argparse
import csv
import json
from pathlib import Path

from .config import CSV_PATH, MODEL_PATH, VOCAB_PATH
from .extract import load_csv
from .model import Recommender


def run(csv_path=CSV_PATH, model_path=MODEL_PATH, vocab_path=None, k=4):
    if not Path(csv_path).exists():
        raise SystemExit(f"{csv_path} not found. Run step 1 first:  python -m recommender.extract")
    docs = load_csv(csv_path)
    if len(docs) < 2:
        raise SystemExit(f"Only {len(docs)} articles with text in {csv_path}; need at least 2.")
    model = Recommender.train(docs, k=k)
    model.save(model_path)
    if vocab_path:
        enc = model.enc
        Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["index", "term", "document_frequency"])
            for t, i in enc.vocabulary_.items():
                w.writerow([i, t, enc.df_[t]])
    return model, docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_PATH))
    a = ap.parse_args()
    model, docs = run(a.csv, vocab_path=VOCAB_PATH)
    r = model.report
    print(f"articles (classes): {r['n_articles']} | training examples: {r['n_training_examples']} | vocabulary: {r['vocabulary_size']} terms")
    print(f"Multinomial Naive Bayes, smoothing alpha = {r['alpha']}")
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
    print(f"saved {MODEL_PATH}, {VOCAB_PATH}")


if __name__ == "__main__":
    main()
