"""Recommend an article:  python -m recommender.predict "how do I add an employee"   (no text = interactive)"""
import argparse

from .model import Recommender


def show(res):
    if res["recommended"] is None:
        print(res["note"]); return
    print(f"Recommended article: {res['matches'][0]['name']}")
    for i, m in enumerate(res["matches"], 1):
        print(f"  {i}. {m['name']}  (score {m['score']:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    model = Recommender.load()
    if a.query:
        show(model.predict(" ".join(a.query), a.top)); return
    while True:
        q = input("\nyour question (blank to quit)> ").strip()
        if not q:
            break
        show(model.predict(q, a.top))


if __name__ == "__main__":
    main()
