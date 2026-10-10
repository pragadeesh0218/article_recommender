"""Recommend an article:  python -m recommender.predict "how do I add an employee"   (no text = interactive)"""
import argparse

from .model import Recommender


def show(res):
    if res["decision"] == "agent":
        print(f"-> Suggest a human agent ({res['reason']}): {res['note']}\n   Contact: {res['agent_url']}")
        if res["matches"]:
            print("   (closest articles, not confident enough: " + ", ".join(m["name"] for m in res["matches"]) + ")")
        return
    a = res["article"]
    print(f"-> Recommended article: {a['name']}  (confidence {res['confidence']:.2f})")
    print(f"   Open: {a['link'] or '(no link in the CSV yet)'}")
    for m in res["matches"][1:]:
        print(f"   also: {m['name']}  ({m['score']:.2f})")


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
