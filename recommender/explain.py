"""See every step for one question:  python -m recommender.explain "how do I add an employee" """
import sys

from .model import Recommender

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("question> ")
    e = Recommender.load().explain(q)
    print("1. tokens (lower-case, stop-words removed, stemmed):\n  ", e["tokens"])
    print("2. n-grams (1-2 words):\n  ", e["ngrams"])
    print("3. found in the vocabulary:\n  ", e["in_vocabulary"], "\n   unknown (ignored):", e["not_in_vocabulary"])
    print("4. encoded as word counts:\n  ", e["count_vector"])
    print("5. Naive Bayes log-scores (closer to 0 = better):")
    for m in e["log_scores"]:
        print(f"   {m['label']:40} {m['log_score']}")
    if e.get("terms_favouring_top_article"):
        print("   words that pushed the model toward the top article:", e["terms_favouring_top_article"])
    print("=> recommended:", e["recommended"])
