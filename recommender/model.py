"""Naive Bayes recommender: every article (PDF file name) is a class; the words in the PDF are the features.

Training examples for one article = its sentence groups ("chunks") + its file name (repeated, so the title words count).
"""
import random
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from .config import MODEL_PATH
from .preprocess import analyze, chunk, tokens

VARIANTS = {"MultinomialNB": MultinomialNB, "ComplementNB": ComplementNB}
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0]
NAME_WEIGHT = 3


def _docs(doc, skip=()):
    return [c for i, c in enumerate(chunk(doc["text"])) if i not in skip] + [doc["name"]] * NAME_WEIGHT


def _fit(texts, labels, variant, alpha):
    vec = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
    return vec, VARIANTS[variant](alpha=alpha).fit(vec.fit_transform(texts), labels)


def _span(text, rng, lo=4, hi=8):
    w = text.split()
    if len(w) <= lo:
        return text
    n = rng.randint(lo, min(hi, len(w)))
    s = rng.randint(0, len(w) - n)
    return " ".join(w[s:s + n])


def _folds(docs, k, seed):
    rng, out = random.Random(seed), {}
    for lab, d in docs.items():
        f = [j % k for j in range(len(chunk(d["text"])))]
        rng.shuffle(f)
        out[lab] = f
    return out


def _eval_fold(docs, folds, f, variant, alpha, seed, confusion=None):
    rng = random.Random(seed + f)
    texts, labels, tests = [], [], []
    for lab, d in docs.items():
        ch = chunk(d["text"])
        held = {i for i, x in enumerate(folds[lab]) if x == f} if len(ch) >= 2 else set()
        for t in _docs(d, held):
            texts.append(t); labels.append(lab)
        for i in held:
            tests += [("chunk", ch[i], lab), ("span", _span(ch[i], rng), lab)]
    if not tests:
        return {}
    vec, clf = _fit(texts, labels, variant, alpha)
    stat = {}
    for (kind, _, gold), p in zip(tests, clf.predict_proba(vec.transform([q for _, q, _ in tests]))):
        order = clf.classes_[np.argsort(-p)]
        s = stat.setdefault(kind, [0, 0, 0])
        s[0] += order[0] == gold; s[1] += gold in order[:3]; s[2] += 1
        if confusion is not None and kind == "span" and order[0] != gold:
            confusion[(gold, order[0])] += 1
    return stat


def cross_validate(docs, k=4, seed=0):
    """Hold out sentence groups of every article, train on the rest, ask the model to find the article again."""
    folds, table = _folds(docs, k, seed), {}
    for variant in VARIANTS:
        for alpha in ALPHAS:
            agg = {}
            for f in range(k):
                for kind, (a, b, n) in _eval_fold(docs, folds, f, variant, alpha, seed).items():
                    t = agg.setdefault(kind, [0, 0, 0]); t[0] += a; t[1] += b; t[2] += n
            table[(variant, alpha)] = {kd: {"top1": round(a / n, 3), "top3": round(b / n, 3), "n": n} for kd, (a, b, n) in agg.items()}
    usable = [c for c in table if "span" in table[c]]
    if not usable:
        raise ValueError("articles are too short to test: each needs at least 2 sentence groups (~50 words)")
    best = max(usable, key=lambda c: (table[c]["span"]["top1"], table[c]["span"]["top3"]))
    confusion = Counter()
    for f in range(k):
        _eval_fold(docs, folds, f, *best, seed, confusion)
    return best, table, confusion


class Recommender:
    def __init__(self, vec, clf, names, report):
        self.vec, self.clf, self.names, self.report = vec, clf, names, report
        self.vocab = {w for w in vec.vocabulary_ if "_" not in w}

    @classmethod
    def train(cls, docs: dict, k=4, seed=0):
        if len(docs) < 2:
            raise ValueError("need at least 2 articles")
        (variant, alpha), table, confusion = cross_validate(docs, k, seed)
        texts, labels = [], []
        for lab, d in docs.items():
            for t in _docs(d):
                texts.append(t); labels.append(lab)
        vec, clf = _fit(texts, labels, variant, alpha)
        report = {"n_articles": len(docs), "n_training_examples": len(texts), "selected": [variant, alpha],
                  "held_out": table[(variant, alpha)],
                  "most_confused": [{"article": a, "predicted_instead": b, "times": n} for (a, b), n in confusion.most_common(5)],
                  "trained_at": datetime.now(timezone.utc).isoformat(),
                  "note": "proxy accuracy: test queries are fragments of the articles themselves, not real user questions"}
        return cls(vec, clf, {lab: d["name"] for lab, d in docs.items()}, report)

    def predict(self, query: str, top: int = 3) -> dict:
        t = tokens(query)
        known = sum(w in self.vocab for w in t) / len(t) if t else 0.0
        if known == 0:
            return {"query": query, "recommended": None, "matches": [],
                    "note": "None of the words in your question appear in the articles."}
        p = self.clf.predict_proba(self.vec.transform([query]))[0]
        matches = [{"label": self.clf.classes_[i], "name": self.names[self.clf.classes_[i]], "score": round(float(p[i]), 3)}
                   for i in np.argsort(-p)[:top]]
        return {"query": query, "recommended": matches[0]["label"], "matches": matches, "note": None}

    def save(self, path=MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vec": self.vec, "clf": self.clf, "names": self.names, "report": self.report}, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        d = joblib.load(path)
        return cls(d["vec"], d["clf"], d["names"], d["report"])
