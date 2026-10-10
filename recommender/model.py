"""STEP 2: Multinomial Naive Bayes on word-count vectors (see nlp.py for the text steps).

Every article (CSV label = PDF file name) is a class. Training examples for one article = its sentence groups
(~25 words each) + its file name (repeated, so the title words count).
"""
import random
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB

from .config import MODEL_PATH
from .nlp import TextEncoder, chunk, tokenize

ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0]       # Laplace/Lidstone smoothing values tried
NAME_WEIGHT = 3


def _docs(doc, skip=()):
    return [c for i, c in enumerate(chunk(doc["text"])) if i not in skip] + [doc["name"]] * NAME_WEIGHT


def _fit(texts, labels, alpha):
    enc = TextEncoder()
    return enc, MultinomialNB(alpha=alpha).fit(enc.fit_transform(texts), labels)


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


def _eval_fold(docs, folds, f, alpha, seed, confusion=None):
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
    enc, clf = _fit(texts, labels, alpha)
    stat = {}
    for (kind, _, gold), p in zip(tests, clf.predict_proba(enc.transform([q for _, q, _ in tests]))):
        order = clf.classes_[np.argsort(-p)]
        s = stat.setdefault(kind, [0, 0, 0])
        s[0] += order[0] == gold; s[1] += gold in order[:3]; s[2] += 1
        if confusion is not None and kind == "span" and order[0] != gold:
            confusion[(gold, order[0])] += 1
    return stat


def cross_validate(docs, k=4, seed=0):
    """Hold out sentence groups of every article, train on the rest, ask the model to find the article again."""
    folds, table = _folds(docs, k, seed), {}
    for alpha in ALPHAS:
        agg = {}
        for f in range(k):
            for kind, (a, b, n) in _eval_fold(docs, folds, f, alpha, seed).items():
                t = agg.setdefault(kind, [0, 0, 0]); t[0] += a; t[1] += b; t[2] += n
        table[alpha] = {kd: {"top1": round(a / n, 3), "top3": round(b / n, 3), "n": n} for kd, (a, b, n) in agg.items()}
    usable = [a for a in table if "span" in table[a]]
    if not usable:
        raise ValueError("articles are too short to test: each needs at least 2 sentence groups (~50 words)")
    best = max(usable, key=lambda a: (table[a]["span"]["top1"], table[a]["span"]["top3"], a))
    confusion = Counter()
    for f in range(k):
        _eval_fold(docs, folds, f, best, seed, confusion)
    return best, table, confusion


class Recommender:
    def __init__(self, enc, clf, names, report):
        self.enc, self.clf, self.names, self.report = enc, clf, names, report

    @classmethod
    def train(cls, docs: dict, k=4, seed=0):
        if len(docs) < 2:
            raise ValueError("need at least 2 articles")
        alpha, table, confusion = cross_validate(docs, k, seed)
        texts, labels = [], []
        for lab, d in docs.items():
            for t in _docs(d):
                texts.append(t); labels.append(lab)
        enc, clf = _fit(texts, labels, alpha)
        report = {"n_articles": len(docs), "n_training_examples": len(texts), "vocabulary_size": len(enc.vocabulary_),
                  "alpha": alpha, "held_out": table[alpha],
                  "most_confused": [{"article": a, "predicted_instead": b, "times": n} for (a, b), n in confusion.most_common(5)],
                  "trained_at": datetime.now(timezone.utc).isoformat(),
                  "note": "proxy accuracy: test queries are fragments of the articles themselves, not real user questions"}
        return cls(enc, clf, {lab: d["name"] for lab, d in docs.items()}, report)

    def known_ratio(self, query):
        t = tokenize(query)
        return sum(w in self.enc.vocabulary_ for w in t) / len(t) if t else 0.0

    def predict(self, query: str, top: int = 3) -> dict:
        if self.known_ratio(query) == 0:
            return {"query": query, "recommended": None, "matches": [],
                    "note": "None of the words in your question appear in the articles."}
        p = self.clf.predict_proba(self.enc.transform([query]))[0]
        matches = [{"label": self.clf.classes_[i], "name": self.names[self.clf.classes_[i]], "score": round(float(p[i]), 3)}
                   for i in np.argsort(-p)[:top]]
        return {"query": query, "recommended": matches[0]["label"], "matches": matches, "note": None}

    def explain(self, query: str, top_terms: int = 5) -> dict:
        """Show every step for one question: tokens -> n-grams -> vocabulary hits -> counts -> class scores."""
        toks = tokenize(query)
        grams = self.enc.grams(query)
        vocab = self.enc.vocabulary_
        x = self.enc.transform([query])
        inv = list(vocab)                                     # column index -> term (vocabulary is stored in index order)
        counts = {inv[i]: int(c) for i, c in zip(x.indices, x.data)}
        jll = self.clf.class_log_prior_ + x @ self.clf.feature_log_prob_.T      # log P(class) + sum(count * log P(term | class))
        jll = np.asarray(jll).ravel()
        order = np.argsort(-jll)[:3]
        out = {"tokens": toks, "ngrams": grams, "in_vocabulary": [g for g in grams if g in vocab],
               "not_in_vocabulary": [g for g in grams if g not in vocab], "count_vector": counts,
               "log_scores": [{"label": self.clf.classes_[i], "log_score": round(float(jll[i]), 3)} for i in order],
               "recommended": self.clf.classes_[order[0]] if counts else None}
        if counts and len(order) > 1:
            a, b = order[0], order[1]
            lift = {t: c * float(self.clf.feature_log_prob_[a, vocab[t]] - self.clf.feature_log_prob_[b, vocab[t]])
                    for t, c in counts.items()}
            out["terms_favouring_top_article"] = [(t, round(v, 3)) for t, v in sorted(lift.items(), key=lambda kv: -kv[1])[:top_terms]]
        return out

    def save(self, path=MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"enc": self.enc, "clf": self.clf, "names": self.names, "report": self.report}, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        d = joblib.load(path)
        return cls(d["enc"], d["clf"], d["names"], d["report"])
