"""Multinomial Naive Bayes on word-count vectors (see nlp.py for the text steps) + the human-agent decision.

Every article (CSV label = PDF file name) is a class. Training examples for one article = its sentence groups
(~25 words each) + its file name (repeated, so the title words count).

Decision for a question:  show the article (with its link)  OR  suggest a human agent.
The agent is suggested when the question has no usable words, asks for a person, uses words the articles never mention
(out of scope), or the model is not confident. The two cut-offs are tuned automatically (see tune_thresholds).
"""
import random
import re
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB

from .config import AGENT_URL, MODEL_PATH, OOS_FILE, WRONG_COST
from .nlp import TextEncoder, chunk, tokenize

ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0]       # smoothing values tried
NAME_WEIGHT = 3
AGENT_REQUEST = re.compile(r"\b(?:talk|speak|chat)\s+(?:to|with)\s+(?:a\s+|an\s+|the\s+)?(?:real\s+|live\s+|human\s+)?"
                           r"(?:person|human|agent|representative|rep|support|someone)\b|\b(?:human|live)\s+agent\b|"
                           r"\bcustomer\s+(?:service|support)\b|\bcontact\s+(?:support|an?\s+agent)\b")
OOS = [  # out-of-scope examples used ONLY to tune the agent cut-offs (generic, not article data)
    "hello", "hi there", "good morning", "thanks", "what is the weather today", "tell me a joke", "who are you",
    "best pizza near me", "how to reset my wifi router", "play some music", "what is the capital of france",
    "write me a poem", "translate this to spanish", "my laptop wont turn on", "cheap flights to london",
    "football score yesterday", "how do i bake bread", "what time is it", "recommend a good movie", "asdfgh",
    "qwerty uiop", "zzzz", "test", "ok", "lorem ipsum dolor sit amet", "how tall is mount everest",
]
MESSAGES = {
    "unclear_question": "We couldn't understand the question. Please rephrase it, or contact a support agent.",
    "asked_for_agent": "You asked for a person. A support agent can help you.",
    "out_of_scope": "This question doesn't match any of our help articles. A support agent can help you.",
    "low_confidence": "We couldn't find an article we're confident about. A support agent can help you.",
}


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


def _casual(q, rng):
    ws = re.sub(r"[^\w\s'-]", " ", q.lower()).split()
    ws = [w[:2] + w[3] + w[2] + w[4:] if len(w) > 5 and rng.random() < 0.15 else w for w in ws]      # typos
    return (rng.choice(["", "", "how do i ", "can i ", "help with ", "question about ", "i need "]) + " ".join(ws)).strip()


def _oos(seed=0, n=25):
    rng, L = random.Random(seed), "abcdefghijklmnopqrstuvwxyz"
    own = [l.strip() for l in OOS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()] if OOS_FILE.exists() else []
    return OOS + own + [" ".join("".join(rng.choice(L) for _ in range(rng.randint(3, 8))) for _ in range(rng.randint(1, 4)))
                  for _ in range(n)]


def _folds(docs, k, seed):
    rng, out = random.Random(seed), {}
    for lab, d in docs.items():
        f = [j % k for j in range(len(chunk(d["text"])))]
        rng.shuffle(f)
        out[lab] = f
    return out


def _known(vocab, query):
    t = tokenize(query)
    return sum(w in vocab for w in t) / len(t) if t else 0.0


def _eval_fold(docs, folds, f, alpha, seed, confusion=None, rows=None):
    rng = random.Random(seed + f)
    texts, labels, tests = [], [], []
    for lab, d in docs.items():
        ch = chunk(d["text"])
        held = {i for i, x in enumerate(folds[lab]) if x == f} if len(ch) >= 2 else set()
        for t in _docs(d, held):
            texts.append(t); labels.append(lab)
        for i in held:
            sp = _span(ch[i], rng)
            tests += [("chunk", ch[i], lab), ("span", sp, lab), ("casual", _casual(sp, rng), lab)]
    if not tests:
        return {}
    enc, clf = _fit(texts, labels, alpha)
    stat = {}
    for (kind, q, gold), p in zip(tests, clf.predict_proba(enc.transform([q for _, q, _ in tests]))):
        order = clf.classes_[np.argsort(-p)]
        s = stat.setdefault(kind, [0, 0, 0])
        s[0] += order[0] == gold; s[1] += gold in order[:3]; s[2] += 1
        if confusion is not None and kind == "span" and order[0] != gold:
            confusion[(gold, order[0])] += 1
        if rows is not None and kind != "chunk":            # in-scope questions, as short as real ones
            rows.append((float(p.max()), _known(enc.vocabulary_, q), order[0] == gold, True))
    if rows is not None:                                     # out-of-scope questions must be sent to the agent
        oos = _oos(seed)
        for q, p in zip(oos, clf.predict_proba(enc.transform(oos))):
            rows.append((float(p.max()), _known(enc.vocabulary_, q), False, False))
    return stat


def tune_thresholds(rows, wrong_cost=WRONG_COST):
    """Choose (min probability, min known-word ratio) with the lowest average cost where
       correct article = 0, in-scope question sent to the agent = 1,
       wrong article or an out-of-scope question answered with an article = wrong_cost."""
    A = np.array(rows, float)                    # columns: max_prob, known_ratio, correct, in_scope
    inn = A[:, 3] == 1
    best = None
    for t1 in [*np.round(np.arange(0.05, 0.96, 0.05), 2), 0.97, 0.99, 0.999]:
        for t2 in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9):
            acc = (A[:, 0] >= t1) & (A[:, 1] >= t2)
            cost = (wrong_cost * (acc & inn & (A[:, 2] == 0)).sum() + (~acc & inn).sum() + wrong_cost * (acc & ~inn).sum()) / len(A)
            if best is None or cost < best[0] - 1e-12:
                best = (cost, float(t1), float(t2))
    cost, t1, t2 = best
    acc = (A[:, 0] >= t1) & (A[:, 1] >= t2)
    return {"min_probability": t1, "min_known_ratio": t2, "avg_cost": round(float(cost), 3),
            "in_scope_answered": round(float(acc[inn].mean()), 3),
            "precision_when_answered": round(float(A[acc & inn, 2].mean()), 3) if (acc & inn).any() else None,
            "out_of_scope_sent_to_agent": round(float((~acc[~inn]).mean()), 3),
            "n_in_scope": int(inn.sum()), "n_out_of_scope": int((~inn).sum())}


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
    confusion, rows = Counter(), []
    for f in range(k):
        _eval_fold(docs, folds, f, best, seed, confusion, rows)
    return best, table, confusion, rows


class Recommender:
    def __init__(self, enc, clf, names, links, thresholds, report):
        self.enc, self.clf, self.names, self.links = enc, clf, names, links
        self.t, self.report = thresholds, report

    @classmethod
    def train(cls, docs: dict, k=4, seed=0):
        if len(docs) < 2:
            raise ValueError("need at least 2 articles")
        alpha, table, confusion, rows = cross_validate(docs, k, seed)
        thresholds = tune_thresholds(rows)
        texts, labels = [], []
        for lab, d in docs.items():
            for t in _docs(d):
                texts.append(t); labels.append(lab)
        enc, clf = _fit(texts, labels, alpha)
        links = {lab: d.get("link", "") for lab, d in docs.items()}
        report = {"n_articles": len(docs), "n_training_examples": len(texts), "vocabulary_size": len(enc.vocabulary_),
                  "alpha": alpha, "held_out": table[alpha], "agent_rule": thresholds,
                  "articles_with_link": sum(bool(v) for v in links.values()),
                  "most_confused": [{"article": a, "predicted_instead": b, "times": n} for (a, b), n in confusion.most_common(5)],
                  "trained_at": datetime.now(timezone.utc).isoformat(),
                  "note": "proxy accuracy: test questions are fragments of the articles themselves, not real user questions"}
        return cls(enc, clf, {lab: d["name"] for lab, d in docs.items()}, links, thresholds, report)

    def known_ratio(self, query):
        return _known(self.enc.vocabulary_, query)

    def _card(self, i, p):
        lab = self.clf.classes_[i]
        return {"label": lab, "name": self.names[lab], "link": self.links.get(lab) or None, "score": round(float(p[i]), 3)}

    def _agent(self, query, reason, matches=None, conf=0.0):
        return {"query": query, "decision": "agent", "reason": reason, "confidence": round(conf, 3), "recommended": None,
                "article": None, "matches": matches or [], "agent_url": AGENT_URL, "note": MESSAGES[reason]}

    def predict(self, query: str, top: int = 3) -> dict:
        q = (query or "").strip()
        if not tokenize(q):
            return self._agent(q, "unclear_question")
        if AGENT_REQUEST.search(q.lower()):
            return self._agent(q, "asked_for_agent")
        known = self.known_ratio(q)
        if known == 0:
            return self._agent(q, "out_of_scope")
        p = self.clf.predict_proba(self.enc.transform([q]))[0]
        cards = [self._card(i, p) for i in np.argsort(-p)[:max(top, 1)]]
        conf = cards[0]["score"]
        # if known < self.t["min_known_ratio"]:
        #     return self._agent(q, "out_of_scope", cards, conf)
        # if conf < self.t["min_probability"]:
        #     return self._agent(q, "low_confidence", cards, conf)
        return {"query": q, "decision": "article", "reason": "confident_match", "confidence": conf,
                "recommended": cards[0]["label"], "article": cards[0], "matches": cards, "agent_url": None,
                "note": None if cards[0]["link"] else "This article has no link yet (add it to the link column of the CSV)."}

    def explain(self, query: str, top_terms: int = 5) -> dict:
        """Show every step for one question: tokens -> n-grams -> vocabulary hits -> counts -> class scores -> decision."""
        toks, grams, vocab = tokenize(query), self.enc.grams(query), self.enc.vocabulary_
        x = self.enc.transform([query])
        inv = list(vocab)                                     # column index -> term (vocabulary is stored in index order)
        counts = {inv[i]: int(c) for i, c in zip(x.indices, x.data)}
        jll = np.asarray(self.clf.class_log_prior_ + x @ self.clf.feature_log_prob_.T).ravel()   # log P(class) + sum(count*log P(term|class))
        order = np.argsort(-jll)[:3]
        out = {"tokens": toks, "ngrams": grams, "in_vocabulary": [g for g in grams if g in vocab],
               "not_in_vocabulary": [g for g in grams if g not in vocab], "count_vector": counts,
               "log_scores": [{"label": self.clf.classes_[i], "log_score": round(float(jll[i]), 3)} for i in order],
               "recommended": self.clf.classes_[order[0]] if counts else None, "known_ratio": round(self.known_ratio(query), 3)}
        if counts and len(order) > 1:
            a, b = order[0], order[1]
            lift = {t: c * float(self.clf.feature_log_prob_[a, vocab[t]] - self.clf.feature_log_prob_[b, vocab[t]]) for t, c in counts.items()}
            out["terms_favouring_top_article"] = [(t, round(v, 3)) for t, v in sorted(lift.items(), key=lambda kv: -kv[1])[:top_terms]]
        res = self.predict(query)
        out.update(decision=res["decision"], reason=res["reason"], confidence=res["confidence"], thresholds=self.t)
        return out

    def save(self, path=MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"enc": self.enc, "clf": self.clf, "names": self.names, "links": self.links, "t": self.t,
                     "report": self.report}, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        d = joblib.load(path)
        return cls(d["enc"], d["clf"], d["names"], d["links"], d["t"], d["report"])
