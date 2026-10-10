"""The text pipeline, one explicit step per function:

    text -> tokenize -> make_ngrams -> build_vocabulary -> encode (word-count vector) -> Multinomial Naive Bayes
"""
import re
from collections import Counter
from functools import lru_cache

import numpy as np
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix

_STEM = PorterStemmer()
STOP = set("""a an the of to in on for and or is are was were be been am i my me we you your it its this that these those
with at by from as do does did how what when where why which who can could should would will if then than so about into
over after before any some there here have has had get got please help need want just also not no yes""".split())
_PII = [(re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), " "), (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), " ")]


# ---- step 1: tokenization -------------------------------------------------------------------------------
@lru_cache(maxsize=50000)
def stem(w):
    return _STEM.stem(w)


def tokenize(text):
    """lower-case -> remove emails/SSN -> split into words -> drop stop-words -> stem.  'w-2' also gives 'w2'."""
    for rx, rep in _PII:
        text = rx.sub(rep, text or "")
    out = []
    for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()):
        if w in STOP:
            continue
        out.append(stem(w))
        if "-" in w:
            out.append(stem(w.replace("-", "")))
    return out


# ---- step 2: n-grams ------------------------------------------------------------------------------------
def make_ngrams(tokens, ngram_range=(1, 2)):
    """['add','new','employe'] -> ['add','new','employe','add_new','new_employe']  (unigrams + bigrams)"""
    lo, hi = ngram_range
    return ["_".join(tokens[i:i + n]) for n in range(lo, hi + 1) for i in range(len(tokens) - n + 1)]


# ---- step 3: vocabulary ---------------------------------------------------------------------------------
def build_vocabulary(gram_docs, min_df=1, max_features=None):
    """gram_docs: list of n-gram lists. -> (vocabulary {term: column index}, document_frequency {term: n docs})."""
    df = Counter()
    for g in gram_docs:
        df.update(set(g))
    terms = [t for t, c in df.items() if c >= min_df]
    if max_features:
        terms = sorted(terms, key=lambda t: (-df[t], t))[:max_features]
    terms.sort()
    return {t: i for i, t in enumerate(terms)}, {t: df[t] for t in terms}


# ---- step 4: encoding -----------------------------------------------------------------------------------
def encode(gram_docs, vocabulary):
    """n-gram lists -> sparse matrix of word counts: rows = documents, columns = vocabulary terms."""
    indptr, indices, data = [0], [], []
    for g in gram_docs:
        counts = Counter(t for t in g if t in vocabulary)
        for t, c in counts.items():
            indices.append(vocabulary[t]); data.append(c)
        indptr.append(len(indices))
    return csr_matrix((np.array(data, dtype=float), indices, indptr), shape=(len(gram_docs), len(vocabulary)))


class TextEncoder:
    """tokenize + n-grams + vocabulary + encode, bundled so training and prediction use identical steps."""

    def __init__(self, ngram_range=(1, 2), min_df=1, max_features=None):
        self.ngram_range, self.min_df, self.max_features = ngram_range, min_df, max_features
        self.vocabulary_, self.df_ = {}, {}

    def grams(self, text):
        return make_ngrams(tokenize(text), self.ngram_range)

    def fit(self, texts):
        self.vocabulary_, self.df_ = build_vocabulary([self.grams(t) for t in texts], self.min_df, self.max_features)
        return self

    def transform(self, texts):
        return encode([self.grams(t) for t in texts], self.vocabulary_)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


# ---- helper: cut an article into training examples ------------------------------------------------------
def chunk(text, target=25, min_words=5):
    """Cut an article into sentence groups of ~25 words. Each group is one training example for that article."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]
    chunks, cur, n = [], [], 0
    for s in sents:
        w = s.split()
        for piece in ([" ".join(w[i:i + 30]) for i in range(0, len(w), 30)] if len(w) > 45 else [s]):
            cur.append(piece); n += len(piece.split())
            if n >= target:
                chunks.append(" ".join(cur)); cur, n = [], 0
    if cur:
        tail = " ".join(cur)
        if len(tail.split()) >= min_words or not chunks:
            chunks.append(tail)
        else:
            chunks[-1] += " " + tail
    return chunks
