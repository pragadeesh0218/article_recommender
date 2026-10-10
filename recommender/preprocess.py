"""Text preprocessing used for BOTH training text and user questions."""
import re
from functools import lru_cache

from nltk.stem import PorterStemmer

_STEM = PorterStemmer()
STOP = set("""a an the of to in on for and or is are was were be been am i my me we you your it its this that these those
with at by from as do does did how what when where why which who can could should would will if then than so about into
over after before any some there here have has had get got please help need want just also not no yes""".split())
_PII = [(re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), " "), (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), " ")]


@lru_cache(maxsize=50000)
def stem(w):
    return _STEM.stem(w)


def tokens(text):
    """lower-case -> remove emails/SSN -> words -> drop stop-words -> stem ('w-2' also gives 'w2')."""
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


def analyze(text):
    """Features: stemmed words + word pairs."""
    t = tokens(text)
    return t + [f"{a}_{b}" for a, b in zip(t, t[1:])]


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
