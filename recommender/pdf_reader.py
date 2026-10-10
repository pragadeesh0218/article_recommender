"""Read the downloaded article PDFs.  label = file name (without .pdf), features = the text inside."""
import math
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

MIN_WORDS = 20          # a PDF with fewer words than this is reported as "no text" (probably a scanned image)
_FULL_LINE_NOISE = [re.compile(r"^\s*(?:page\s*)?\d+\s*(?:/|of)\s*\d+\s*$", re.I), re.compile(r"^\s*\d+\s*$"),
                    re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}.*$")]   # "10/8/26, 3:15 PM ..." browser header
_URL = re.compile(r"https?://\S+|www\.\S+")


def extract_lines(path):
    """-> (list of cleaned text lines, number of pages)"""
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = re.sub(r"-\n(?=[a-z])", "", "\n".join(pages))            # re-join words hyphenated at a line break
    lines = []
    for ln in text.splitlines():
        if any(rx.match(ln) for rx in _FULL_LINE_NOISE):
            continue
        ln = " ".join(_URL.sub(" ", ln).split())
        if ln:
            lines.append(ln)
    return lines, len(pages)


def display_name(label):
    return re.sub(r"[_]+", " ", label.split("/")[-1]).strip()


def read_folder(folder, recursive=True):
    """-> (docs, report). docs = {label: {"label","name","text","pages","source"}}"""
    folder = Path(folder)
    paths = sorted(p for p in (folder.rglob("*") if recursive else folder.glob("*")) if p.suffix.lower() == ".pdf")
    raw, report = {}, {"n_pdfs": len(paths), "unreadable": [], "no_text": [], "boilerplate_removed": []}
    for p in paths:
        label = p.stem
        if label in raw:                                              # same file name in two folders
            label = f"{p.parent.name}/{p.stem}"
        n = 2
        while label in raw:
            label = f"{p.stem}-{n}"; n += 1
        try:
            raw[label] = (extract_lines(p), str(p))
        except Exception as e:
            report["unreadable"].append(f"{p.name} ({type(e).__name__})")
    # header/footer text repeated across many PDFs (site name, "Print", ...) is noise, not article content
    if len(raw) >= 5:
        df = Counter()
        for (lines, _), _src in raw.values():
            df.update({ln.lower() for ln in lines})
        limit = max(3, math.ceil(0.4 * len(raw)))
        common = {ln for ln, c in df.items() if c >= limit}
        report["boilerplate_removed"] = sorted(common)[:10]
    else:
        common = set()
    docs = {}
    for label, ((lines, pages), src) in raw.items():
        text = " ".join(ln for ln in lines if ln.lower() not in common)
        if len(text.split()) < MIN_WORDS:
            report["no_text"].append(label); continue
        docs[label] = {"label": label, "name": display_name(label), "text": text, "pages": pages, "source": src}
    return docs, report
