"""Tests use PDFs generated here (reportlab) with made-up text; they are never used for the real model."""
import csv
import random
import warnings

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

warnings.filterwarnings("ignore")

from recommender import extract, nlp, pdf_reader
from recommender.model import Recommender
from recommender.train import run

TOPICS = {
    "payroll": "payroll paycheck wage deduction withholding stub", "invoice": "invoice customer receivable billing remittance quote",
    "password": "password login credential reset lockout username", "report": "report export spreadsheet filter column chart",
    "employee": "employee hire onboarding profile department manager", "deposit": "deposit routing banking transfer ledger account",
    "tax": "tax filing bracket quarterly deduction credit", "timesheet": "timesheet overtime shift attendance clock schedule",
}


def make_pdf(path, body, pages_header="Pistus Knowledge Base", footer="https://www.pistus.com/kb/article 1/1"):
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    c.drawString(40, 820, pages_header)
    for line in simpleSplit(body, "Helvetica", 11, 500):
        if y < 60:
            c.drawString(40, 30, footer); c.showPage(); c.drawString(40, 820, pages_header); y = 800
        c.drawString(40, y, line); y -= 15
    c.drawString(40, 30, footer)
    c.save()


def body_for(topic, seed=1, n=14):
    rng, w = random.Random(seed), TOPICS[topic].split()
    return " ".join(f"To {rng.choice(w)} the {rng.choice(w)} you must {rng.choice(w)} and then check the {rng.choice(w)} {rng.choice(w)} carefully." for _ in range(n))


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    d = tmp_path_factory.mktemp("articles")
    for t in TOPICS:
        make_pdf(d / f"How_to_{t}.pdf", body_for(t))
    return d


@pytest.fixture(scope="module")
def csv_file(folder, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "articles.csv"
    extract.run(folder, path)
    return path


# ---------------- step 1: PDF -> CSV ----------------
def test_read_labels_text_and_boilerplate(folder):
    docs, rep = pdf_reader.read_folder(folder)
    assert set(docs) == {f"How_to_{t}" for t in TOPICS}                      # label = file name
    assert docs["How_to_payroll"]["name"] == "How to payroll"
    assert "payroll" in docs["How_to_payroll"]["text"]
    assert "Pistus Knowledge Base" not in docs["How_to_payroll"]["text"] and "pistus.com" not in docs["How_to_payroll"]["text"]
    assert rep["n_pdfs"] == 8 and not rep["no_text"] and rep["boilerplate_removed"]


def test_csv_has_label_and_text_and_roundtrips(csv_file):
    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["label", "file", "pages", "words", "text"] and len(rows) == 8
    r = {x["label"]: x for x in rows}["How_to_tax"]
    assert r["file"] == "How_to_tax.pdf" and int(r["words"]) > 50 and "tax" in r["text"]
    docs = extract.load_csv(csv_file)
    assert set(docs) == {f"How_to_{t}" for t in TOPICS} and docs["How_to_tax"]["text"] == r["text"]


def test_csv_survives_commas_quotes_and_unicode(tmp_path):
    d = tmp_path / "pdfs"; d.mkdir()
    for t in list(TOPICS)[:5]:
        make_pdf(d / f"{t}.pdf", body_for(t) + ' He said "hello", then left, café.')
    docs, _ = extract.run(d, tmp_path / "a.csv")
    assert extract.load_csv(tmp_path / "a.csv")["payroll"]["text"] == docs["payroll"]["text"]


def test_blank_corrupt_and_duplicate_names(tmp_path):
    for t in list(TOPICS)[:5]:
        make_pdf(tmp_path / f"a_{t}.pdf", body_for(t))
    c = canvas.Canvas(str(tmp_path / "scanned.pdf")); c.showPage(); c.save()          # page without text
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    (tmp_path / "sub").mkdir()
    make_pdf(tmp_path / "sub" / "a_payroll.pdf", body_for("tax"))                      # same file name in a sub folder
    docs, rep = pdf_reader.read_folder(tmp_path)
    assert rep["no_text"] == ["scanned"] and len(rep["unreadable"]) == 1
    assert "a_payroll" in docs and "sub/a_payroll" in docs and len(docs) == 6


# ---------------- step 2: tokenization, n-gram, vocabulary, encode ----------------
def test_tokenize_ngrams_vocabulary_encode():
    toks = nlp.tokenize("How do I add a new employee? My SSN 123-45-6789")
    assert toks == ["add", "new", "employe", "ssn"]
    assert nlp.make_ngrams(toks, (1, 2)) == toks + ["add_new", "new_employe", "employe_ssn"]
    assert nlp.make_ngrams(toks, (2, 3))[-1] == "new_employe_ssn" and nlp.make_ngrams(["a"], (2, 2)) == []
    docs = [["pay", "tax", "pay"], ["tax", "rare"], ["pay"]]
    vocab, df = nlp.build_vocabulary(docs)
    assert vocab == {"pay": 0, "rare": 1, "tax": 2} and df == {"pay": 2, "rare": 1, "tax": 2}
    v2, _ = nlp.build_vocabulary(docs, min_df=2)
    assert set(v2) == {"pay", "tax"} and len(nlp.build_vocabulary(docs, max_features=1)[0]) == 1
    X = nlp.encode(docs, vocab)
    assert X.shape == (3, 3) and X.toarray().tolist() == [[2, 0, 1], [0, 1, 1], [1, 0, 0]]
    assert nlp.encode([["unknown"]], vocab).toarray().tolist() == [[0, 0, 0]]


def test_encoder_is_consistent_between_fit_and_transform():
    enc = nlp.TextEncoder().fit(["the payroll check runs", "reset the password"])
    X = enc.transform(["payroll check", "password"])
    assert X.shape == (2, len(enc.vocabulary_)) and X.sum(axis=1).A1.tolist() == [3.0, 1.0]       # payroll, check, payroll_check


# ---------------- step 3: Multinomial Naive Bayes ----------------
def test_train_predict_explain_roundtrip(csv_file, tmp_path):
    model, docs = run(csv_file, tmp_path / "m.joblib", tmp_path / "vocab.csv")
    assert len(docs) == 8 and model.report["held_out"]["span"]["top1"] > 0.8 and model.report["vocabulary_size"] > 50
    assert type(model.clf).__name__ == "MultinomialNB"
    rows = list(csv.DictReader(open(tmp_path / "vocab.csv", encoding="utf-8-sig")))
    assert {"index", "term", "document_frequency"} == set(rows[0]) and len(rows) == model.report["vocabulary_size"]
    for t, words in TOPICS.items():
        res = model.predict("how do i " + " ".join(words.split()[:3]))
        assert res["recommended"] == f"How_to_{t}" and len(res["matches"]) == 3
        assert res["matches"][0]["score"] >= res["matches"][1]["score"]
    again = Recommender.load(tmp_path / "m.joblib")
    assert again.predict("reset password login")["matches"] == model.predict("reset password login")["matches"]
    e = again.explain("reset my password login")
    assert e["recommended"] == "How_to_password" and "password" in e["count_vector"] and e["terms_favouring_top_article"]
    assert "reset" in e["in_vocabulary"] and e["log_scores"][0]["log_score"] >= e["log_scores"][1]["log_score"]


def test_unknown_words_recommend_nothing(csv_file, tmp_path):
    model, _ = run(csv_file, tmp_path / "m.joblib")
    for q in ["xyzzy plugh", "", "the of and"]:
        res = model.predict(q)
        assert res["recommended"] is None and res["matches"] == [] and res["note"]
    assert model.explain("xyzzy")["recommended"] is None


def test_train_needs_csv_with_two_articles(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path / "missing.csv", tmp_path / "m.joblib")
    one = tmp_path / "one.csv"
    one.write_text("label,file,pages,words,text\nonly,only.pdf,1,60," + "tax filing bracket quarterly deduction credit " * 10 + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        run(one, tmp_path / "m.joblib")
