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
    assert list(rows[0]) == ["label", "link", "file", "pages", "words", "text"] and len(rows) == 8 and rows[0]["link"] == ""
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


# ================= new features: links, human agent, UI =================
def add_links(src, dst, mapping, header="link"):
    """Do what the user does in Excel: add a link column at the end of articles.csv."""
    with open(src, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    head = [h for h in rows[0] if h != "link"]
    keep = [i for i, h in enumerate(rows[0]) if h != "link"]
    out = [head + [header]] + [[r[i] for i in keep] + [mapping.get(r[0], "")] for r in rows[1:]]
    with open(dst, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(out)


LINKS = {f"How_to_{t}": f"https://kb.example.com/{t}" for t in TOPICS if t != "backup"}      # (no "backup" article: all 8 get a link)


@pytest.fixture(scope="module")
def linked_csv(csv_file, tmp_path_factory):
    dst = tmp_path_factory.mktemp("linked") / "articles.csv"
    add_links(csv_file, dst, {k: v for k, v in LINKS.items() if k != "How_to_tax"} | {"How_to_report": "javascript:alert(1)"}, header=" Link ")
    return dst


@pytest.fixture(scope="module")
def linked_model(linked_csv, tmp_path_factory):
    model, docs = run(linked_csv, tmp_path_factory.mktemp("m") / "m.joblib")
    return model, docs


def test_link_column_is_read_case_insensitively_and_made_safe(linked_csv):
    docs = extract.load_csv(linked_csv)
    assert docs["How_to_payroll"]["link"] == "https://kb.example.com/payroll"
    assert docs["How_to_tax"]["link"] == "" and docs["How_to_tax"]["link_problem"] == ""
    assert docs["How_to_report"]["link"] == "" and docs["How_to_report"]["link_problem"] == "javascript:alert(1)"
    assert extract.clean_link("  http://a.b/c  ") == "http://a.b/c" and extract.clean_link("file:///etc/passwd") == ""
    assert extract.load_csv(linked_csv.parent / "articles.csv") is not None


def test_extract_again_keeps_links_you_typed(folder, linked_csv, tmp_path):
    import shutil
    csv_path = tmp_path / "articles.csv"
    shutil.copy(linked_csv, csv_path)
    docs, rep = extract.run(folder, csv_path)                       # re-run step 1 on the edited CSV
    kept = extract.load_csv(csv_path)
    assert kept["How_to_payroll"]["link"] == "https://kb.example.com/payroll"
    assert rep["links_kept"] == 6 and rep["links_lost"] == []        # 8 articles - tax (no link) - report (unsafe link dropped)
    assert [r["label"] for r in csv.DictReader(open(csv_path, encoding="utf-8-sig"))][0] == "How_to_deposit"
    assert list(next(csv.DictReader(open(csv_path, encoding="utf-8-sig")))) == ["label", "link", "file", "pages", "words", "text"]


def test_confident_question_returns_article_with_link(linked_model):
    model, _ = linked_model
    res = model.predict("how do i reset my password login credential")
    assert res["decision"] == "article" and res["reason"] == "confident_match" and res["agent_url"] is None
    assert res["article"]["label"] == "How_to_password" and res["article"]["link"] == "https://kb.example.com/password"
    assert res["recommended"] == "How_to_password" and 0 < res["confidence"] <= 1


def test_article_without_link_is_still_recommended_with_a_note(linked_model):
    model, _ = linked_model
    res = model.predict("tax filing bracket quarterly deduction credit")
    assert res["decision"] == "article" and res["article"]["label"] == "How_to_tax"
    assert res["article"]["link"] is None and "no link" in res["note"].lower()


@pytest.mark.parametrize("q,reason", [
    ("what is the weather today", "out_of_scope"), ("tell me a joke", "out_of_scope"), ("asdf qwer zxcv", "out_of_scope"),
    ("hello", "out_of_scope"), ("i want to talk to a human agent", "asked_for_agent"), ("???", "unclear_question"), ("", "unclear_question")])
def test_out_of_scope_goes_to_agent(linked_model, q, reason):
    model, _ = linked_model
    res = model.predict(q)
    assert res["decision"] == "agent" and res["reason"] == reason and res["article"] is None and res["recommended"] is None
    assert res["agent_url"].startswith("http") and res["note"]


def test_low_confidence_branches_go_to_agent_and_keep_closest(linked_model):
    model, _ = linked_model
    saved = dict(model.t)
    try:
        model.t = {**saved, "min_probability": 1.1}                         # nothing can be confident enough
        res = model.predict("how do i reset my password login credential")
        assert res["decision"] == "agent" and res["reason"] == "low_confidence" and res["article"] is None
        assert res["matches"][0]["label"] == "How_to_password"              # the closest article is still reported
        model.t = {**saved, "min_probability": 0.0, "min_known_ratio": 1.1}  # words not known well enough
        assert model.predict("reset password and zzz qqq")["reason"] == "out_of_scope"
    finally:
        model.t = saved


def test_agent_rule_is_tuned_and_reported(linked_model):
    model, _ = linked_model
    t, r = model.t, model.report
    assert 0 < t["min_probability"] <= 0.999 and 0 <= t["min_known_ratio"] <= 0.9
    assert t["out_of_scope_sent_to_agent"] >= 0.9 and t["in_scope_answered"] >= 0.7 and t["precision_when_answered"] >= 0.9
    assert r["articles_with_link"] == 6 and r["agent_rule"] == t and "casual" in r["held_out"]


def test_explain_shows_the_decision(linked_model):
    model, _ = linked_model
    e = model.explain("reset my password login")
    assert e["decision"] == "article" and e["reason"] == "confident_match" and e["known_ratio"] > 0.5
    assert model.explain("what is the weather")["decision"] == "agent"


def test_web_app(linked_model, monkeypatch):
    from fastapi.testclient import TestClient
    from recommender import app as app_mod
    model, _ = linked_model
    monkeypatch.setattr(app_mod.Recommender, "load", classmethod(lambda cls, path=None: model))
    app_mod.state.update(model=None, error=None, n=0, article=0, agent=0)
    with TestClient(app_mod.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        a = c.post("/predict", json={"query": "export the report spreadsheet column"}).json()
        assert a["decision"] == "article" and a["article"]["label"] == "How_to_report" and a["article"]["link"] is None   # unsafe link dropped
        b = c.post("/predict", json={"query": "reset password login credential"}).json()
        assert b["article"]["link"] == "https://kb.example.com/password"
        g = c.post("/predict", json={"query": "what is the weather"}).json()
        assert g["decision"] == "agent" and g["agent_url"]
        assert c.post("/predict", json={"query": ""}).status_code == 422
        s = c.get("/stats").json()
        assert s["questions"] == 3 and s["article_shown"] == 2 and s["sent_to_agent"] == 1 and s["articles_with_link"] == 6
        page = c.get("/")
        assert page.status_code == 200 and "Find your answer" in page.text


def test_web_app_without_model_explains(monkeypatch):
    from fastapi.testclient import TestClient
    from recommender import app as app_mod
    def boom(cls, path=None):
        raise FileNotFoundError("no model")
    monkeypatch.setattr(app_mod.Recommender, "load", classmethod(boom))
    app_mod.state.update(model=None, error=None)
    with TestClient(app_mod.app) as c:
        assert c.get("/health").json()["status"] == "not_ready"
        r = c.post("/predict", json={"query": "hello"})
        assert r.status_code == 503 and "recommender.train" in r.json()["detail"]


def test_your_own_out_of_scope_questions_are_used_for_tuning(tmp_path, monkeypatch):
    from recommender import model as model_mod
    f = tmp_path / "out_of_scope.txt"
    f.write_text("how do I pay my electricity bill\n\n  can I book a hotel  \n", encoding="utf-8")
    monkeypatch.setattr(model_mod, "OOS_FILE", f)
    qs = model_mod._oos()
    assert "how do I pay my electricity bill" in qs and "can I book a hotel" in qs and "hello" in qs
    monkeypatch.setattr(model_mod, "OOS_FILE", tmp_path / "missing.txt")
    assert "can I book a hotel" not in model_mod._oos()
