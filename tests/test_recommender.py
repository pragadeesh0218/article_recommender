"""Tests use PDFs generated here (reportlab) with made-up text; they are never used for the real model."""
import random
import warnings

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

warnings.filterwarnings("ignore")

from recommender import pdf_reader
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


def test_read_labels_text_and_boilerplate(folder):
    docs, rep = pdf_reader.read_folder(folder)
    assert set(docs) == {f"How_to_{t}" for t in TOPICS}                      # label = file name
    assert docs["How_to_payroll"]["name"] == "How to payroll"
    assert "payroll" in docs["How_to_payroll"]["text"]
    assert "Pistus Knowledge Base" not in docs["How_to_payroll"]["text"] and "pistus.com" not in docs["How_to_payroll"]["text"]
    assert rep["n_pdfs"] == 8 and not rep["no_text"] and rep["boilerplate_removed"]


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


def test_train_predict_and_roundtrip(folder, tmp_path):
    model, info = run(folder, tmp_path / "m.joblib")
    assert model.report["n_articles"] == 8 and model.report["held_out"]["span"]["top1"] > 0.8
    for t, words in TOPICS.items():
        res = model.predict("how do i " + " ".join(words.split()[:3]))
        assert res["recommended"] == f"How_to_{t}" and len(res["matches"]) == 3
        assert res["matches"][0]["score"] >= res["matches"][1]["score"]
    again = Recommender.load(tmp_path / "m.joblib")
    assert again.predict("reset password login")["matches"] == model.predict("reset password login")["matches"]


def test_unknown_words_recommend_nothing(folder, tmp_path):
    model, _ = run(folder, tmp_path / "m.joblib")
    for q in ["xyzzy plugh", "", "the of and"]:
        res = model.predict(q)
        assert res["recommended"] is None and res["matches"] == [] and res["note"]


def test_needs_two_readable_pdfs(tmp_path):
    make_pdf(tmp_path / "only.pdf", body_for("tax"))
    with pytest.raises(SystemExit):
        run(tmp_path, tmp_path / "m.joblib")
