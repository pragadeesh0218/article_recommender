# Article recommender (Naive Bayes, from PDFs)

Type a question, get the recommended knowledge-base article. Nothing else.

- **Class (label)** = the PDF file name (without `.pdf`)
- **Features** = the words inside the PDF
- **Model** = Naive Bayes (chosen between Multinomial and Complement NB by testing)

## Use
```bash
pip install -r requirements.txt
# 1. put all the downloaded article PDFs in the  articles/  folder (sub-folders are fine)
python -m recommender.train                       # reads the PDFs, tests the model, saves models/model.joblib
python -m recommender.predict "how do I add an employee"
python -m recommender.predict                     # interactive: type questions one after another
pytest -q                                         # tests (make their own fake PDFs)
```
Output looks like:
```
Recommended article: How to add an employee
  1. How to add an employee  (score 0.81)
  2. Edit employee details   (score 0.10)
  3. Run payroll             (score 0.04)
```

## Name your PDFs well
The file name is both the class and a set of training words (counted 3x), so a clear name helps:
`How_to_add_an_employee.pdf` is better than `article_17.pdf`. Underscores are shown as spaces.

## What happens inside
1. **Read PDFs** (`pdf_reader.py`): extract text, join words split across lines, drop page numbers, dates and URLs.
   If the same line (site name, "Print", ...) appears in 40%+ of the PDFs it is removed as header/footer noise.
2. **Preprocess** (`preprocess.py`), the same for PDFs and questions: lower-case, remove stop-words, stem
   (refunds -> refund), keep word pairs.
3. **Training examples:** each article is cut into sentence groups of about 25 words, each labeled with the article's file name.
4. **Train** (`model.py`): TF-IDF + Naive Bayes. Training holds out some sentence groups of every article, checks if the model
   finds the right article from them, and keeps the best variant and smoothing.
5. **Predict:** same preprocessing on the question, then the top 3 articles with scores.
   If none of the question's words appear in any article, it recommends nothing instead of guessing.

## Reading the numbers
`train` prints top-1 / top-3 on held-out fragments. These are **proxy numbers**: the test questions are pieces of the articles
themselves, so real users (who phrase things differently) will score lower. The "score" is a Naive Bayes probability and is
usually over-confident; use it to rank, not as a true percentage. `most confused article pairs` shows which articles overlap.

## Limits
- PDFs that are scanned images have no text. `train` lists them; they need OCR first.
- Articles on similar topics will compete; the top-3 list helps.
- There is no "out of scope" handling beyond the no-matching-words case: any question with a known word gets an article.
