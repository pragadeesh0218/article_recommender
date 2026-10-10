# Article recommender: PDFs -> CSV -> Multinomial Naive Bayes

Type a question, get the recommended knowledge-base article.

```
STEP 1  articles/*.pdf --open and read--> data/articles.csv   (label = file name, text = words in the PDF)
STEP 2  csv text -> tokenization -> n-grams -> vocabulary -> encoding (word counts) -> Multinomial Naive Bayes
STEP 3  question -> same text steps -> Naive Bayes -> recommended article
```

## Run
```bash
pip install -r requirements.txt
# put your 30-40 article PDFs in the  articles/  folder (sub-folders are fine)

python -m recommender.extract      # STEP 1: reads every PDF, writes data/articles.csv
python -m recommender.train        # STEP 2: trains, tests, saves models/model.joblib and models/vocabulary.csv
python -m recommender.predict "how do I add an employee"
python -m recommender.explain "how do I add an employee"     # shows every step for one question
pytest -q
```

## Step 1: PDF -> CSV (`extract.py`, `pdf_reader.py`)
One row per PDF. Open `data/articles.csv` in Excel to check it before training.

| column | meaning |
|---|---|
| `label` | the PDF file name without `.pdf` (this is the class) |
| `file` | path of the PDF inside `articles/` |
| `pages`, `words` | size check |
| `text` | all the text read from the PDF (features) |

While reading: words hyphenated at line ends are re-joined; page numbers, dates and URLs are dropped; lines repeated in 40%+ of
the PDFs (site header/footer) are removed. `extract` lists PDFs with no text (scanned images need OCR) or that cannot be opened.
Name PDFs clearly: the file name words are also used as training text.

## Step 2: the model (`nlp.py`, `model.py`)
1. **Tokenization** `tokenize`: lower-case, remove emails/SSN, split into words, drop stop-words, stem (refunds -> refund).
2. **N-grams** `make_ngrams`: single words plus word pairs (`add`, `employe`, `add_new`, `new_employe`).
3. **Vocabulary** `build_vocabulary`: every distinct n-gram gets a column number (`min_df`, `max_features` available). Saved in `models/vocabulary.csv`.
4. **Encoding** `encode`: each text becomes a row of word counts over the vocabulary (bag of words, sparse matrix).
5. **Multinomial Naive Bayes** (scikit-learn): learns, for each article, how likely each vocabulary term is; smoothing `alpha` is chosen by testing.

Training examples: each article is cut into sentence groups of ~25 words (about 10 to 20 per article), each labeled with the article's
file name, plus the file name itself. 30-40 articles = 30-40 classes.

## Step 3: predicting
The question goes through the same four text steps, then Naive Bayes ranks the articles. Output is the top 3 with scores.
If none of the question's words are in the vocabulary it recommends nothing. `explain` prints tokens, n-grams, which are in the
vocabulary, the count vector, the class log-scores, and which words pushed the model to the top article.

## Reading the numbers
`train` holds out sentence groups of every article, trains on the rest, and checks whether the model finds the article again
(top-1 / top-3). These are **proxy numbers**: test questions are pieces of the articles themselves, so real users will score lower.
The score is a Naive Bayes probability, usually over-confident: use it to rank. `most confused article pairs` shows overlapping articles.

## Limits
- Naive Bayes counts words and ignores order, so "add" vs "remove" an employee can look alike.
- Similar articles compete; the top-3 list helps.
- Any question with at least one known word gets an article (no out-of-scope detection beyond that).
