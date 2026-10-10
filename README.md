# Article recommender: PDFs -> CSV -> Multinomial Naive Bayes -> article link or human agent

Type a question. The app opens the right help article's link, or suggests a human agent when the question is out of scope
or the model is not confident.

```
STEP 1  articles/*.pdf --read--> data/articles.csv        (label = file name, link = web address, text = words in the PDF)
STEP 2  text -> tokenization -> n-grams -> vocabulary -> encoding (word counts) -> Multinomial Naive Bayes
STEP 3  question -> same steps -> article + link   OR   human agent
```

## Run
```bash
pip install -r requirements.txt
# put your 30-40 article PDFs in  articles/

python -m recommender.extract      # STEP 1: PDFs -> data/articles.csv
#  open data/articles.csv in Excel, fill the  link  column (https://...) for each article, save as CSV
python -m recommender.train        # STEP 2: trains, tunes the agent rule, saves models/model.joblib
uvicorn recommender.app:app --reload          # UI at http://localhost:8000
AGENT_URL=https://your-support-page uvicorn recommender.app:app    # where "Contact a support agent" goes

python -m recommender.predict "how do I add an employee"      # same decision in the terminal
python -m recommender.explain "how do I add an employee"      # every step for one question
pytest -q
```

## The `link` column
- Add it (or fill it) in `data/articles.csv`: one web address per article, starting with `http://` or `https://`.
  The header may be `link` or `url` in any capitalization. Anything else (for example `javascript:`) is ignored for safety.
- Running `extract` again **keeps the links you typed** (matched by the PDF file name). Rename a PDF and its link must be re-entered.
- `train` warns about articles without a link. Such an article can still be recommended; the UI says it has no link yet.

## When a human agent is suggested
| reason | when |
|---|---|
| `unclear_question` | no usable words (empty, only stop-words/symbols) |
| `asked_for_agent` | the user asks for a person ("talk to a human agent") |
| `out_of_scope` | none or too few of the question's words appear in any article (known-word ratio below the cut-off) |
| `low_confidence` | Naive Bayes' top probability is below the cut-off |

The two cut-offs are **tuned automatically** during `train`: short fragments of the held-out articles (plus versions with typos and
filler) must be answered correctly, while a list of out-of-scope questions must go to the agent. A wrong article, or an article for an
out-of-scope question, counts 3x worse than sending an in-scope question to the agent (`WRONG_COST` in `config.py`).
`train` prints how many in-scope questions are answered, how precise they are, and how many out-of-scope ones reach the agent.

**Improve it with your own examples:** put questions that should go to an agent (one per line) in `data/out_of_scope.txt`,
for example "how do I pay my electricity bill", then run `train` again. This is the most useful tuning you can do.

## The text steps (`nlp.py`)
1. **Tokenization**: lower-case, remove emails/SSN, split into words, drop stop-words, stem (refunds -> refund).
2. **N-grams**: single words plus word pairs (`add`, `employe`, `add_new`, `new_employe`).
3. **Vocabulary**: each distinct n-gram gets a column number (saved to `models/vocabulary.csv`).
4. **Encoding**: each text becomes a row of word counts.
5. **Multinomial Naive Bayes** learns how likely each term is for each article; smoothing `alpha` is chosen by testing.

Training examples: each article is cut into sentence groups of ~25 words, labeled with the file name, plus the file name itself.

## Reading the numbers
Test questions are pieces of the articles themselves, so the printed accuracy is a **proxy**; real users phrase things differently.
The probability is usually over-confident, so the known-word ratio does most of the out-of-scope work. Out-of-scope detection is
word overlap plus probability: a question that happens to reuse words from an article (for example "pay" or "report") can still
get an article. Add such cases to `data/out_of_scope.txt` and retrain.

## Files
`recommender/`: `extract.py` (PDF -> CSV), `pdf_reader.py`, `nlp.py`, `model.py`, `train.py`, `predict.py`, `explain.py`, `app.py`;
`static/index.html` (UI); `tests/test_recommender.py` (26 tests, they make their own fake PDFs).
