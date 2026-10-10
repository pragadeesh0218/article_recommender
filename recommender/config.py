import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", ROOT / "articles"))     # the downloaded PDFs
CSV_PATH = Path(os.getenv("CSV_PATH", ROOT / "data" / "articles.csv"))  # step 1 output: label + text, one row per PDF
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
VOCAB_PATH = ROOT / "models" / "vocabulary.csv"
