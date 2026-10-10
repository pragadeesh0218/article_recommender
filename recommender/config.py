import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", ROOT / "articles"))     # put the downloaded PDFs here
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
