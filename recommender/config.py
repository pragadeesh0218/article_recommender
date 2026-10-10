import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", ROOT / "articles"))     # the downloaded PDFs
CSV_PATH = Path(os.getenv("CSV_PATH", ROOT / "data" / "articles.csv"))  # step 1 output: label + text, one row per PDF
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
VOCAB_PATH = ROOT / "models" / "vocabulary.csv"
OOS_FILE = Path(os.getenv("OOS_FILE", ROOT / "data" / "out_of_scope.txt"))   # optional: your own out-of-scope questions, one per line
STATIC_DIR = ROOT / "static"

# Where "talk to a human agent" sends the user. Set the real support / contact page:  AGENT_URL=https://... uvicorn ...
AGENT_URL = os.getenv("AGENT_URL", "https://www.pistus.com/")
# Showing a wrong article (or answering a question that is out of scope) is this many times worse than handing over to an agent.
WRONG_COST = 3.0
