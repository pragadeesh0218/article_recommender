"""Web UI + API:  uvicorn recommender.app:app --reload      then open http://localhost:8000

POST /predict {"query": "..."}  ->  decision "article" (with link) or "agent"
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import MODEL_PATH, STATIC_DIR
from .model import Recommender

state = {"model": None, "error": None, "n": 0, "article": 0, "agent": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = Recommender.load(MODEL_PATH)
    except Exception as e:           # keep the server up so the page can say what is missing
        state["error"] = f"Model not found or unreadable ({type(e).__name__}). Run: python -m recommender.extract, then python -m recommender.train"
    yield


app = FastAPI(title="Article recommender", version="2.0.0", lifespan=lifespan)


class Query(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


@app.post("/predict")
def predict(body: Query):
    if state["model"] is None:
        raise HTTPException(503, state["error"] or "model not loaded")
    out = state["model"].predict(body.query)
    state["n"] += 1
    state["article" if out["decision"] == "article" else "agent"] += 1
    return out


@app.get("/stats")
def stats():
    m = state["model"]
    return {"questions": state["n"], "article_shown": state["article"], "sent_to_agent": state["agent"],
            "articles": len(m.names) if m else 0, "articles_with_link": m.report["articles_with_link"] if m else 0,
            "agent_rule": m.t if m else None}


@app.get("/health")
def health():
    return {"status": "ok" if state["model"] else "not_ready", "error": state["error"]}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
