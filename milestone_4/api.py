from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.local_sentiment import SentimentEngine, SentimentModelError

app = FastAPI(title="Local News Sentiment API", version="1.0.0")
try:
    ENGINE = SentimentEngine()
    LOAD_ERROR = None
except SentimentModelError as exc:
    ENGINE = None
    LOAD_ERROR = str(exc)

class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)

class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=128)

@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    return response

def require_engine() -> SentimentEngine:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail=LOAD_ERROR or "Model unavailable")
    return ENGINE

@app.get("/health")
def health() -> dict:
    return {"status": "ok" if ENGINE else "degraded", "model_loaded": ENGINE is not None,
            "model_version": ENGINE.model_version if ENGINE else None,
            "device": str(ENGINE.device) if ENGINE else None}

@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    result = require_engine().predict(payload.text)
    return {"label": result.label, "score": result.score, "confidence": result.confidence,
            "probabilities": result.probabilities, "latency_ms": result.latency_ms,
            "model_version": result.model_version}

@app.post("/predict/batch")
def predict_batch(payload: BatchRequest) -> dict:
    results = require_engine().predict_batch(payload.texts)
    return {"count": len(results), "predictions": [
        {"label": r.label, "score": r.score, "confidence": r.confidence,
         "probabilities": r.probabilities, "latency_ms": r.latency_ms,
         "model_version": r.model_version} for r in results
    ]}
