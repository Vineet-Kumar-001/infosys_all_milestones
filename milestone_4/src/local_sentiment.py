from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ("NEGATIVE", "NEUTRAL", "POSITIVE")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
SCORE_AXIS = torch.tensor([-1.0, 0.0, 1.0])

@dataclass(frozen=True)
class SentimentPrediction:
    label: str
    score: float
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    model_version: str

class SentimentModelError(RuntimeError):
    pass

class SentimentEngine:
    """Local, batch-first sentiment inference with no remote LLM dependency."""
    def __init__(self, model_dir: str | Path | None = None, device: str | None = None, max_length: int = 256) -> None:
        self.model_dir = Path(model_dir or os.getenv("LOCAL_SENTIMENT_MODEL_DIR", "models/sentiment-distilbert"))
        self.max_length = max(32, min(int(max_length), 512))
        self.device = self._select_device(device)
        if not self.model_dir.exists():
            raise SentimentModelError(f"Local model not found at '{self.model_dir}'. Run: python training/train_sentiment.py")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, local_files_only=True)
        except Exception as exc:
            raise SentimentModelError(f"Failed to load local sentiment model from '{self.model_dir}': {exc}") from exc
        self.model.to(self.device)
        self.model.eval()
        self.model_version = self._read_model_version()

    @staticmethod
    def _select_device(requested: str | None) -> torch.device:
        requested = requested or os.getenv("SENTIMENT_DEVICE", "auto")
        if requested != "auto":
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _read_model_version(self) -> str:
        metadata = self.model_dir / "model_metadata.json"
        if metadata.exists():
            try:
                payload = json.loads(metadata.read_text(encoding="utf-8"))
                return str(payload.get("model_version") or payload.get("run_id") or "local-unknown")
            except (OSError, json.JSONDecodeError):
                pass
        return self.model_dir.name

    @torch.inference_mode()
    def predict_batch(self, texts: Iterable[str], batch_size: int = 16) -> list[SentimentPrediction]:
        values = ["" if text is None else str(text) for text in texts]
        if not values:
            return []
        batch_size = max(1, min(int(batch_size), 128))
        results: list[SentimentPrediction] = []
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            t0 = time.perf_counter()
            encoded = self.tokenizer(batch, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            probabilities = torch.softmax(self.model(**encoded).logits, dim=-1)
            scores = (probabilities * SCORE_AXIS.to(probabilities.device)).sum(dim=-1)
            confidence, indices = probabilities.max(dim=-1)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            for row_idx in range(len(batch)):
                idx = int(indices[row_idx].item())
                probs = probabilities[row_idx].detach().cpu().tolist()
                results.append(SentimentPrediction(
                    label=ID_TO_LABEL.get(idx, str(idx)),
                    score=float(scores[row_idx].item()),
                    confidence=float(confidence[row_idx].item()),
                    probabilities={LABELS[pos]: float(probs[pos]) for pos in range(min(len(probs), len(LABELS)))},
                    latency_ms=elapsed_ms / len(batch),
                    model_version=self.model_version,
                ))
        return results

    def predict(self, text: str) -> SentimentPrediction:
        return self.predict_batch([text], batch_size=1)[0]
