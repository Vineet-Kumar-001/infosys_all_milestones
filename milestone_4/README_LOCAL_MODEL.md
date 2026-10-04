# Production Local Sentiment Model

This upgrade removes Gemini from the article-level sentiment path and replaces it with a local fine-tuned transformer classifier. A discriminative encoder is a better fit for sentiment: it is smaller, deterministic, batchable, and easier to validate than a generative LLM.

## Architecture
Historical CSV reports -> validation/deduplication -> historical teacher labels -> DistilBERT fine-tuning -> local model artifact -> batch inference -> label + continuous score + confidence.

Continuous score:
score = P(POSITIVE) - P(NEGATIVE)

## Training

From milestone_4:

pip install -r requirements-training.txt
python training/train_sentiment.py --data-dir Datasets --output-dir models/sentiment-distilbert

The trainer scans CSV reports, normalizes labels, removes duplicates, uses a stratified 70/15/15 split, applies class-weighted cross entropy, reports accuracy/macro-F1/weighted-F1 and teacher-score MAE, and stores model metadata plus a SHA-256 hash of the exact training records.

Important: historical Gemini labels are weak/teacher labels, not independent ground truth. A production release should maintain a manually reviewed gold set.

## Local inference

from src.local_sentiment import SentimentEngine
engine = SentimentEngine()
result = engine.predict("The company reported strong earnings and raised guidance.")
print(result.label, result.score, result.confidence)

## API

uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints: GET /health, POST /predict, POST /predict/batch.

## ONNX

After training, use training/export_onnx.py for an ONNX Runtime deployment artifact. Only claim NPU acceleration after installing the matching provider and benchmarking it on the target device.