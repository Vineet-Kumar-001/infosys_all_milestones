from __future__ import annotations

import argparse
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained sentiment model to ONNX Runtime format.")
    parser.add_argument("--model-dir", default="models/sentiment-distilbert")
    parser.add_argument("--output-dir", default="models/sentiment-distilbert-onnx")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"ONNX export written to {output_dir}")

if __name__ == "__main__":
    main()
