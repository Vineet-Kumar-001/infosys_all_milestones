from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

LABELS = ("NEGATIVE", "NEUTRAL", "POSITIVE")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune the local news sentiment classifier")
    p.add_argument("--data-dir", default="Datasets")
    p.add_argument("--output-dir", default="models/sentiment-distilbert")
    p.add_argument("--base-model", default="distilbert/distilbert-base-uncased")
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def normalize_label(value):
    if pd.isna(value):
        return None
    label = str(value).strip().upper()
    label = {"POS": "POSITIVE", "NEG": "NEGATIVE", "NEU": "NEUTRAL"}.get(label, label)
    return label if label in LABEL_TO_ID else None

def load_dataset(data_dir):
    frames = []
    for path in sorted(Path(data_dir).rglob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] skipping {path}: {exc}")
            continue
        df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]
        if "text" not in df.columns:
            title = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
            desc = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
            df["text"] = (title + ". " + desc).str.strip()
        label_col = "sentiment_label" if "sentiment_label" in df.columns else "predicted_sentiment"
        if label_col not in df.columns:
            continue
        df["label"] = df[label_col].map(normalize_label)
        df["teacher_score"] = pd.to_numeric(df.get("sentiment_score", np.nan), errors="coerce")
        df["text"] = df["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        df = df[df["text"].str.len().between(15, 5000)]
        df = df[df["label"].notna()]
        frames.append(df[["text", "label", "teacher_score"]])
    if not frames:
        raise RuntimeError(f"No usable labeled CSV files found under {data_dir}")
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    data["label_id"] = data["label"].map(LABEL_TO_ID).astype(int)
    data["teacher_score"] = data["teacher_score"].clip(-1, 1)
    return data

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def make_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"] = "epoch"
    if "save_strategy" in sig:
        kwargs["save_strategy"] = "epoch"
    return TrainingArguments(**kwargs)

def evaluate_scores(trainer, test_dataset, teacher_scores):
    prediction = trainer.predict(test_dataset)
    probs = torch.softmax(torch.tensor(prediction.predictions), dim=-1).numpy()
    pred_ids = probs.argmax(axis=1)
    model_scores = probs @ np.array([-1.0, 0.0, 1.0])
    valid = np.isfinite(teacher_scores)
    return {
        "accuracy": float(accuracy_score(prediction.label_ids, pred_ids)),
        "macro_f1": float(f1_score(prediction.label_ids, pred_ids, average="macro")),
        "weighted_f1": float(f1_score(prediction.label_ids, pred_ids, average="weighted")),
        "teacher_score_mae": float(mean_absolute_error(teacher_scores[valid], model_scores[valid])) if valid.any() else None,
        "classification_report": classification_report(prediction.label_ids, pred_ids, target_names=list(LABELS), output_dict=True, zero_division=0),
    }

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); set_seed(args.seed)
    data = load_dataset(args.data_dir)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    counts = data["label"].value_counts().reindex(LABELS, fill_value=0)
    if (counts < 5).any():
        raise RuntimeError(f"Too few examples in one or more classes: {counts.to_dict()}")
    train_df, temp_df = train_test_split(data, test_size=0.30, random_state=args.seed, stratify=data["label_id"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=args.seed, stratify=temp_df["label_id"])
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=3, id2label=ID_TO_LABEL, label2id=LABEL_TO_ID)
    train_ds = NewsDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer, args.max_length)
    val_ds = NewsDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer, args.max_length)
    test_ds = NewsDataset(test_df["text"].tolist(), test_df["label_id"].tolist(), tokenizer, args.max_length)
    train_counts = train_df["label_id"].value_counts().reindex(range(3), fill_value=0).to_numpy()
    weights = train_counts.sum() / np.maximum(train_counts, 1); weights = weights / weights.mean()
    train_args = make_training_args(output_dir=str(output_dir / "checkpoints"), learning_rate=args.learning_rate, num_train_epochs=args.epochs, per_device_train_batch_size=args.train_batch_size, per_device_eval_batch_size=args.eval_batch_size, weight_decay=0.01, logging_steps=25, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2, report_to="none", fp16=bool(torch.cuda.is_available()), seed=args.seed)
    trainer = WeightedTrainer(class_weights=torch.tensor(weights, dtype=torch.float32), model=model, args=train_args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
    trainer.train()
    metrics = evaluate_scores(trainer, test_ds, test_df["teacher_score"].to_numpy(dtype=float))
    trainer.save_model(output_dir); tokenizer.save_pretrained(output_dir)
    canonical = data[["text", "label", "teacher_score"]].sort_values("text").to_csv(index=False).encode("utf-8")
    metadata = {"model_version": f"sentiment-distilbert-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}", "base_model": args.base_model, "labels": list(LABELS), "max_length": args.max_length, "dataset_rows": int(len(data)), "train_rows": int(len(train_df)), "validation_rows": int(len(val_df)), "test_rows": int(len(test_df)), "class_distribution": counts.to_dict(), "dataset_sha256": hashlib.sha256(canonical).hexdigest(), "seed": args.seed, "metrics": metrics, "created_at_utc": datetime.now(timezone.utc).isoformat(), "label_provenance": "Historical Gemini sentiment labels/scores are weak teacher labels, not independent ground truth."}
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, indent=2))

if __name__ == "__main__":
    main()
