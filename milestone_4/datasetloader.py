from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from newsdataapi import NewsDataApiClient

from src.local_sentiment import SentimentEngine, SentimentModelError


def setup_environment():
    load_dotenv()
    newsdata_api_key = os.getenv("NEWSDATA_API_KEY")
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not newsdata_api_key:
        st.error("Missing NEWSDATA_API_KEY in the environment.")
        st.stop()
    try:
        engine = SentimentEngine()
    except SentimentModelError as exc:
        st.error(str(exc))
        st.stop()
    return engine, newsdata_api_key, slack_webhook_url


def fetch_news_data(api_key: str, query: str, max_results: int = 50) -> list[dict]:
    articles = []
    try:
        api_client = NewsDataApiClient(apikey=api_key)
        response = api_client.latest_api(q=query, language="en")
        for item in response.get("results", []):
            title = (item.get("title") or "").strip()
            description = (item.get("description") or "").strip()
            if not title and not description:
                continue
            articles.append({
                "source": item.get("source_id") or "Newsdata.io",
                "text": f"{title}. {description}".strip(". "),
                "url": item.get("link"),
            })
    except Exception as exc:
        st.error(f"News API error: {exc}")
        return []
    return articles[:max_results]


def send_slack(webhook_url: str | None, message: str) -> None:
    if not webhook_url:
        st.warning("Slack webhook is not configured.")
        return
    try:
        response = requests.post(webhook_url, json={"text": message}, timeout=10)
        response.raise_for_status()
        st.success("Slack notification sent.")
    except requests.RequestException as exc:
        st.error(f"Slack error: {exc}")


def run_dataset_loader():
    st.subheader("Local Sentiment Inference")
    st.caption("All article sentiment scores are produced locally; no Gemini API call is used.")

    engine, news_api_key, slack_url = setup_environment()

    col1, col2 = st.columns(2)
    with col1:
        selected_date: date = st.date_input("Report date")
    with col2:
        topic = st.text_input("Topic", "AI OR artificial intelligence OR technology")

    max_results = st.slider("Articles", min_value=10, max_value=100, value=50, step=10)
    batch_size = st.slider("Inference batch size", min_value=1, max_value=64, value=16, step=1)

    run = st.button("Run local sentiment analysis", type="primary")
    if not run:
        return

    articles = fetch_news_data(news_api_key, topic, max_results)
    if not articles:
        st.warning("No articles found.")
        return

    df = pd.DataFrame(articles)
    progress = st.progress(0.0)
    predictions = []

    for start in range(0, len(df), batch_size):
        batch = df["text"].iloc[start:start + batch_size].tolist()
        results = engine.predict_batch(batch, batch_size=batch_size)
        predictions.extend(results)
        progress.progress(min(1.0, (start + len(batch)) / len(df)))

    df["predicted_sentiment"] = [r.label.title() for r in predictions]
    df["sentiment_score"] = [r.score for r in predictions]
    df["sentiment_confidence"] = [r.confidence for r in predictions]
    df["negative_probability"] = [r.probabilities.get("NEGATIVE", 0.0) for r in predictions]
    df["neutral_probability"] = [r.probabilities.get("NEUTRAL", 0.0) for r in predictions]
    df["positive_probability"] = [r.probabilities.get("POSITIVE", 0.0) for r in predictions]
    df["model_version"] = [r.model_version for r in predictions]
    df["inference_latency_ms"] = [r.latency_ms for r in predictions]
    df["report_date"] = selected_date.isoformat()

    output_dir = Path("Datasets")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"news_sentiment_report_{selected_date.isoformat()}.csv"
    df.to_csv(output_file, index=False)

    st.success(f"Completed {len(df)} articles with model {engine.model_version}.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Articles", len(df))
    m2.metric("Positive", int((df["sentiment_score"] > 0).sum()))
    m3.metric("Negative", int((df["sentiment_score"] < 0).sum()))
    m4.metric("Avg score", f"{df['sentiment_score'].mean():.3f}")

    st.dataframe(
        df[["source", "text", "predicted_sentiment", "sentiment_score", "sentiment_confidence", "url"]],
        use_container_width=True,
    )

    top_negative = df.nsmallest(3, "sentiment_score")
    top_positive = df.nlargest(3, "sentiment_score")
    summary = (
        f"Local sentiment report {selected_date.isoformat()}\\n"
        f"Articles: {len(df)}\\n"
        f"Average score: {df['sentiment_score'].mean():.3f}\\n"
        f"Most negative: {top_negative.iloc[0]['sentiment_score']:.3f}\\n"
        f"Most positive: {top_positive.iloc[0]['sentiment_score']:.3f}"
    )

    if st.button("Send summary to Slack"):
        send_slack(slack_url, summary)
