from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from prophet import Prophet
from wordcloud import WordCloud

from src.local_sentiment import SentimentModelError, SentimentEngine


st.set_page_config(page_title="Local Sentiment Intelligence", layout="wide")


def setup_env():
    load_dotenv()
    slack_url = os.getenv("SLACK_WEBHOOK_URL")
    try:
        engine = SentimentEngine()
    except SentimentModelError as exc:
        st.error(str(exc))
        st.stop()
    return slack_url, engine


def load_sentiment_data():
    dataset_dir = Path(__file__).parent / "Datasets"
    csv_files = sorted(glob.glob(str(dataset_dir / "news_sentiment_report_*.csv")))
    if not csv_files:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    daily, counts, all_frames = [], [], []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "sentiment_score" not in df.columns:
            continue
        stamp = Path(file_path).stem.split("report_", 1)[-1]
        try:
            report_date = pd.to_datetime(stamp)
        except Exception:
            continue
        df["date"] = report_date
        score = pd.to_numeric(df["sentiment_score"], errors="coerce")
        df["sentiment_score"] = score
        daily.append({"date": report_date, "mean_score": score.mean()})
        counts.append({
            "date": report_date,
            "positive": int((score > 0).sum()),
            "negative": int((score < 0).sum()),
            "neutral": int((score == 0).sum()),
        })
        all_frames.append(df.dropna(subset=["sentiment_score"]))

    if not all_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return (
        pd.DataFrame(daily).sort_values("date"),
        pd.DataFrame(counts).sort_values("date"),
        pd.concat(all_frames, ignore_index=True),
    )


def prophet_forecast(daily_df):
    if len(daily_df) < 3:
        return pd.DataFrame()
    prophet_df = daily_df.rename(columns={"date": "ds", "mean_score": "y"})
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    model.fit(prophet_df)
    return model.predict(model.make_future_dataframe(periods=5))


def send_slack(webhook_url, message):
    if not webhook_url:
        st.warning("Slack webhook is not configured.")
        return
    try:
        response = requests.post(webhook_url, json={"text": message}, timeout=10)
        response.raise_for_status()
        st.success("Slack message sent.")
    except requests.RequestException as exc:
        st.error(f"Slack error: {exc}")


def deterministic_summary(daily_df, combined_df):
    avg = float(combined_df["sentiment_score"].mean())
    pos = int((combined_df["sentiment_score"] > 0).sum())
    neg = int((combined_df["sentiment_score"] < 0).sum())
    if len(daily_df) >= 2:
        change = float(daily_df.iloc[-1]["mean_score"] - daily_df.iloc[0]["mean_score"])
        direction = "improving" if change > 0.03 else "deteriorating" if change < -0.03 else "stable"
    else:
        change, direction = 0.0, "insufficient history"
    mood = "positive" if avg > 0.10 else "negative" if avg < -0.10 else "neutral"
    return (
        f"The selected window has {len(combined_df):,} articles with an average sentiment score "
        f"of {avg:.3f}, indicating an overall {mood} narrative. "
        f"There are {pos:,} positive and {neg:,} negative articles. "
        f"The mean sentiment changed by {change:+.3f} across the window, so the trend is {direction}."
    )


def run_data_visualization():
    slack_url, engine = setup_env()
    st.title("Local Sentiment Intelligence Dashboard")
    st.caption(f"Model: {engine.model_version} • Device: {engine.device} • No Gemini dependency")

    daily_df, sentiment_df, combined_df = load_sentiment_data()
    if daily_df.empty:
        st.warning("No sentiment reports found in Datasets/. Run local inference first.")
        return

    min_date = daily_df["date"].min().to_pydatetime()
    max_date = daily_df["date"].max().to_pydatetime()
    start_date, end_date = st.slider(
        "Analysis window",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
    )
    daily_df = daily_df[(daily_df["date"] >= start_date) & (daily_df["date"] <= end_date)]
    sentiment_df = sentiment_df[(sentiment_df["date"] >= start_date) & (sentiment_df["date"] <= end_date)]
    combined_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= end_date)]

    total = len(combined_df)
    avg_score = float(combined_df["sentiment_score"].mean()) if total else 0.0
    pos = int((combined_df["sentiment_score"] > 0).sum())
    neg = int((combined_df["sentiment_score"] < 0).sum())
    neu = int((combined_df["sentiment_score"] == 0).sum())
    confidence = float(combined_df["sentiment_confidence"].mean()) if "sentiment_confidence" in combined_df else float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Articles", f"{total:,}")
    c2.metric("Average score", f"{avg_score:.3f}")
    c3.metric("Positive", f"{pos:,}")
    c4.metric("Negative", f"{neg:,}")
    c5.metric("Avg confidence", f"{confidence:.3f}" if confidence == confidence else "n/a")

    forecast = prophet_forecast(daily_df)
    left, right = st.columns([2.5, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_df["date"], y=daily_df["mean_score"], mode="lines+markers", name="Actual"))
        if not forecast.empty:
            past = forecast[forecast["ds"] <= daily_df["date"].max()]
            future = forecast[forecast["ds"] > daily_df["date"].max()]
            fig.add_trace(go.Scatter(x=past["ds"], y=past["yhat"], mode="lines", name="Prophet fit"))
            fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat"], mode="lines+markers", name="5-day forecast"))
        fig.update_layout(template="plotly_dark", height=360, yaxis_title="Sentiment score")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        pie = go.Figure(data=[go.Pie(labels=["Positive", "Negative", "Neutral"], values=[pos, neg, neu], hole=0.58)])
        pie.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(pie, use_container_width=True)

    st.subheader("Executive summary")
    summary = deterministic_summary(daily_df, combined_df)
    st.info(summary)

    st.subheader("Sentiment heatmap by date")
    heat = sentiment_df.melt(id_vars=["date"], value_vars=["positive", "negative", "neutral"], var_name="class", value_name="articles")
    heatmap = px.density_heatmap(heat, x="date", y="class", z="articles", histfunc="sum", color_continuous_scale="Blues")
    st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("Positive vs negative keywords")
    col1, col2 = st.columns(2)
    with col1:
        positive_text = " ".join(combined_df.loc[combined_df["sentiment_score"] > 0, "text"].astype(str))
        if positive_text.strip():
            wc = WordCloud(width=700, height=320, background_color="white", colormap="Greens").generate(positive_text)
            st.image(wc.to_array(), caption="Positive keywords", use_container_width=True)
    with col2:
        negative_text = " ".join(combined_df.loc[combined_df["sentiment_score"] < 0, "text"].astype(str))
        if negative_text.strip():
            wc = WordCloud(width=700, height=320, background_color="white", colormap="Reds").generate(negative_text)
            st.image(wc.to_array(), caption="Negative keywords", use_container_width=True)

    st.subheader("Lowest-confidence predictions")
    if "sentiment_confidence" in combined_df:
        st.dataframe(
            combined_df.nsmallest(20, "sentiment_confidence")[["date", "text", "predicted_sentiment", "sentiment_score", "sentiment_confidence"]],
            use_container_width=True,
        )

    if st.button("Send dashboard summary to Slack"):
        send_slack(slack_url, summary + f"\\nModel: {engine.model_version}")


if __name__ == "__main__":
    run_data_visualization()
