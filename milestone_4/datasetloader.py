# --- 1. IMPORT NECESSARY LIBRARIES ---
import os
import sys
import json
import time
import requests
import pandas as pd
import nltk
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from newsdataapi import NewsDataApiClient
from nltk.corpus import stopwords

# --- 2. SETUP AND CONFIGURATION ---
def setup_environment():
    """Loads environment variables and configures APIs."""
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    newsdata_api_key = os.getenv("NEWSDATA_API_KEY")
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")

    if not all([gemini_api_key, newsdata_api_key, slack_webhook_url]):
        st.error("❌ Missing API keys in `.env` file (GEMINI_API_KEY, NEWSDATA_API_KEY, SLACK_WEBHOOK_URL).")
        st.stop()

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("models/gemini-2.5-flash-preview-09-2025")

    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    return model, newsdata_api_key, slack_webhook_url


# --- 3. FETCH NEWS DATA ---
def fetch_news_data(api_key, query, max_results=50):
    st.write(f"🔍 Fetching up to {max_results} news articles for **{query}**...")
    all_articles = []

    try:
        api_client = NewsDataApiClient(apikey=api_key)
        response = api_client.latest_api(q=query, language='en')
        if 'results' not in response or not response['results']:
            st.warning("⚠️ No articles found.")
            return []

        for a in response.get('results', []):
            if a.get('title') and a.get('description'):
                all_articles.append({
                    "source": "Newsdata.io",
                    "text": f"{a.get('title', '')}. {a.get('description', '')}",
                    "url": a.get('link')
                })
        return all_articles[:max_results]
    except Exception as e:
        st.error(f"❌ News API Error: {e}")
        return []


# --- 4. GEMINI SENTIMENT ANALYSIS ---
def get_sentiment_with_gemini(model, text):
    if not text or len(text.strip()) < 15:
        return "Neutral", 0.0

    prompt = f"""
    Analyze the sentiment of the following text.
    Classify as 'Positive', 'Negative', or 'Neutral'.
    Give a sentiment score between -1.0 and 1.0.
    Return ONLY JSON with keys: sentiment_label, sentiment_score.

    Text: "{text}"
    """

    for _ in range(3):
        try:
            response = model.generate_content(prompt)
            cleaned = response.text.strip().replace("```json", "").replace("```", "").replace("'", '"')
            result = json.loads(cleaned)
            return result.get("sentiment_label", "Neutral"), float(result.get("sentiment_score", 0.0))
        except google_exceptions.ResourceExhausted:
            st.warning("⚠️ Gemini rate limit reached. Retrying...")
            time.sleep(5)
        except Exception:
            return "Neutral", 0.0
    return "Neutral", 0.0


# --- 5. SLACK ALERTS ---
def send_slack_summary(webhook_url, df):
    """Send a short summary alert of total sentiment counts."""
    if df.empty:
        st.warning("No data available for summary alert.")
        return

    counts = df['predicted_sentiment'].value_counts()
    msg = (
        f"📢 *Daily Sentiment Summary Alert*\n\n"
        f"📰 Total Articles: {len(df)}\n"
        f"✅ Positive: {counts.get('Positive', 0)}\n"
        f"❌ Negative: {counts.get('Negative', 0)}\n"
        f"⚪ Neutral: {counts.get('Neutral', 0)}\n"
    )

    try:
        requests.post(webhook_url, json={"text": msg})
        st.success("✅ Summary alert successfully posted to Slack!")
    except Exception as e:
        st.error(f"❌ Slack Summary Error: {e}")


def send_slack_alert(webhook_url, df):
    """Send detailed sentiment report to Slack."""
    if df.empty:
        st.warning("No data to send to Slack.")
        return

    counts = df['predicted_sentiment'].value_counts()
    msg = (
        f"📊 *Detailed Strategic News Sentiment Report*\n\n"
        f"• Positive: {counts.get('Positive', 0)} 👍\n"
        f"• Negative: {counts.get('Negative', 0)} 👎\n"
        f"• Neutral: {counts.get('Neutral', 0)} 😐\n\n"
    )

    for _, row in df.head(5).iterrows():
        title = row['text'].split('.')[0]
        msg += f"• <{row['url']}|{title}> — {row['predicted_sentiment']} ({row['sentiment_score']:.2f})\n"

    try:
        requests.post(webhook_url, json={"text": msg})
        st.success("✅ Detailed report successfully posted to Slack!")
    except Exception as e:
        st.error(f"❌ Slack Report Error: {e}")


# --- 6. STREAMLIT INTEGRATION ---
def run_dataset_loader():
    # --- STYLING ---
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #0f172a, #1e293b 70%);
            font-family: 'Inter', sans-serif;
            color: #f8fafc;
        }

        h1 {
            text-align: center;
            font-size: 2.5em;
            color: #f1f5f9;
            font-weight: 800;
            margin-bottom: 0.3em;
        }

        .subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.05em;
            margin-bottom: 1.5em;
        }

        .glass-card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 1.5rem;
        }

        .stButton>button {
            background: linear-gradient(90deg, #2563eb, #3b82f6);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.7em 1.4em;
            font-weight: 600;
            font-size: 1.1em;
            box-shadow: 0 0 12px rgba(59,130,246,0.4);
            transition: 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #1d4ed8, #2563eb);
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(59,130,246,0.6);
        }

        .alert-box {
            background: rgba(59, 130, 246, 0.1);
            border-left: 4px solid #3b82f6;
            padding: 0.8rem 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- HEADER ---
    

    # --- CONFIGURATION PANEL ---
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🧩 Configuration")

    model, news_api_key, slack_url = setup_environment()
    col1, col2 = st.columns(2)
    with col1:
        date_input = st.date_input("📅 Select Date")
    with col2:
        topic = st.text_input("🧠 Enter Topic", "AI OR artificial intelligence OR technology")

    max_results = st.slider("📑 Number of Articles to Fetch", 10, 100, 50)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- RUN BUTTON ---
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    run = st.button("🚀 Run Full Sentiment Analysis")
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        with st.spinner("⚙️ Processing live news data..."):
            articles = fetch_news_data(news_api_key, query=topic, max_results=max_results)
            if not articles:
                st.stop()

            df = pd.DataFrame(articles)
            st.progress(0)
            results = []

            for i, row in enumerate(df.itertuples(), start=1):
                label, score = get_sentiment_with_gemini(model, row.text)
                results.append({'predicted_sentiment': label, 'sentiment_score': score})
                st.progress(i / len(df))
                time.sleep(1.2)

            df = df.join(pd.DataFrame(results))
            st.success("✅ Sentiment analysis completed!")

            timestamp = date_input.strftime("%Y-%m-%d")
            output_file = f"news_sentiment_report_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            st.markdown(f"<div class='alert-box'>💾 Report saved as <b>{output_file}</b></div>", unsafe_allow_html=True)
            st.dataframe(df.head(10))

            # --- SLACK NOTIFICATIONS ---
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("📡 Slack Notification Options")
            st.caption("Choose what type of report to send to your Slack workspace.")
            send_summary = st.checkbox("📢 Send Slack Summary Alert (Counts Only)")
            send_report = st.checkbox("📤 Send Detailed Report (Top 5 Articles)")

            if send_summary:
                send_slack_summary(slack_url, df)
            if send_report:
                send_slack_alert(slack_url, df)
            st.markdown("</div>", unsafe_allow_html=True)
