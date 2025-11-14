# --- IMPORTS ---
import os
import glob
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import requests
from dotenv import load_dotenv
from wordcloud import WordCloud
import google.generativeai as genai
import collections
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="", layout="wide")

# --- GLOBAL STYLING ---
st.markdown("""
<style>
.block-container {padding:0.5rem 1rem;}
h1,h2,h3,p{color:#f8fafc;}
[data-testid="stMetricValue"]{font-size:1.6rem;}
.stButton>button{
    background:linear-gradient(90deg,#3b82f6,#2563eb);
    color:white;border:none;border-radius:10px;
    padding:0.5rem 1.2rem;font-weight:600;
    box-shadow:0 0 10px rgba(37,99,235,0.4);
    transition:0.2s ease;
}
.stButton>button:hover{transform:scale(1.05);}
.analysis-box{
    background:#0f172a;padding:15px;border-radius:10px;
    font-size:0.9rem;line-height:1.5;
    border:1px solid rgba(255,255,255,0.1);
    box-shadow:0 0 10px rgba(0,0,0,0.4);
}
.small-caption{color:#94a3b8;font-size:0.9rem;margin-top:0.3rem;}
</style>
""", unsafe_allow_html=True)

# --- ENVIRONMENT SETUP ---
def setup_env():
    load_dotenv()
    slack_url = os.getenv("SLACK_WEBHOOK_URL")
    gemini_api = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=gemini_api)
    model = genai.GenerativeModel("models/gemini-2.5-flash-preview-09-2025")
    return slack_url, model

# --- LOAD DATA ---
def load_sentiment_data():
    dataset_dir = os.path.join(os.path.dirname(__file__), "Datasets")
    csv_files = sorted(glob.glob(os.path.join(dataset_dir, "news_sentiment_report_*.csv")))

    if not csv_files:
        st.error("❌ No sentiment reports found.")
        return pd.DataFrame(), pd.DataFrame(), []

    daily, counts, all_df = [], [], []
    for f in csv_files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.lower()
        if "sentiment_score" not in df.columns:
            continue
        date = pd.to_datetime(os.path.basename(f).split("report_")[1].replace(".csv", ""))
        mean = df["sentiment_score"].mean()
        pos, neg, neu = (df["sentiment_score"] > 0).sum(), (df["sentiment_score"] < 0).sum(), (df["sentiment_score"] == 0).sum()
        daily.append({"date": date, "mean_score": mean})
        counts.append({"date": date, "positive": pos, "negative": neg, "neutral": neu})
        df["date"] = date
        all_df.append(df)
    return (pd.DataFrame(daily).sort_values("date"),
            pd.DataFrame(counts).sort_values("date"),
            pd.concat(all_df, ignore_index=True))

# --- PROPHET FORECAST ---
def prophet_forecast(daily_df):
    prophet_df = daily_df.rename(columns={"date": "ds", "mean_score": "y"})
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=5)
    return model.predict(future)

# --- SLACK HELPERS ---
def send_forecast_to_slack(webhook_url, forecast):
    msg = "📢 *5-Day Sentiment Forecast*\n\n"
    f5 = forecast.tail(5)[["ds", "yhat"]]
    for _, r in f5.iterrows():
        e = "📈" if r.yhat > 0.1 else "📉" if r.yhat < 0 else "➖"
        msg += f"• *{r.ds.strftime('%b %d')}* → `{r.yhat:.3f}` {e}\n"
    avg = f5["yhat"].mean()
    mood = "🌞 Positive" if avg > 0.1 else "🌤 Neutral" if avg >= 0 else "🌧 Negative"
    msg += f"\n📊 *Average:* `{avg:.3f}` → {mood}"
    try:
        requests.post(webhook_url, json={"text": msg})
        st.success("✅ Forecast sent to Slack!")
    except Exception as e:
        st.error(f"Slack Error: {e}")

def send_analysis_to_slack(webhook_url, analysis_text):
    msg = f"🧩 *Comprehensive Sentiment & News Analysis (Since Oct 22)*\n\n{analysis_text}"
    try:
        requests.post(webhook_url, json={"text": msg})
        st.success("✅ AI Analysis sent to Slack!")
    except Exception as e:
        st.error(f"Slack Error: {e}")

# --- DASHBOARD ---
def run_data_visualization():
    slack_url, model = setup_env()
    st.markdown("<h1 style='text-align:center;'></h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    daily_df, sentiment_df, combined_df = load_sentiment_data()
    if daily_df.empty:
        st.stop()

    # --- INTERACTIVE TIMELINE SLIDER ---
    st.subheader("🕹️ Interactive Timeline: Zoom & Filter")
    min_date = daily_df["date"].min().to_pydatetime()
    max_date = daily_df["date"].max().to_pydatetime()
    start_date, end_date = st.slider(
        "Select analysis window:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    daily_df = daily_df[(daily_df["date"] >= start_date) & (daily_df["date"] <= end_date)]
    sentiment_df = sentiment_df[(sentiment_df["date"] >= start_date) & (sentiment_df["date"] <= end_date)]
    combined_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= end_date)]

    # --- METRICS ROW ---
    total = len(combined_df)
    pos, neg, neu = (combined_df["sentiment_score"] > 0).sum(), (combined_df["sentiment_score"] < 0).sum(), (combined_df["sentiment_score"] == 0).sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📰 Total", total)
    c2.metric("✅ Positive", pos)
    c3.metric("❌ Negative", neg)
    c4.metric("⚪ Neutral", neu)

    # --- FORECAST + PIE ---
    forecast = prophet_forecast(daily_df)
    daily_df["rolling"] = daily_df["mean_score"].rolling(3).mean()
    past = forecast[forecast["ds"] <= daily_df["date"].max()]
    future = forecast[forecast["ds"] > daily_df["date"].max()]

    c1, c2 = st.columns([2.5, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_df["date"], y=daily_df["mean_score"], mode="lines+markers",
                                 name="Daily Sentiment", line=dict(color="#00FFFF")))
        fig.add_trace(go.Scatter(x=past["ds"], y=past["yhat"], mode="lines", name="Prophet Fit",
                                 line=dict(color="#38bdf8", width=2)))
        fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat"], mode="lines+markers",
                                 name="Next 5-Day", line=dict(color="yellow", dash="dash")))
        fig.update_layout(template="plotly_dark", height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pie = go.Figure(data=[go.Pie(labels=["Positive", "Negative", "Neutral"], values=[pos, neg, neu],
                                     hole=0.6, marker=dict(colors=["#22c55e", "#ef4444", "#94a3b8"]))])
        pie.update_layout(template="plotly_dark", height=340)
        st.plotly_chart(pie, use_container_width=True)

    # --- COMPACT WORD CLOUDS BELOW PIE ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🔤 WordClouds: Positive vs Negative (Compact View)")
    col1, col2 = st.columns(2)
    with col1:
        pos_text = " ".join(combined_df.loc[combined_df["sentiment_score"] > 0, "text"].astype(str))
        if pos_text.strip():
            pos_wc = WordCloud(width=500, height=250, background_color="black", colormap="Greens").generate(pos_text)
            st.image(pos_wc.to_array(), caption="🟢 Positive Keywords", use_container_width=True)
        else:
            st.info("No positive keywords available in this range.")
    with col2:
        neg_text = " ".join(combined_df.loc[combined_df["sentiment_score"] < 0, "text"].astype(str))
        if neg_text.strip():
            neg_wc = WordCloud(width=500, height=250, background_color="black", colormap="Reds").generate(neg_text)
            st.image(neg_wc.to_array(), caption="🔴 Negative Keywords", use_container_width=True)
        else:
            st.info("No negative keywords available in this range.")

    # --- AI ANALYSIS ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🧩 Comprehensive Sentiment & News Analysis (Dynamic Range)")
    ai_analysis_text = None
    if not combined_df.empty:
        try:
            latest_sentiments = daily_df[["date", "mean_score"]].to_dict("records")
            sample_texts = " ".join(combined_df["text"].astype(str).tolist()[:1000])
            prompt = f"""
            Summarize key sentiment changes from {start_date.strftime('%b %d')} to {end_date.strftime('%b %d')}:
            - Identify turning points
            - Describe general sentiment trend
            - Conclude with one-line insight
            Data: {latest_sentiments}
            Text sample: {sample_texts[:1500]}
            """
            ai_analysis = model.generate_content(prompt)
            ai_analysis_text = ai_analysis.text
            st.markdown(f"<div class='analysis-box'>{ai_analysis_text}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"AI Summary Error: {e}")

    # --- HEATMAP ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🌡️ Sentiment Heatmap (-1 = Negative → +1 = Positive)")
    daily_df["clamped"] = daily_df["mean_score"].clip(-1, 1)
    heatmap = px.density_heatmap(
        daily_df, x="date", y="mean_score", z="clamped",
        nbinsx=10, color_continuous_scale=[(0, "#7f0000"), (0.5, "#ffffcc"), (1, "#006d2c")],
        range_color=[-1, 1], template="plotly_dark"
    )
    st.plotly_chart(heatmap, use_container_width=True)

    # --- SLACK BUTTONS ---
    st.markdown("<hr>", unsafe_allow_html=True)
    cols = st.columns([4, 1, 1])
    with cols[1]:
        if slack_url and st.button("📡 Send Forecast to Slack"):
            send_forecast_to_slack(slack_url, forecast)
    with cols[2]:
        if slack_url and ai_analysis_text and st.button("🧠 Send AI Analysis to Slack"):
            send_analysis_to_slack(slack_url, ai_analysis_text)

# --- RUN APP ---
if __name__ == "__main__":
    run_data_visualization()
