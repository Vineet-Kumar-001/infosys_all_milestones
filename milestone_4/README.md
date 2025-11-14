# 🧠 Strategic Sentiment Intelligence Platform

A modern, AI‑powered news intelligence system that fetches global news in real time, analyzes sentiment using Google Gemini, visualizes long‑term trends, and sends smart alerts directly to Slack.

This platform is designed for analysts, researchers, students, and businesses who want a fast, automated way to understand how the world’s sentiment shifts across topics.

---

## 🚀 Key Features

### 🔍 Live News Fetching

Fetch global news articles using **NewsData.io API** with keyword‑based filtering.

### 🤖 AI Sentiment Analysis (Google Gemini)

Each article is classified as:

* Positive
* Negative
* Neutral
  with a sentiment score between ‑1 and +1.

### 📩 Slack Auto‑Reporting

Send:

* Quick summary alerts
* Detailed sentiment reports
* 5‑day forecast predictions
* AI‑generated comprehensive analysis

directly to your Slack workspace.

### 📊 Visual Analytics Dashboard

Built using **Streamlit**, it offers:

* Time‑series sentiment tracking
* Prophet‑based forecasting
* Sentiment heatmaps
* Word clouds
* Article‑level exploration
* Interactive timeline slider

### 📁 Dataset Loader

Automatically loads all previously analyzed datasets and aggregates them for long‑term insights.

---

## 📦 Project Structure

```
📂 project/
│
├── app_file.py               # Visualization dashboard
├── datasetloader.py          # News fetcher + sentiment analyzer
├── main.py                   # Main Streamlit entry point
├── .env                      # API keys
├── 📂 Datasets/              # Auto‑generated daily sentiment CSV files
└── README.md                 # Documentation
```

---

## 🔧 Installation

```
git clone https://github.com/yourusername/strategic-sentiment-platform.git
cd strategic-sentiment-platform
pip install -r requirements.txt
```

---

## 🔑 Environment Variables (.env)

Create a `.env` file in the project root:

```
GEMINI_API_KEY="your_key"
NEWSDATA_API_KEY="your_key"
SLACK_WEBHOOK_URL="your_key"
SLACK_BOT_TOKEN="your_key"
```

These values are required for the platform to function.

---

## ▶️ Running the Platform

Launch Streamlit:

```
streamlit run main.py
```

You’ll see two modules:

### 📰 New Dataset

Fetch fresh articles and run sentiment analysis.

### 📈 Data Visualization

Explore historical datasets, forecasts, heatmaps, and more.

---

## 📊 Prophet Forecasting

The system uses **Facebook Prophet** to generate:

* Trend fitting for past data
* 5‑day future sentiment prediction

These predictions can also be pushed to Slack.

---

## 🧠 AI‑Generated Insights

Gemini summarizes:

* Sentiment shifts
* Turning points
* Overall emotional landscape

These insights help analysts understand narrative changes over time.

---

## 📡 Slack Integration

The platform sends:

* Summary alert
* Detailed top‑5 article report
* Forecast insights
* AI analysis summary

to your Slack channel using an Incoming Webhook.

---

## 🎨 UI & Styling

The app comes with a premium modern UI:

* Neon gradient buttons
* Glassmorphism cards
* Custom sidebar theme
* Dark dashboard layout

Built entirely in Streamlit and Plotly.

---

## 🛠 Requirements

* Python 3.10+
* Streamlit
* Prophet
* Plotly
* Google Generative AI
* NewsData API

Install all dependencies:

```
pip install -r requirements.txt
```

