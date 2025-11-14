# 🧠 Strategic Intelligence Platform

> Real-Time AI-Powered News Analysis & Sentiment Intelligence System

🚀 This project uses Google Gemini, and NewsAPI

It automatically:
* Fetches the latest news 📰
* Analyzes sentiment using Gemini 🤖
* Sends summarized insights to Slack 💬
* Generates interactive visualizations 📊
* Builds word clouds to highlight trending topics ☁️

---

## ⚙️ Features

* ✅ **Automated News Fetching:** Pulls fresh, relevant articles via NewsAPI.org
* ✅ **Dual Sentiment Engine:** Uses Gemini for precision, Hugging Face for fallback
* ✅ **Slack Notifications:** Instantly delivers insights to your Slack workspace
* ✅ **Data Visualization:** Interactive Plotly charts and word clouds
* ✅ **Smart Error Handling:** Detects and handles API limits gracefully

---

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Configuration

Before running the script, you need to set up your API keys. Create a `.env` file in the root directory and add your keys:

```ini
# .env file
NEWS_API_KEY="YOUR_NEWSAPI_KEY_HERE"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
SLACK_WEBHOOK_URL="YOUR_SLACK_WEBHOOK_URL_HERE"
```

```
           ┌──────────────────────┐
           │  NewsAPI.org         │
           └─────────┬────────────┘
                     │  (Fetches news)
                     ▼
           ┌──────────────────────┐
           │  Gemini + HF Model   │
           │ (Sentiment Analysis) │
           └─────────┬────────────┘
                     │
                     ▼
           ┌──────────────────────┐
           │   Pandas Processing  │
           │   (Store + Analyze)  │
           └─────────┬────────────┘
                     │
                     ▼
   ┌───────────────────────────────┐
   │ Slack Alerts + Visual Reports │
   │  (Plotly, WordCloud)          │
   └───────────────────────────────┘
```

## 4.  Next Part Next code 

## 🔮 Sentiment Forecasting Module

> Predicting future sentiment trends using Prophet & Polynomial Regression

This module extends your Strategic Intelligence Platform by forecasting sentiment dynamics over time 📈. It automatically detects all previously saved CSV sentiment reports, computes the average sentiment trend, and predicts future values using two modeling techniques — Facebook Prophet and Polynomial Regression.

5. ### Workflow Overview

```text
📁 CSV Folder (Historical Reports)
     │
     ▼
🧮 Data Aggregation
     │
     ├── Seaborn Grid Visualization (per dataset)
     │
     ├── Prophet Model (Time-series forecast)
     │
     ├── Polynomial Regression (Trend prediction)
     │
     ▼
📊 Plotly Comparison Chart
     │
     ▼
💬 Slack Report (Predicted sentiment for next 5 intervals)
```


⚙️ Key Features
✅ Automatic CSV Detection — No manual file input needed; the script scans the folder and loads all .csv sentiment reports.

✅ Beautiful Visualizations — Seaborn grid for dataset insights, Plotly chart comparing Prophet vs Polynomial Regression.

✅ Forecasting Engine — Uses both Prophet and Polynomial Regression to predict the next 5 sentiment points.

✅ Slack Integration — Instantly sends the forecast summary to your Slack workspace for quick team insights.

✅ Multi-Model Comparison — Helps identify which forecasting method fits sentiment data more accurately.

6. 🚀 How to Run
Ensure previous sentiment reports (e.g., news_sentiment_report_YYYY-MM-DD_HH-MM-SS.csv) are in the reports/ folder.

Run the script:
```predict_future_sentiment.py```

7. Wait for outputs:

🖼️ sentiment_grid_chart.png → Scatter plots for all CSVs

🧠 sentiment_forecast_chart.png → Prophet vs Polynomial forecast comparison

💬 Slack message → Predicted next 5 sentiment scores
