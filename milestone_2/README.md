# Explanation: What Each File Does

This document explains the purpose of each Python file and the `.env` configuration used in the sentiment news analysis platform.

---

## 1. `fetch_news.py`

This script connects to the **NewsData.io API** and downloads fresh news articles based on a topic you provide. Here's what it does step by step:

* Loads the NewsData API key from `.env`.
* Asks the user to type a topic (e.g., "AI", "Finance", "Elections").
* Calls the NewsData API and fetches the latest English-language news articles.
* Extracts the title, description, full text, and URL from each article.
* Saves all the collected news into a CSV file named like:

  ```
  fetched_2025-01-10.csv
  ```

This file becomes the input for the sentiment analysis script.

---

## 2. `analyze_sentiment.py`

This is the main analysis engine. It takes the CSV from `fetch_news.py`, cleans the text, sends it to Gemini for sentiment analysis, and generates visuals.

### Here is what it does:

### **a. Loads environment variables**

Gets your `GEMINI_API_KEY` from `.env` and configures the Gemini model.

### **b. Reads CSV**

Asks which fetched CSV file to analyze.

### **c. Cleans text**

Removes URLs, punctuation, and stopwords using NLTK.
Example:

```
"AI will change the world!!!" → "ai change world"
```

### **d. Sends cleaned text to Gemini**

Each cleaned article is sent to Gemini with a prompt asking for:

* sentiment_label → Positive / Neutral / Negative
* sentiment_score → between -1 and 1

### **e. Adds new columns to the dataframe**

* `clean_text`
* `word_count`
* `predicted_sentiment`
* `sentiment_score`

### **f. Generates EDA visualizations**

Creates:

* Sentiment distribution bar chart
* Word count histogram
* Positive sentiment wordcloud
* Negative sentiment wordcloud

All visuals are saved as PNG files.

### **g. Saves Final Output**

Exports a final CSV as:

```
analysis_sentiment_2025-01-10.csv
```

---

## 3. `.env`

This file stores your private keys. It keeps secrets out of the code.

### Contains:

```
GEMINI_API_KEY=your_gemini_api_key_here
NEWSDATA_API_KEY=your_newsdata_api_key_here
SLACK_WEBHOOK_URL=your_slack_webhook_here
```

* `GEMINI_API_KEY` → Used for sentiment analysis
* `NEWSDATA_API_KEY` → Needed for fetching news from NewsData.io
* `SLACK_WEBHOOK_URL` → Optional, used only if you want Slack notifications

---

## Summary

* **fetch_news.py** → Gets the raw news and saves it.
* **analyze_sentiment.py** → Cleans, analyzes, visualizes, and exports results.
* **.env** → Stores all API keys safely.

Together they form a complete mini intelligence system for real-time news sentiment analysis.
