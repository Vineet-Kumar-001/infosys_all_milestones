
import os
import sys
import json
import time
import requests
import pandas as pd
import nltk
from datetime import datetime
from dotenv import load_dotenv
from newsapi import NewsApiClient  # Using newsapi.org client
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from transformers import pipeline
import google.generativeai as genai


# --- 1. SETUP AND CONFIGURATION ---
def setup_environment():
    print("🚀 Starting the Strategic Intelligence Platform...")

    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    newsapi_key = os.getenv("NEWSAPI_KEY")
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")

    if not all([gemini_api_key, newsapi_key, slack_webhook_url]):
        print("\n❌ FATAL ERROR: Missing one or more environment variables.")
        print("Ensure GEMINI_API_KEY, NEWSAPI_KEY, and SLACK_WEBHOOK_URL are in your .env file.")
        sys.exit()

    # Configure Gemini
    genai.configure(api_key=gemini_api_key)
    
    # --- *** THE FIX *** ---
    # 1. Changed model to the stable "gemini-1.0-pro".
    # 2. 100% removed the "models/" prefix that caused all 404 errors.
    # This will work *after* you run "pip install --upgrade google-generativeai"
    gemini_model = genai.GenerativeModel("models/gemini-2.5-flash") 

    # Ensure NLTK stopwords
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    # Local Hugging Face fallback model
    fallback_model = pipeline("sentiment-analysis")

    return gemini_model, fallback_model, newsapi_key, slack_webhook_url


# --- 2. FETCH NEWS ---
def fetch_news_data(api_key, query="next-generation battery technology OR quantum battery OR advanced materials for green hydrogen", max_results=50):
    print(f"🔍 Fetching news from NewsAPI.org for query: '{query}' (Target: {max_results} articles)...")
    try:
        api_client = NewsApiClient(api_key=api_key)
        
        response = api_client.get_everything(q=query,
                                             language='en',
                                             sort_by='relevancy',
                                             page_size=max_results)

        articles = [
            {"source": a.get('source', {}).get('name', 'NewsAPI.org'),
             "text": f"{a.get('title', '')}. {a.get('description', '')}",
             "url": a.get('url')}
            for a in response.get('articles', [])
            if a.get('title') and a.get('description')
        ]
        
        if not articles:
            print("⚠️ No articles found.")
        print(f"✅ Fetched {len(articles)} articles.")
        return articles
    except Exception as e:
        print(f"❌ Error fetching from NewsAPI.org: {e}")
        return []


# --- 3. SENTIMENT ANALYSIS (GEMINI + FALLBACK) ---
def get_batch_sentiment(model, texts, fallback_model, batch_size=5):
    """Batch process sentiment with Gemini; fallback to Hugging Face if quota exceeded."""
    results = []
    
    # Corrected batch count logic
    num_batches = (len(texts) // batch_size) + (1 if len(texts) % batch_size > 0 else 0)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"   > Processing batch {i//batch_size + 1}/{num_batches}")
        combined_text = "\n\n".join(batch)

        prompt = f"""
        Analyze the sentiment of each paragraph below (each represents one news article).
        Return ONLY a valid JSON array (no other text). Each array element must be an object
        with two keys: "sentiment_label" (string: "Positive", "Negative", or "Neutral") 
        and "sentiment_score" (float: from -1.0 for very negative to 1.0 for very positive).
        
        Texts:
        {combined_text}
        """

        try:
            response = model.generate_content(prompt)
            clean = response.text.replace("```json", "").replace("```", "").strip()
            batch_results = json.loads(clean)
            
            if len(batch_results) == len(batch):
                results.extend(batch_results)
            else:
                print(f"⚠️ Gemini output mismatch (expected {len(batch)}, got {len(batch_results)}). Defaulting batch to Neutral.")
                results.extend([{"sentiment_label": "Neutral", "sentiment_score": 0.0}] * len(batch))

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print("⚠️ Gemini quota exceeded — switching to Hugging Face fallback model for this batch.")
                for t in batch:
                    hf_res = fallback_model(t[:512])[0]
                    label = "Positive" if hf_res["label"] == "POSITIVE" else "Negative"
                    score = hf_res["score"] if label == "Positive" else -hf_res["score"]
                    results.append({"sentiment_label": label, "sentiment_score": score})
            else:
                print(f"⚠️ Gemini/JSON error: {e} — defaulting batch to Neutral.")
                results.extend([{"sentiment_label": "Neutral", "sentiment_score": 0.0}] * len(batch))

        time.sleep(1) # Respect rate limits

    return results


# --- 4. SLACK ALERTING ---
def send_slack_alert(webhook_url, df, overall_average): # <-- MODIFIED
    if df.empty:
        print("⚠️ No data to send to Slack.")
        return

    print("📢 Sending summarized report to Slack...")

    sentiment_counts = df['sentiment_label'].value_counts()
    msg = (
        f"📈 *Daily Strategic News Analysis Report*\n\n"
        f"Total Articles Analyzed: {len(df)}\n"
        f"👍 Positive: {sentiment_counts.get('Positive', 0)}\n"
        f"👎 Negative: {sentiment_counts.get('Negative', 0)}\n"
        f"😐 Neutral: {sentiment_counts.get('Neutral', 0)}\n"
        f"📊 *Overall Average Score: {overall_average:.4f}*\n"  # <-- MODIFIED
        f"---------------------------------------\n"
    )

    df_sorted = df.reindex(df['sentiment_score'].abs().sort_values(ascending=False).index)

    msg += "\n*Top 5 Most Impactful Articles:*\n"
    for _, row in df_sorted.head(5).iterrows():
        msg += f"• <{row['url']}|{row['text'].split('.')[0]}>\n   *{row['sentiment_label']}* ({row['sentiment_score']:.2f})\n\n"

    try:
        requests.post(webhook_url, json={"text": msg})
        print("✅ Slack report sent successfully.")
    except Exception as e:
        print(f"❌ Slack send error: {e}")


# --- 5. VISUALIZATION ---
def create_visualizations(df):
    if df.empty:
        print("⚠️ No data to visualize.")
        return

    print("📊 Creating visualizations...")

    print("   > Generating Plotly scatter plot...")
    df_plot = df.reset_index().rename(columns={"index": "article_number"})
    
    fig = px.scatter(
        df_plot,
        x="article_number",
        y="sentiment_score",
        color="sentiment_label", 
        title="Sentiment Score per Article",
        hover_data=["text"], 
        color_discrete_map={
            "Positive": "#4CAF50",
            "Negative": "#F44336",
            "Neutral": "#9E9E9E"
        }
    )
    fig.update_layout(xaxis_title="Article Number", yaxis_title="Sentiment Score")
    
    plotly_filename = "sentiment_score_per_article.html"
    fig.write_html(plotly_filename)
    print(f"   ✅ Plotly chart saved as {plotly_filename}")

    print("   > Generating Positive/Negative word clouds...")
    stop_words = set(stopwords.words("english"))
    
    df["cleaned_text"] = df["text"].str.lower().str.replace(r"http\S+|[^a-z\s]", "", regex=True)
    df["cleaned_text"] = df["cleaned_text"].apply(
        lambda x: " ".join([word for word in x.split() if word not in stop_words and len(word) > 2])
    )

    positive_corpus = " ".join(df[df["sentiment_label"] == "Positive"]["cleaned_text"].dropna())
    negative_corpus = " ".join(df[df["sentiment_label"] == "Negative"]["cleaned_text"].dropna())

    if positive_corpus.strip():
        wc_pos = WordCloud(width=1200, height=700, background_color="white", colormap="Greens").generate(positive_corpus)
        plt.figure(figsize=(12, 8))
        plt.imshow(wc_pos, interpolation="bilinear")
        plt.title("Key Topics in Positive News", fontsize=16)
        plt.axis("off")
        plt.savefig("positive_key_topics_wordcloud.png", dpi=300, bbox_inches='tight')
        print("   ✅ Positive word cloud saved.")
        plt.close()
    else:
        print("   ⚠️ No positive articles found; skipping positive word cloud.")

    if negative_corpus.strip():
        wc_neg = WordCloud(width=1200, height=700, background_color="white", colormap="Reds").generate(negative_corpus)
        plt.figure(figsize=(12, 8))
        plt.imshow(wc_neg, interpolation="bilinear")
        plt.title("Key Topics in Negative News", fontsize=16)
        plt.axis("off")
        plt.savefig("negative_key_topics_wordcloud.png", dpi=300, bbox_inches='tight')
        print("   ✅ Negative word cloud saved.")
        plt.close()
    else:
        print("   ⚠️ No negative articles found; skipping negative word cloud.")


# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    gemini_model, hf_model, news_api_key, slack_url = setup_environment()
    
    news_articles = fetch_news_data(news_api_key, max_results=50)

    if news_articles:
        df = pd.DataFrame(news_articles)
        print(f"🧠 Performing sentiment analysis on {len(df)} articles...")

        texts = df["text"].tolist()
        sentiments = get_batch_sentiment(gemini_model, texts, hf_model, batch_size=5)

        sentiment_df = pd.DataFrame(sentiments)
        df = pd.concat([df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)

        # Calculate and display the overall average sentiment
        average_score = df['sentiment_score'].mean() # <-- MODIFIED
        print(f"📊 Overall Average Sentiment Score: {average_score:.4f}") # <-- MODIFIED

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"news_sentiment_report_{timestamp}.csv"
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"💾 Report saved as {filename}")

        # Pass the calculated average to the Slack function
        send_slack_alert(slack_url, df, average_score) # <-- MODIFIED
        
        create_visualizations(df)

        print("\n🎉 Analysis and notification complete!")
    else:
        print("⏹️ No news articles fetched. Execution halted.")