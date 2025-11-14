# ---------------- analyze_sentiment.py ----------------
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

nltk.download("stopwords")

# ------------------ TEXT CLEANING ------------------
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = " ".join([w for w in text.split() if w not in stopwords.words("english")])
    return text


# ------------------ GEMINI SENTIMENT ------------------
def get_sentiment(model, text):
    prompt = f"""
    Analyze the sentiment.
    Return JSON only with:
    sentiment_label: Positive/Negative/Neutral
    sentiment_score: value between -1 and 1

    Text: "{text}"
    """
    try:
        response = model.generate_content(prompt)
        cleaned = response.text.replace("```json","").replace("```","")
        data = json.loads(cleaned)
        return data["sentiment_label"], float(data["sentiment_score"])
    except:
        return "Neutral", 0.0


# ------------------ EDA VISUALIZATIONS ------------------
def save_wordcloud(df, label, filename):
    text = " ".join(df[df["predicted_sentiment"] == label]["clean_text"])
    if not text.strip():
        return
    wc = WordCloud(width=800, height=500, background_color="white").generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(filename)
    plt.close()


def plot_distribution(df):
    plt.figure(figsize=(7,5))
    df["predicted_sentiment"].value_counts().plot(kind="bar")
    plt.title("Sentiment Distribution")
    plt.savefig("eda_sentiment_distribution.png")
    plt.close()

    plt.figure(figsize=(7,5))
    df["word_count"].plot(kind="hist", bins=30)
    plt.title("Word Count Distribution")
    plt.savefig("eda_word_count_distribution.png")
    plt.close()


# ------------------ MAIN PIPELINE ------------------
def main():
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("models/gemini-2.5-flash-preview-09-2025")

    filename = input("Enter fetched CSV filename (example: fetched_2025-01-10.csv): ")

    df = pd.read_csv(filename)

    df["clean_text"] = df["text"].apply(clean_text)
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

    sentiments = df["clean_text"].apply(lambda t: get_sentiment(model, t))

    df["predicted_sentiment"] = sentiments.apply(lambda x: x[0])
    df["sentiment_score"] = sentiments.apply(lambda x: x[1])

    # EDA
    plot_distribution(df)

    # Wordclouds
    save_wordcloud(df, "Positive", "wordcloud_positive.png")
    save_wordcloud(df, "Negative", "wordcloud_negative.png")

    date_stamp = datetime.now().strftime("%Y-%m-%d")
    output_csv = f"analysis_sentiment_{date_stamp}.csv"
    df.to_csv(output_csv, index=False)

    print("Analysis complete.")
    print(f"Saved: {output_csv}")
    print("PNG files saved: wordcloud_positive.png, wordcloud_negative.png, eda_*.png")


if __name__ == "__main__":
    main()
