# ------------------ fetch_news.py ------------------
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from newsdataapi import NewsDataApiClient

def fetch_news(api_key, query, max_results=50):
    api_client = NewsDataApiClient(apikey=api_key)
    response = api_client.latest_api(q=query, language='en')

    articles = []

    for a in response.get("results", []):
        if a.get("title") and a.get("description"):
            articles.append({
                "title": a.get("title"),
                "description": a.get("description"),
                "text": f"{a.get('title')}. {a.get('description')}",
                "url": a.get("link")
            })

    return articles[:max_results]


def main():
    load_dotenv()

    api_key = os.getenv("NEWSDATA_API_KEY")
    query = input("Enter topic to fetch news: ")
    max_res = 50

    articles = fetch_news(api_key, query, max_res)

    if not articles:
        print("No data fetched.")
        return

    df = pd.DataFrame(articles)

    date_stamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"fetched_{date_stamp}.csv"

    df.to_csv(filename, index=False)
    print(f"Saved → {filename}")


if __name__ == "__main__":
    main()
