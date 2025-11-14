import requests
import pandas as pd
import time
import sys
import os


API_KEY = "pub_42d17633ae9c4c40bb8fe7c205bdb3ff"
API_URL = "https://newsdata.io/api/1/news"

# The desired number of articles to fetch
MAX_ARTICLES = 100

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "news_articles.csv")


# Fetch a Page of News
def fetch_news(page_token=None):
    """
    Fetches a single page of news articles.
    """
    params = {
        "apikey": API_KEY,
        "language": "en",
        # the category to business, sports, etc.
        "category": "technology,business",
    }
    # If a token for the next page is provided, add it to the request
    if page_token:
        params['page'] = page_token

    try:
        response = requests.get(API_URL, params=params)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get("results", [])
        next_page = data.get("nextPage") # Token for the next page of results
        return articles, next_page

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return [], None


# Main Data Collection Loop
def main():
    """
    Main function to orchestrate fetching articles and saving them to a CSV.
    """
    all_articles = []
    next_page_token = None # Start with no token for the first request

    print("Starting data collection...")
    # Loop until we have collected the desired number of articles
    while len(all_articles) < MAX_ARTICLES:
        articles, next_page_token = fetch_news(next_page_token)

        if not articles:
            print("No more articles found or an error occurred. Stopping collection.")
            break

        all_articles.extend(articles)
        print(f"Collected {len(all_articles)} articles so far...")

        # If the API response has no 'nextPage' token, we've reached the end
        if not next_page_token:
            print("Reached the last page of results.")
            break

        # Polite pause between requests to avoid overwhelming the API server
        time.sleep(1)

    # Process and Save the Data
    if not all_articles:
        print("\nCollection failed. No data to process. Exiting.")
        sys.exit()

    print(f"\nFinished collection. Total articles fetched: {len(all_articles)}")
    # Trim the list to the exact number if the last fetch went over
    final_articles = all_articles[:MAX_ARTICLES]

    df = pd.DataFrame(final_articles)

    # Define the mapping from API field names to our desired column names
    columns_to_use = {
        'title': 'title',
        'description': 'description',
        'content': 'content',
        'link': 'url',
        'pubDate': 'publishedAt',
        'source_id': 'source'
    }

    # Filter for columns that actually exist in the DataFrame to prevent errors
    existing_cols = [api_name for api_name in columns_to_use.keys() if api_name in df.columns]
    df_processed = df[existing_cols]

    # Rename the columns to your desired final names
    df_processed = df_processed.rename(columns=columns_to_use)

    # Save the processed data to a CSV file
    df_processed.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"✅ Successfully saved {len(df_processed)} articles to '{CSV_FILE}'")


# Execute the script
if __name__ == "__main__":
    main()
