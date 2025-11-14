import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Imports for Preprocessing & EDA 
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
 
# Input and Output 
SOURCE_CSV = 'news_articles.csv'
PREDICTIONS_CSV = 'analyzed_news_predictions.csv'
SENTIMENT_CHART_IMAGE = 'sentiment_distribution.png'
WORDCLOUD_IMAGE = 'word_cloud.png'

# Custom Sentiment Dictionary 
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'positive', 'fortunate', 'correct', 'superior',
    'gain', 'gains', 'surpass', 'rise', 'rising', 'soar', 'soaring', 'bullish',
    'win', 'winning', 'success', 'successful', 'achieve', 'achieved', 'benefit',
    'improve', 'improvement', 'hope', 'optimistic', 'profit', 'profitable', 'boom'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'horrible', 'negative', 'unfortunate', 'wrong', 'inferior',
    'loss', 'losses', 'decline', 'fall', 'falling', 'plunge', 'plunging', 'bearish',
    'lose', 'losing', 'fail', 'failure', 'downturn', 'sad', 'pessimistic', 'disappointing',
    'risk', 'risky', 'crisis', 'slump', 'fear', 'crash'
}

# Text Cleaning and Preprocessing
def preprocess_text(text, stop_words):
    """Cleans and preprocesses a single piece of text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Lowercase text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    # Remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# Exploratory Data Analysis (EDA)
def perform_eda(df):
    """Performs EDA by generating and displaying a word cloud."""
    print("\nStep 2: Performing Exploratory Data Analysis (EDA)...")
    
    # Combine all cleaned text into one large corpus
    text_corpus = " ".join(text for text in df['cleaned_text'])
    
    if not text_corpus.strip():
        print("Cannot generate word cloud, no text available after cleaning.")
        return
        
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_corpus)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Words in News Articles')
    plt.savefig(WORDCLOUD_IMAGE)
    plt.show()
    print(f"✅ Word cloud saved to '{WORDCLOUD_IMAGE}'")


# Sentiment Analysis Function
def get_sentiment_from_lexicon(text):
    """Analyzes sentiment by counting positive and negative words in pre-cleaned text."""
    words = text.split()
    word_counts = Counter(words)
    positive_score = sum(word_counts[word] for word in POSITIVE_WORDS)
    negative_score = sum(word_counts[word] for word in NEGATIVE_WORDS)

    if positive_score > negative_score:
        return 'Positive'
    elif negative_score > positive_score:
        return 'Negative'
    else:
        return 'Neutral'

# Visualization & Saving 
def visualize_and_save(df):
    """Visualizes sentiment distribution and saves the results."""
    print("\nStep 4: Visualizing and Saving Final Results...")
    
    # Visualize Sentiment Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='predicted_sentiment', data=df,
                  palette={'Positive': 'green', 'Negative': 'red', 'Neutral': 'grey'},
                  order=['Positive', 'Negative', 'Neutral'])
    plt.title('Sentiment Distribution of News Articles')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Number of Articles')
    plt.savefig(SENTIMENT_CHART_IMAGE)
    plt.show()
    print(f"✅ Sentiment chart saved to '{SENTIMENT_CHART_IMAGE}'")
    
    # Save results to CSV
    df.to_csv(PREDICTIONS_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ Analyzed predictions saved to '{PREDICTIONS_CSV}'")

# Main Execution 
if __name__ == "__main__":
    # Check if the source CSV file exists
    if not os.path.exists(SOURCE_CSV):
        print(f"Error: The file '{SOURCE_CSV}' was not found.")
        print("Please make sure your data file is in the same directory as the script.")
    else:
        print(f"Step 1: Reading data from '{SOURCE_CSV}'...")
        df = pd.read_csv(SOURCE_CSV)
        print(f"✅ Loaded {len(df)} articles.")
        
        # Combine title and description for analysis, handling potential missing values
        df['text_to_analyze'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        
        # Preprocess the text data
        print("\nStep 2: Preprocessing text data...")
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
        
        df['cleaned_text'] = df['text_to_analyze'].apply(lambda text: preprocess_text(text, stop_words))
        print("✅ Text preprocessing complete.")

        # Perform EDA
        perform_eda(df)
        
        # Apply sentiment analysis
        print("\nStep 3: Applying sentiment analysis...")
        df['predicted_sentiment'] = df['cleaned_text'].apply(get_sentiment_from_lexicon)
        print("✅ Sentiment analysis complete.")
        
        # Keep original columns and add the new ones for the final output
        # This prevents errors if a column like 'source_id' or 'link' doesn't exist
        output_columns = [col for col in ['title', 'description', 'url', 'source'] if col in df.columns]
        output_columns.extend(['predicted_sentiment', 'cleaned_text'])
        
        final_df = df[output_columns]
        
        # Visualize and save the final, analyzed data
        visualize_and_save(final_df)
        
        print("\nPipeline finished successfully!")