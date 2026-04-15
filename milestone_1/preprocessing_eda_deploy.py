import pandas as pd
import re
import logging
from pathlib import Path
from typing import Optional, Set

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class NewsSentimentPipeline:
    """An end-to-end pipeline for processing, visualizing, and analyzing news sentiment."""

    def __init__(self, source_csv: str, output_dir: str = '.'):
        self.source_file = Path(source_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions_csv = self.output_dir / 'analyzed_news_predictions.csv'
        self.sentiment_chart = self.output_dir / 'sentiment_distribution.png'
        self.wordcloud_img = self.output_dir / 'word_cloud.png'
        
        self.df: Optional[pd.DataFrame] = None
        self.stop_words: Set[str] = set()
        self.sia = SentimentIntensityAnalyzer()
        
        self._setup_nltk()

    def _setup_nltk(self) -> None:
        """Downloads required NLTK datasets quietly."""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {e}")

    def load_data(self) -> bool:
        """Loads data from the source CSV."""
        if not self.source_file.exists():
            logger.error(f"Source file '{self.source_file}' not found.")
            return False
            
        try:
            self.df = pd.read_csv(self.source_file)
            logger.info(f"Successfully loaded {len(self.df)} articles from {self.source_file.name}")
            return True
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return False

    def preprocess_text(self, text: str) -> str:
        """Cleans text by removing URLs, special characters, and stopwords."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # URLs
        text = re.sub(r'[^a-z\s]', '', text)                                    # Punctuation/Numbers
        
        words = [word for word in text.split() if word not in self.stop_words]
        return " ".join(words)

    def preprocess_dataframe(self) -> None:
        """Prepares and cleans the dataframe for analysis."""
        logger.info("Preprocessing text data...")
        
        # Combine title and description safely
        self.df['text_to_analyze'] = self.df.get('title', '').fillna('') + ' ' + self.df.get('description', '').fillna('')
        
        # Apply vector-friendly text cleaning
        self.df['cleaned_text'] = self.df['text_to_analyze'].apply(self.preprocess_text)
        
        # Drop rows where cleaned text is entirely empty to prevent noise
        self.df = self.df[self.df['cleaned_text'].str.strip() != ''].copy()
        logger.info(f"Preprocessing complete. {len(self.df)} valid rows remain.")

    def perform_eda(self) -> None:
        """Generates and saves a Word Cloud for Exploratory Data Analysis."""
        logger.info("Performing Exploratory Data Analysis (EDA)...")
        text_corpus = " ".join(self.df['cleaned_text'])
        
        if not text_corpus.strip():
            logger.warning("No text available to generate a Word Cloud.")
            return

        try:
            wordcloud = WordCloud(
                width=1000, height=500, 
                background_color='white',
                colormap='viridis',
                max_words=200
            ).generate(text_corpus)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Frequent Words in News Articles', fontsize=16)
            plt.tight_layout(pad=0)
            plt.savefig(self.wordcloud_img)
            plt.close() # Close plot to free up memory
            logger.info(f"Word cloud saved to '{self.wordcloud_img.name}'")
        except Exception as e:
            logger.error(f"Failed to generate Word Cloud: {e}")

    def analyze_sentiment(self) -> None:
        """Uses NLTK's VADER to accurately analyze sentiment polarity."""
        logger.info("Applying VADER Sentiment Analysis...")
        
        def get_vader_label(text: str) -> str:
            # VADER outputs a compound score between -1 (extreme negative) and +1 (extreme positive)
            compound_score = self.sia.polarity_scores(text)['compound']
            if compound_score >= 0.05:
                return 'Positive'
            elif compound_score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'

        # Note: We analyze the raw text with VADER, as it utilizes punctuation and capitalization for scoring context
        self.df['predicted_sentiment'] = self.df['text_to_analyze'].apply(get_vader_label)
        logger.info("Sentiment analysis complete.")

    def visualize_and_export(self) -> None:
        """Creates visualizations and exports the final dataset."""
        logger.info("Visualizing and saving final results...")
        
        # 1. Visualize Distribution
        try:
            plt.figure(figsize=(9, 6))
            sns.countplot(
                x='predicted_sentiment', 
                data=self.df,
                hue='predicted_sentiment', # Updated to modern Seaborn syntax
                palette={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'},
                order=['Positive', 'Neutral', 'Negative'],
                legend=False
            )
            plt.title('Sentiment Distribution of News Articles', fontsize=14)
            plt.xlabel('Predicted Sentiment', fontsize=12)
            plt.ylabel('Number of Articles', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(self.sentiment_chart)
            plt.close()
            logger.info(f"Sentiment chart saved to '{self.sentiment_chart.name}'")
        except Exception as e:
            logger.error(f"Failed to generate sentiment chart: {e}")

        # 2. Export to CSV
        try:
            # Dynamically select output columns to avoid KeyError
            standard_cols = ['title', 'description', 'url', 'source']
            output_columns = [col for col in standard_cols if col in self.df.columns]
            output_columns.extend(['text_to_analyze', 'predicted_sentiment'])
            
            final_df = self.df[output_columns]
            final_df.to_csv(self.predictions_csv, index=False, encoding='utf-8-sig')
            logger.info(f"Analyzed predictions saved to '{self.predictions_csv.name}'")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    def run(self) -> None:
        """Executes the pipeline sequentially."""
        logger.info("--- Starting News Sentiment Pipeline ---")
        if self.load_data():
            self.preprocess_dataframe()
            self.perform_eda()
            self.analyze_sentiment()
            self.visualize_and_export()
        logger.info("--- Pipeline Execution Finished ---")


if __name__ == "__main__":
    # Initialize and run the pipeline
    pipeline = NewsSentimentPipeline(
        source_csv='news_articles.csv', 
        output_dir='./analysis_output'
    )
    pipeline.run()
