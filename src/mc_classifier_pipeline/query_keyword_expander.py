import argparse
import logging
import math
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from mc_classifier_pipeline import utils

# Configure logging
utils.configure_logging()
logger = logging.getLogger(__name__)

class PMIKeywordExpander:
    """
    A class to expand keywords using Pointwise Mutual Information (PMI) analysis.
    Analyzes text corpus to find words most associated with a seed keyword.
    """

    CSV_PATH = "data/search_results.csv"  #stores results from running doc_retriever.py
    TOP_KEYWORDS = 20

    def __init__(self, seed_word):
        """Initialize the PMI keyword expander."""
        self.article_df = None
        self.seed_word = seed_word
        self.doc_frequencies = Counter()  # documents containing each word
        self.cooccurrence_counts = Counter()  # documents containing both word and healthcare
        self.total_documents = 0
        self.seed_doc_count = 0
        self.pmi_scores = {}

    def load(self):
        """Load text data from CSV file."""
        try:
            self.article_df = pd.read_csv(self.CSV_PATH)
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False

    def preprocess(self):
        """Clean and tokenize text data."""
        # Download required NLTK data (only needs to be done once)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)

        # Filter for English articles only - modify original DataFrame
        self.article_df = self.article_df[self.article_df["language"] == "en"]

        # Convert to lowercase and remove punctuation
        self.article_df["cleaned_text"] = self.article_df["text"].str.lower().str.replace(r"[^\w\s]", "", regex=True)

        # Tokenize and remove stop words
        stop_words = set(stopwords.words("english"))
        self.article_df["tokens"] = self.article_df["cleaned_text"].apply(
            lambda x: [word for word in word_tokenize(x) if word not in stop_words and len(word) > 1]
        )

        # Drop the cleaned_text column as we no longer need it
        self.article_df = self.article_df.drop("cleaned_text", axis=1)

    def doc_frequency_count(self):
        """Count document frequencies and co-occurrences with seed word."""
        self.total_documents = len(self.article_df)

        # Count documents for each word
        for tokens in self.article_df["tokens"]:
            unique_words = set(tokens)

            # Count documents containing each word
            self.doc_frequencies.update(unique_words)
            # If this document contains seed word, count co-occurrences
            if self.seed_word in unique_words:
                self.cooccurrence_counts.update(unique_words)
        # Count total documents containing seed word
        self.seed_doc_count = self.doc_frequencies[self.seed_word]

    def PMI_calc(self):
        """Calculate PMI scores for all words with seed keyword."""
        for word in self.doc_frequencies:
            # Skip the seed word itself
            if word == self.seed_word or self.doc_frequencies[word] < 50:
                continue

            # Calculate probabilities
            P_word = self.doc_frequencies[word] / self.total_documents
            P_seed = self.seed_doc_count / self.total_documents
            P_both = self.cooccurrence_counts[word] / self.total_documents

            # Avoid division by zero or log(0)
            if P_both > 0 and P_word > 0 and P_seed > 0:
                pmi = math.log(P_both / (P_word * P_seed))
                self.pmi_scores[word] = pmi

    def ranking(self):
        """Rank words by PMI scores and filter top results."""
        # Sort words by PMI score (highest first)
        sorted_words = sorted(self.pmi_scores.items(), key=lambda x: x[1], reverse=True)

        # Take top N keywords
        self.top_keywords = sorted_words[: self.TOP_KEYWORDS]

    def expanded_keyword_set(self):
        """Return the final set of expanded keywords."""
        return [word for word, score in self.top_keywords]

    def generate_search_query(self):
        """Generate a search query using seed word and all top keywords with OR operators."""
        keywords = self.expanded_keyword_set()
        return f"{self.seed_word} OR " + " OR ".join(keywords)
    

def build_argument_parser(add_help: bool = True)-> argparse.ArgumentParser:
    """
    Build the argument parser for the keyword expander script.

    Args:
        add_help: Whether to add the default help argument

    Returns:
        Argument parser instance
    """
    parser = argparse.ArgumentParser(
    description="Expand keywords using PMI analysis on a text corpus.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
    add_help=add_help)
    
    parser.add_argument(
        "--seed-word",
        type=str,
        required=True,
        help="The seed keyword to expand from (e.g., 'healthcare').",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=PMIKeywordExpander.TOP_KEYWORDS,
        help="Number of top keywords to return based on PMI scores.",
    )
    return parser

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    return build_argument_parser().parse_args()

def main() -> None:
    """Main function to run the keyword expander."""
    args = parse_args()
    expander = PMIKeywordExpander(seed_word=args.seed_word)
    expander.TOP_KEYWORDS = args.top_n

    if not expander.load():
        logger.error("Failed to load data. Exiting.")
        return

    logger.info("Preprocessing text data...")
    expander.preprocess()

    logger.info("Counting document frequencies...")
    expander.doc_frequency_count()

    logger.info("Calculating PMI scores...")
    expander.PMI_calc()

    logger.info("Ranking keywords...")
    expander.ranking()

    expanded_keywords = expander.expanded_keyword_set()
    search_query = expander.generate_search_query()

    logger.info(f"Expanded Keywords: {expanded_keywords}")
    logger.info(f"Generated Search Query: {search_query}")
    return search_query

if __name__ == "__main__":
    main()
