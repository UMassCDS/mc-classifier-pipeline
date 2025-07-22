import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import math


class PMIKeywordExpander:
    """
    A class to expand keywords using Pointwise Mutual Information (PMI) analysis.
    Analyzes text corpus to find words most associated with a seed keyword.
    """

    CSV_PATH = "data/search_results.csv"  # currently has results from running python src/mc_classifier_pipeline/doc_retriever.py --query "healthcare" --start-date 2024-12-01 --end-date 2025-07-01 --limit 1000 --output data/search_results.csv
    TOP_KEYWORDS = 20

    def __init__(self, seed_word="health"):
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


# Create an instance
health_expander = PMIKeywordExpander()

# Test the load method
success = health_expander.load()
print(f"Loading successful: {success}")

# Check what was loaded
if success:
    print(f"Shape of data: {health_expander.article_df.shape}")
    print(f"Columns: {health_expander.article_df.columns.tolist()}")
    print(f"First few rows:\n{health_expander.article_df.head()}")

health_expander.preprocess()

# PREPROCESSING
print(f"Number of English articles: {len(health_expander.article_df)}")
print(f"Columns after preprocessing: {health_expander.article_df.columns.tolist()}")

# Look at some sample tokens
print("\nSample tokens from first article:")
print(health_expander.article_df["tokens"].iloc[0][:20])  # First 20 tokens

# Check token statistics
token_lengths = health_expander.article_df["tokens"].apply(len)
print("\nToken count statistics:")
print(f"Average tokens per article: {token_lengths.mean():.1f}")
print(f"Min tokens: {token_lengths.min()}")
print(f"Max tokens: {token_lengths.max()}")

# Check if 'climate' appears in tokens (since that's our seed word)
healthcare_count = sum(1 for tokens in health_expander.article_df["tokens"] if "healthcare" in tokens)
print(f"\nArticles containing 'healthcare': {healthcare_count}")

health_expander.doc_frequency_count()

# PMI FREQUENCY COUNTS
print(f"Total documents: {health_expander.total_documents}")
print(f"Documents containing '{health_expander.seed_word}': {health_expander.seed_doc_count}")
print(f"Total unique words: {len(health_expander.doc_frequencies)}")

# Check some co-occurrence counts
print(f"\nTop words co-occurring with '{health_expander.seed_word}':")
print(health_expander.cooccurrence_counts.most_common(10))

# Check that healthcare co-occurs with itself correctly
print(
    f"\n'{health_expander.seed_word}' co-occurs with itself in {health_expander.cooccurrence_counts[health_expander.seed_word]} documents"
)

# Verify this matches the seed_doc_count
print(f"This should match seed_doc_count: {health_expander.seed_doc_count}")

# PMI calculation
health_expander.PMI_calc()

print(f"Total words with PMI scores: {len(health_expander.pmi_scores)}")
print("Sample PMI scores:")
for word, score in list(health_expander.pmi_scores.items())[:10]:
    print(f"  {word}: {score:.4f}")

health_expander.ranking()
print(f"Top {health_expander.TOP_KEYWORDS} keywords by PMI:")
for word, score in health_expander.top_keywords:
    print(f"  {word}: {score:.4f}")

keywords = health_expander.expanded_keyword_set()
print(f"Final expanded keyword set: {keywords}")

query = health_expander.generate_search_query()
print(f"Search query: {query}")
