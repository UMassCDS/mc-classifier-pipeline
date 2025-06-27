import pandas as pd
import os
import json
import datetime as dt
import argparse
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import mediacloud.api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data/doc_retriever.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configuration and constants
load_dotenv()
MC_API_KEY = os.getenv("MC_API_KEY")
RAW_ARTICLES_DIR = "data/raw_articles"
FAILED_URLS_LOG = "data/failed_urls.txt"
ARTICLES_INDEX_FILE = "data/articles_index.json"

# Initialize Media Cloud API
search_api = None
try:
    search_api = mediacloud.api.SearchApi(MC_API_KEY)
    logger.info("Media Cloud API initialized.")
except Exception as e:
    logger.error(f"Could not initialize Media Cloud API. Check your key. Error: {e}.")
    logger.warning("Article fetching from Media Cloud will be skipped.")

# Helper Functions


def load_articles_index(index_file: str) -> dict:
    """
    Load the persistent index of retrieved articles.

    Args:
        index_file: Path to the articles index file

    Returns:
        Dictionary mapping story IDs to article metadata
    """
    if os.path.exists(index_file):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading articles index: {e}")
            return {}
    return {}


def save_articles_index(index: dict, index_file: str):
    """
    Save the persistent index of retrieved articles.

    Args:
        index: Dictionary mapping story IDs to article metadata
        index_file: Path to the articles index file
    """
    try:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving articles index: {e}")


def is_article_retrieved(story_id: str, articles_index: dict) -> bool:
    """
    Check if an article has already been retrieved.

    Args:
        story_id: Media Cloud story ID
        articles_index: Dictionary of retrieved articles

    Returns:
        True if article exists and has valid content
    """
    if story_id not in articles_index:
        return False

    article_info = articles_index[story_id]
    status = article_info.get("status", "unknown")
    text_length = article_info.get("text_length", 0)

    # Consider retrieved if status is success and has text content
    return status == "success" and text_length > 0


def search_mediacloud_by_query(
    query: str, start_date: dt.date = None, end_date: dt.date = None, limit: int = 100, articles_index: dict = None
) -> list:
    """
    Search Media Cloud for articles using a query string, avoiding re-retrieval of existing articles.

    Args:
        query: Search query string
        start_date: Start date for search
        end_date: End date for search
        limit: Maximum number of results to return
        articles_index: Dictionary of already retrieved articles

    Returns:
        List of article dictionaries with URLs and metadata
    """
    if not search_api:
        logger.error("Media Cloud API not initialized. Cannot search.")
        return []

    logger.info(f"Searching Media Cloud for: '{query}'")
    logger.info(f"Date range: {start_date} to {end_date}")

    articles = []
    new_articles = 0
    existing_articles = 0

    try:
        results = search_api.story_list(query, start_date, end_date)
        if results and len(results[0]) > 0:
            stories = results[0]
            logger.info(f"Found {len(stories)} stories")

            for story in tqdm(stories[:limit], desc=f"Processing stories (max {limit})"):
                story_id = story["id"]
                article_data = search_api.story(story_id)
                url = article_data.get("url", "")

                if not url:
                    continue

                # Check if article already exists
                if articles_index and is_article_retrieved(story_id, articles_index):
                    existing_articles += 1
                    # Load existing article data
                    filename = f"{story_id}.json"
                    filepath = os.path.join(RAW_ARTICLES_DIR, filename)

                    if os.path.exists(filepath):
                        try:
                            with open(filepath, "r", encoding="utf-8") as f:
                                existing_article = json.load(f)
                                articles.append(existing_article)
                        except Exception as e:
                            logger.error(f"Error loading existing article {story_id}: {e}")
                    continue

                # New article - process it
                new_articles += 1
                article_info = {
                    "url": url,
                    "title": article_data.get("title", ""),
                    "text": article_data.get("text", ""),
                    "source": "mediacloud_query",
                    "retrieved_at": dt.datetime.now().isoformat(),
                    "status": "success" if article_data.get("text") else "failed_no_text",
                    "story_id": story_id,
                    "publish_date": article_data.get("publish_date", ""),
                    "media_id": article_data.get("media_id", ""),
                    "language": article_data.get("language", ""),
                    "query": query,
                }
                articles.append(article_info)
        else:
            logger.warning("No results found for the query.")
    except Exception as e:
        logger.error(f"Error searching Media Cloud: {e}")

    logger.info(f"Processing complete: {new_articles} new articles, {existing_articles} existing articles")
    return articles


def save_articles_from_query(articles: list, raw_articles_dir: str, failed_urls_log: str, articles_index: dict = None):
    """
    Save articles retrieved from Media Cloud query to JSON files and update index.

    Args:
        articles: List of article dictionaries
        raw_articles_dir: Directory to save article JSON files
        failed_urls_log: Path to log failed URLs
        articles_index: Dictionary of retrieved articles to update
    """
    if not articles:
        logger.info("No articles to save.")
        return

    logger.info(f"Saving {len(articles)} articles...")
    failed_urls = []
    new_articles = 0

    for article in tqdm(articles, desc="Saving articles"):
        url = article.get("url", "")
        story_id = article.get("story_id", "")

        if not url or not story_id:
            continue

        # Check if this is a new article (not in index)
        is_new = articles_index is None or story_id not in articles_index

        # Generate filename based on story ID
        filename = f"{story_id}.json"
        filepath = os.path.join(raw_articles_dir, filename)

        # Save article data
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(article, f, ensure_ascii=False, indent=2)

            # Update index
            if articles_index is not None:
                articles_index[story_id] = {
                    "filename": filename,
                    "filepath": filepath,
                    "status": article.get("status", "unknown"),
                    "retrieved_at": article.get("retrieved_at", ""),
                    "title": article.get("title", ""),
                    "text_length": len(article.get("text", "")),
                    "query": article.get("query", ""),
                    "url": url,
                    "publish_date": article.get("publish_date", ""),
                }

            if article["status"].startswith("failed_"):
                failed_urls.append(url)
            elif is_new:
                new_articles += 1

        except Exception as e:
            logger.error(f"Error saving {story_id}: {e}")
            failed_urls.append(url)

    # Log failed URLs
    if failed_urls:
        with open(failed_urls_log, "w") as f:
            for url in failed_urls:
                f.write(f"{url}\n")
        logger.info(f"Failed URLs logged to {failed_urls_log}")

    logger.info(f"Saving complete. {new_articles} new articles saved, {len(failed_urls)} articles failed.")


def load_all_article_json_data(json_dir: str) -> dict:
    """
    Load all article data from JSON files in the specified directory.

    Args:
        json_dir: Directory containing JSON files

    Returns:
        Dictionary mapping story IDs to article data
    """
    articles_data = {}

    if not os.path.exists(json_dir):
        logger.warning(f"Directory {json_dir} does not exist.")
        return articles_data

    logger.info(f"Loading article data from {json_dir}...")

    for filename in tqdm(os.listdir(json_dir), desc="Loading JSON files"):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(json_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                story_id = data.get("story_id", "")
                if story_id:
                    articles_data[story_id] = data
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            continue

    logger.info(f"Loaded data for {len(articles_data)} articles.")
    return articles_data


def analyze_search_results(articles: list):
    """
    Analyze the results of the search operation.

    Args:
        articles: List of article dictionaries
    """
    if not articles:
        logger.info("No articles to analyze.")
        return

    total_articles = len(articles)
    successful = len([a for a in articles if a["status"] == "success"])
    failed = total_articles - successful

    logger.info("\n=== Search Results Analysis ===")
    logger.info(f"Total articles found: {total_articles}")
    logger.info(f"Successful retrievals: {successful}")
    logger.info(f"Failed retrievals: {failed}")
    logger.info(f"Success rate: {successful / total_articles * 100:.1f}%")

    # Status breakdown
    status_counts = {}
    for article in articles:
        status = article.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    logger.info("\nStatus breakdown:")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")

    # Text length statistics
    text_lengths = [len(article.get("text", "")) for article in articles if article.get("text")]
    if text_lengths:
        logger.info("\nText length statistics:")
        logger.info(f"  Mean: {sum(text_lengths) / len(text_lengths):.0f} characters")
        logger.info(f"  Median: {sorted(text_lengths)[len(text_lengths) // 2]:.0f} characters")
        logger.info(f"  Min: {min(text_lengths):.0f} characters")
        logger.info(f"  Max: {max(text_lengths):.0f} characters")

    # Language breakdown
    language_counts = {}
    for article in articles:
        language = article.get("language", "unknown")
        language_counts[language] = language_counts.get(language, 0) + 1

    logger.info("\nLanguage breakdown:")
    for language, count in language_counts.items():
        logger.info(f"  {language}: {count}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Media Cloud query search and article processing pipeline with persistent tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Search by query and save articles
            python src/mc_classifier_pipeline/doc_retriever.py --query "election" 
                --start-date 2024-12-01 --end-date 2024-12-31 --limit 50 --output data/search_results.csv
        """,
    )

    # Required query argument
    parser.add_argument("--query", type=str, required=True, help="Search query for Media Cloud API")

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="data/search_results.csv",
        help="Output CSV file path (default: search_results.csv)",
    )

    # Directory options
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw_articles",
        help="Directory to store raw article JSON files (default: data/raw_articles)",
    )

    parser.add_argument(
        "--failed-log",
        type=str,
        default="data/failed_urls.txt",
        help="Path to log failed URLs (default: data/failed_urls.txt)",
    )

    parser.add_argument(
        "--index-file",
        type=str,
        default="data/articles_index.json",
        help="Path to articles index file (default: data/articles_index.json)",
    )

    # Processing options
    parser.add_argument(
        "--no-save-json", action="store_true", help="Skip saving individual JSON files (only create CSV output)"
    )

    parser.add_argument(
        "--force-reprocess", action="store_true", help="Force reprocessing of already retrieved articles"
    )

    # Query-specific options
    parser.add_argument(
        "--limit", type=int, default=100, help="Maximum number of results for query search (default: 100)"
    )

    parser.add_argument(
        "--start-date", required=True, type=str, help="Start date for query search (YYYY-MM-DD format)"
    )

    parser.add_argument(
        "--end-date", required=True, type=str, help="End date for query search (YYYY-MM-DD format, default: today)"
    )

    return parser.parse_args()


def main():
    """
    Main function to run the Media Cloud query search and article processing pipeline.
    """
    args = parse_arguments()

    logger.info("Starting Media Cloud query search and article processing pipeline...")

    # Ensure directories exist
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.failed_log), exist_ok=True)
    os.makedirs(os.path.dirname(args.index_file), exist_ok=True)

    # Load articles index for persistent tracking
    articles_index = {}
    if not args.force_reprocess:
        articles_index = load_articles_index(args.index_file)
        logger.info(f"Loaded index with {len(articles_index)} existing articles")

    # Parse dates
    start_date = None
    end_date = None

    if args.start_date:
        try:
            start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid start date format: {args.start_date}. Use YYYY-MM-DD format.")
            return

    if args.end_date:
        try:
            end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid end date format: {args.end_date}. Use YYYY-MM-DD format.")
            return

    # Search Media Cloud
    articles = search_mediacloud_by_query(args.query, start_date, end_date, args.limit, articles_index)

    if articles:
        # Save articles as JSON files if requested
        if not args.no_save_json:
            save_articles_from_query(articles, args.raw_dir, args.failed_log, articles_index)
            # Save updated index
            save_articles_index(articles_index, args.index_file)

        # Create output CSV with search results
        logger.info("\nCreating output CSV...")
        df = pd.DataFrame(articles)
        df.to_csv(args.output, index=False)
        logger.info(f"Search results saved to {args.output}")

        # Analyze results
        analyze_search_results(articles)
    else:
        logger.warning("No new articles found for the query.")

    logger.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
