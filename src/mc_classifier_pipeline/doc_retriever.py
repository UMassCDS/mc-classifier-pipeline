import argparse
import datetime as dt
import json
import logging
import os
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Optional

import mediacloud.api
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from . import utils

# Configure logging
utils.configure_logging()
logger = logging.getLogger(__name__)


class ArticleStatus(str, Enum):
    SUCCESS = "success"
    FAILED_NO_TEXT = "failed_no_text"
    UNKNOWN = "unknown"


# Configuration and constants
load_dotenv()
MC_API_KEY = os.getenv("MC_API_KEY")

RAW_ARTICLES_DIR = Path("data/raw_articles")
FAILED_URLS_LOG = Path("data/failed_urls.txt")
ARTICLES_INDEX_FILE = Path("data/articles_index.json")

# Initialize Media Cloud API
SEARCH_API = None
try:
    SEARCH_API = mediacloud.api.SearchApi(MC_API_KEY)
    logger.info("Media Cloud API initialized.")
except Exception as e:
    logger.error(f"Could not initialize Media Cloud API. Check your key. Error: {e}.")
    logger.warning("Article fetching from Media Cloud will be skipped.")

# Helper Functions


def load_articles_index(index_file: Path) -> dict:
    """
    Load the persistent index of retrieved articles.

    Args:
        index_file: Path to the articles index file

    Returns:
        Dictionary mapping story IDs to article metadata
    """
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading articles index: {e}")
            return {}
    return {}


def save_articles_index(index: dict, index_file: Path):
    """
    Save the persistent index of retrieved articles.

    Args:
        index: Dictionary mapping story IDs to article metadata
        index_file: Path to the articles index file
    """
    try:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False)
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
    status = article_info.get("status", ArticleStatus.UNKNOWN)
    text_length = article_info.get("text_length", 0)

    return status == ArticleStatus.SUCCESS and text_length > 0


def search_mediacloud_by_query(
    query: str,
    start_date: Optional[dt.date] = None,
    end_date: Optional[dt.date] = None,
    limit: int = 100,
    articles_index: Optional[dict] = None,
    raw_articles_dir: Optional[Path] = None,
    collection_ids: Optional[list[int]] = None,
) -> list:
    """
    Search Media Cloud for articles using a query string, avoiding re-retrieval of existing articles.

    Args:
        query: Search query string
        start_date: Start date for search
        end_date: End date for search
        limit: Maximum number of results to return
        articles_index: Dictionary of already retrieved articles
        raw_articles_dir: Directory where existing articles are stored

    Returns:
        List of article dictionaries with URLs and metadata
    """
    if not SEARCH_API:
        logger.error("Media Cloud API not initialized. Cannot search.")
        return []

    logger.info(f"Searching Media Cloud for: '{query}'")
    logger.info(f"Date range: {start_date} to {end_date}")

    articles = []
    new_articles = 0
    existing_articles = 0

    try:
        default_start = dt.date(2020, 1, 1)
        default_end = dt.date.today()
        actual_start = start_date if start_date is not None else default_start
        actual_end = end_date if end_date is not None else default_end
        if collection_ids:
            logger.info(f"Query restricted to {len(collection_ids)}(s) collections with ids: {collection_ids}")
            results = SEARCH_API.story_list(query, actual_start, actual_end, collection_ids=collection_ids)
        else:
            results = SEARCH_API.story_list(query, actual_start, actual_end)
        if results and len(results[0]) > 0:
            stories = results[0]
            logger.info(f"Found {len(stories)} stories")

            # Iterate through all stories, but only collect up to 'limit' new articles
            for story in tqdm(stories, desc=f"Processing stories (max {limit})"):
                if new_articles >= limit:
                    break
                story_id = story["id"]

                # Check if article is already retrieved before making API call
                if articles_index and is_article_retrieved(story_id, articles_index):
                    existing_articles += 1
                    filename = f"{story_id}.json"
                    filepath = Path(raw_articles_dir or RAW_ARTICLES_DIR) / filename

                    if filepath.exists():
                        try:
                            with open(filepath, "r", encoding="utf-8") as f:
                                existing_article = json.load(f)
                                articles.append(existing_article)
                        except Exception as e:
                            logger.error(f"Error loading existing article {story_id}: {e}")
                    continue

                # Only fetch article data if not already retrieved
                article_data = SEARCH_API.story(story_id)
                url = article_data.get("url", "")

                if not url:
                    continue

                new_articles += 1
                article_info = {
                    "url": url,
                    "title": article_data.get("title", ""),
                    "text": article_data.get("text", ""),
                    "source": "mediacloud_query",
                    "retrieved_at": dt.datetime.now().isoformat(),
                    "status": (ArticleStatus.SUCCESS if article_data.get("text") else ArticleStatus.FAILED_NO_TEXT),
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


def save_articles_from_query(
    articles: list,
    raw_articles_dir: Path,
    failed_urls_log: Path,
    articles_index: Optional[dict] = None,
):
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
        status = article.get("status", ArticleStatus.UNKNOWN)

        if not url or not story_id:
            continue

        is_new = articles_index is None or story_id not in articles_index
        filename = f"{story_id}.json"
        filepath = raw_articles_dir.joinpath(filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(article, f, ensure_ascii=False)

            if articles_index is not None:
                articles_index[story_id] = {
                    "filename": filename,
                    "filepath": str(filepath),
                    "status": status,
                    "retrieved_at": article.get("retrieved_at", ""),
                    "title": article.get("title", ""),
                    "text_length": len(article.get("text", "")),
                    "query": article.get("query", ""),
                    "url": url,
                    "publish_date": article.get("publish_date", ""),
                }

            if status != ArticleStatus.SUCCESS:
                failed_urls.append(url)
            elif is_new:
                new_articles += 1

        except Exception as e:
            logger.error(f"Error saving {story_id}: {e}")
            failed_urls.append(url)

    if failed_urls:
        with open(failed_urls_log, "w") as f:
            for url in failed_urls:
                f.write(f"{url}\n")
        logger.info(f"Failed URLs logged to {failed_urls_log}")

    logger.info(f"Saving complete. {new_articles} new articles saved, {len(failed_urls)} articles failed.")


def analyze_search_results(articles: list):
    """
    Calculates and logs summary statistics for the retrieved articles.

    This includes overall success rate, a breakdown of article statuses,
    text length statistics, and a language breakdown.

    Args:
        articles: List of article dictionaries
    """
    if not articles:
        logger.info("No articles to analyze.")
        return

    total_articles = len(articles)

    status_counts = Counter(a.get("status", ArticleStatus.UNKNOWN) for a in articles)
    successful = status_counts.get(ArticleStatus.SUCCESS, 0)
    failed = total_articles - successful

    logger.info("=== Search Results Analysis ===")
    logger.info(f"Total articles found: {total_articles}")
    logger.info(f"Successful retrievals: {successful}")
    logger.info(f"Failed retrievals: {failed}")
    if total_articles > 0:
        logger.info(f"Success rate: {successful / total_articles * 100:.1f}%")

    logger.info("Status breakdown:")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")

    text_lengths = [len(article.get("text", "")) for article in articles if article.get("text")]
    if text_lengths:
        logger.info("Text length statistics:")
        logger.info(f"  Mean: {sum(text_lengths) / len(text_lengths):.0f} characters")
        logger.info(f"  Median: {sorted(text_lengths)[len(text_lengths) // 2]:.0f} characters")
        logger.info(f"  Min: {min(text_lengths):.0f} characters")
        logger.info(f"  Max: {max(text_lengths):.0f} characters")

    language_counts = Counter(a.get("language", "unknown") for a in articles)
    logger.info("Language breakdown:")
    for language, count in language_counts.items():
        logger.info(f"  {language}: {count}")


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """
    Build the argument parser for the Label Studio uploader.

    Args:
        add_help: Whether to add the default help argument

    Returns:
        Argument parser instance
    """
    parser = argparse.ArgumentParser(
        description="Media Cloud query search and article processing pipeline with persistent tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Search by query and save articles
            python src/mc_classifier_pipeline/doc_retriever.py --query "election" --start-date 2024-12-01 --end-date 2024-12-31 --limit 50 --output data/search_results.csv
            # Search by query and save articles in Label Studio JSON format
            python -m mc_classifier_pipeline.doc_retriever --query "election" --start-date 2024-12-01 --end-date 2024-12-31 --limit 50 --label-studio-tasks data/labelstudio_tasks.json
            # Search by query and save articles from a collection in Label Studio JSON format
            python -m mc_classifier_pipeline.doc_retriever --query "election" --start-date 2024-12-01 --end-date 2024-12-31 --limit 50 --collection-ids 34412234 34412118 --label-studio-tasks data/labelstudio_tasks.json
        """,
        add_help=add_help,
    )
    parser.add_argument("--query", type=str, required=True, help="Search query for Media Cloud API")

    parser.add_argument("--output", type=Path, help="Optional: Output CSV file path")

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_ARTICLES_DIR,
        help="Directory to store raw article JSON files",
    )

    parser.add_argument(
        "--failed-log",
        type=Path,
        default=FAILED_URLS_LOG,
        help="Path to log failed URLs",
    )

    parser.add_argument(
        "--index-file",
        type=Path,
        default=ARTICLES_INDEX_FILE,
        help="Path to articles index file",
    )

    parser.add_argument("--no-save-json", action="store_true", help="Skip saving individual JSON files")

    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of already retrieved articles",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of results for query search",
    )

    parser.add_argument(
        "--start-date",
        required=True,
        type=str,
        help="Start date for query search (YYYY-MM-DD format)",
    )

    parser.add_argument(
        "--end-date",
        required=True,
        type=str,
        help="End date for query search (YYYY-MM-DD format)",
    )

    parser.add_argument(
        "--collection-ids",
        nargs="*",
        type=int,
        help=("List of collection IDs to limit the search to (ID1 ID2 ... format)"),
    )

    # json formatted for label studio
    parser.add_argument(
        "--label-studio-tasks",
        type=Path,
        default=Path("data/labelstudio_tasks.json"),
        help="Path to the Label Studio tasks JSON file",
    )

    return parser


def parse_arguments():
    """Parse command line arguments."""
    return build_arg_parser().parse_args()


def main(args: Optional[argparse.Namespace] = None):
    """
    Main function to run the Media Cloud query search and article processing pipeline.
    This function parses command-line arguments, loads a persistent index of articles,
    searches Media Cloud for new articles based on a query, saves the results,
    and provides a summary analysis.
    """
    if args is None:
        args = parse_arguments()

    # Default to output Label Studio Json if not specified
    default_label_studio_path = Path("data/labelstudio_tasks.json")
    if not args.output and not args.label_studio_tasks:
        logger.warning(
            f"No output format specified (--output or --label-studio-tasks). "
            f"Defaulting to Label Studio JSON at {default_label_studio_path}"
        )
        args.label_studio_tasks = default_label_studio_path

    logger.info("Starting Media Cloud query search and article processing pipeline...")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.failed_log.parent.mkdir(parents=True, exist_ok=True)
    args.index_file.parent.mkdir(parents=True, exist_ok=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.label_studio_tasks:
        args.label_studio_tasks.parent.mkdir(parents=True, exist_ok=True)

    articles_index = {}
    if not args.force_reprocess:
        articles_index = load_articles_index(args.index_file)
        logger.info(f"Loaded index with {len(articles_index)} existing articles")

    try:
        start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    except ValueError as e:
        logger.error(f"Invalid date format: {e}. Use YYYY-MM-DD format.")
        return

    articles = search_mediacloud_by_query(
        args.query, start_date, end_date, args.limit, articles_index, args.raw_dir, collection_ids=args.collection_ids
    )

    if articles:
        if not args.no_save_json:
            save_articles_from_query(articles, args.raw_dir, args.failed_log, articles_index)
            save_articles_index(articles_index, args.index_file)
        if args.output:
            logger.info("Creating output CSV...")
            df = pd.DataFrame(articles)
            df.to_csv(args.output, index=False)
            logger.info(f"Search results saved to {args.output}")

        # if requested, then write json in label studio format
        if args.label_studio_tasks:
            tasks = []
            for article in articles:
                text = article.get("text", "").strip()
                story_id = article["story_id"]
                if not text:
                    continue
                data = {"text": text, "story_id": story_id}
                # optionally include metadata fields(Arav was storing these)
                for key in ("title", "url", "language", "publish_date"):
                    if article.get(key):
                        data[key] = article[key]
                tasks.append({"data": data, "external_id": f"mc_story_{story_id}"})

            if not tasks:
                logger.warning("No valid tasks to write, all articles were empty or invalid.")
            else:
                with open(args.label_studio_tasks, "w", encoding="utf-8") as f:
                    json.dump(tasks, f, ensure_ascii=False, indent=2)
                logger.info(f"Label Studio task file saved to {args.label_studio_tasks}")

        analyze_search_results(articles)
    else:
        logger.warning("No new articles found for the query.")
        logger.warning("No new articles found for the query.")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
