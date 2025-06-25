import pandas as pd
import os
import json
import hashlib
import datetime as dt
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv
from tqdm import tqdm
import mediacloud.api

# Configuration and constants
load_dotenv()
MC_API_KEY = os.getenv("MC_API_KEY")
RAW_ARTICLES_DIR = "raw_articles_data"
CSV_INPUT_PATH = 'data/cn-gmmp-story-pull-v1.csv'
MC_URLS_CSV_PATH = 'data/cn-gmmp-mediacloud-urls.csv'
FAILED_URLS_LOG = 'failed_urls.txt'
MERGED_OUTPUT_CSV = "merged_articles.csv"

# Ensure directories exist
os.makedirs(RAW_ARTICLES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_INPUT_PATH), exist_ok=True) 

# Initialize Media Cloud API
search_api = None
try:
    search_api = mediacloud.api.SearchApi(MC_API_KEY)
    print("Media Cloud API initialized.")
except Exception as e:
    print(f"Could not initialize Media Cloud API. Check your key. Error: {e}.")
    print("Article fetching from Media Cloud will be skipped.")

# Helper Functions for URL Normalization and Hashing 

def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent comparison by:
    - Converting to lowercase
    - Removing trailing slashes
    - Removing common tracking parameters
    - Standardizing protocol to https if missing
    - Removing fragments
    """
    if not url:
        return ""

    parsed = urlparse(url.lower())

    # Remove common tracking parameters
    tracking_params = {
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
        'fbclid', 'gclid', 'ref', 'source', 'campaign_id', '_ga', 'mc_cid'
    }

    if parsed.query:
        query_pairs = [pair for pair in parsed.query.split('&')
                       if pair.split('=')[0] not in tracking_params]
        clean_query = '&'.join(query_pairs) if query_pairs else ''
    else:
        clean_query = ''

    # Remove trailing slash from path, but preserve a single '/' for root
    clean_path = parsed.path.rstrip('/')
    if not clean_path:
        clean_path = '/' # Ensure root path is not empty if it was just a slash

    # Reconstruct URL
    normalized_scheme = parsed.scheme or 'https' # Default to https
    normalized_netloc = parsed.netloc

    normalized = urlunparse((
        normalized_scheme,
        normalized_netloc,
        clean_path,
        parsed.params,
        clean_query,
        ''  # Remove fragment
    ))

    return normalized

def get_url_hash(url: str) -> str:
    """Generate consistent hash for normalized URL."""
    normalized_url = normalize_url(url)
    return hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()

# Article Fetching from Media Cloud

def _fetch_article_from_mediacloud(url: str) -> tuple[str, str, str]:
    """
    Attempts to fetch article text and title from Media Cloud for a given URL.
    Returns (article_title, article_text, status).
    """
    if not search_api:
        return "", "", "failed_mediacloud_api_not_initialized"

    article_text = ""
    article_title = ""
    status = "failed_mediacloud_no_results"

    # Media Cloud API often expects `url:http(s)\://domain.com/path/*`
    # Replace : with \: for query string
    escaped_url = url.replace(':', '\\:')

    my_query = f'url:{escaped_url}'
    start_date = dt.date(2025, 1, 1) # Use a wider date range for more robust search
    end_date = dt.date.today()

    try:
        results = search_api.story_list(my_query, start_date, end_date) 
        if results and len(results[0]) > 0:
            story_id = results[0][0]['id']
            article_data = search_api.story(story_id)
            article_title = article_data.get('title', '')
            article_text = article_data.get('text', '')
            status = "success" if article_text else "failed_mediacloud_no_text"
        else:
            status = "failed_mediacloud_no_results"
    except Exception as e:
        status = f"failed_mediacloud_exception: {type(e).__name__}: {e}"

    return article_title, article_text, status

def get_article_text_with_retries(url: str) -> dict:
    """
    Retrieves the full text of an article from a URL using Media Cloud, with retries.

    Args:
        url: The URL of the article to fetch.

    Returns:
        A dictionary containing the URL, fetched text, and other metadata.
    """
    original_url = url
    article_title = ""
    article_text = ""
    source = "mediacloud"
    status = "not_attempted"

    # Attempt 1: Original URL
    article_title, article_text, status = _fetch_article_from_mediacloud(original_url)

    # Attempt 2: If first attempt failed to get text, try with trailing slash
    if not article_text or status.startswith("failed_"):
        # Append a trailing slash if not already present
        retry_url = original_url if original_url.endswith('/') else original_url + '/'
        if retry_url != original_url: # Only retry if the URL actually changed
            # print(f"Retrying with trailing slash: {retry_url}")
            retry_title, retry_text, retry_status = _fetch_article_from_mediacloud(retry_url)
            if retry_text: # If retry was successful, use its data
                article_title = retry_title
                article_text = retry_text
                status = retry_status
            else: # If retry also failed, keep the original status (or update if retry status is more specific)
                status = retry_status if retry_status.startswith("failed_mediacloud") else status

    # Final status check if no text was retrieved
    if not article_text:
        status = status if status.startswith("failed_") else "failed_no_text_retrieved"

    return {
        "url": original_url,
        "title": article_title,
        "text": article_text,
        "source": source,
        "retrieved_at": dt.datetime.now().isoformat(),
        "status": status,
        "normalized_url": normalize_url(original_url)
    }

# Metadata Management and URL Processing

def load_processed_urls_metadata() -> dict:
    """
    Load metadata of all processed URLs from existing files into memory for fast lookup.
    Returns dict mapping normalized URLs to their file info.
    """
    processed_urls = {}
    if not os.path.exists(RAW_ARTICLES_DIR):
        os.makedirs(RAW_ARTICLES_DIR) # Ensure directory exists before scanning
        return processed_urls

    print("Loading already processed URLs metadata...")
    for filename in tqdm(os.listdir(RAW_ARTICLES_DIR), desc="Scanning files"):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(RAW_ARTICLES_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                original_url = data.get('url', '')
                if original_url:
                    normalized_url = normalize_url(original_url)
                    processed_urls[normalized_url] = {
                        'filename': filename,
                        'filepath': filepath,
                        'original_url': original_url,
                        'status': data.get('status', 'unknown'),
                        'retrieved_at': data.get('retrieved_at', ''),
                        'has_text': bool(data.get('text', '').strip())
                    }
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Warning: Could not read {filename}: {e}")
            continue

    print(f"Loaded metadata for {len(processed_urls)} already processed URLs")
    return processed_urls

def identify_urls_for_reprocessing(processed_urls: dict) -> list:
    """
    Identify URLs that failed processing or have no text content from the loaded metadata.
    Returns list of original URLs that could be reprocessed.
    """
    reprocess_candidates = []
    for normalized_url, url_info in processed_urls.items():
        # Reprocess if it has no text or the status indicates a failure
        if not url_info['has_text'] or url_info['status'].startswith('failed_'):
            reprocess_candidates.append(url_info['original_url'])
    return reprocess_candidates

def fetch_and_save_articles(urls_to_process: list, force_reprocess: bool = False):
    """
    Processes a list of URLs, fetches their text, and saves each to a unique file.
    Args:
        urls_to_process: List of URLs to process.
        force_reprocess: If True, reprocess URLs even if they were previously successful.
    """
    processed_urls_metadata = load_processed_urls_metadata()

    failed_this_run = []
    skipped_this_run = []
    successfully_processed_this_run = 0

    # Deduplicate input URLs while preserving order
    unique_urls = []
    seen_normalized_input = set()
    for url in urls_to_process:
        normalized = normalize_url(url)
        if normalized not in seen_normalized_input:
            unique_urls.append(url)
            seen_normalized_input.add(normalized)
        else:
            print(f"Skipping duplicate URL in input list: {url}")

    print(f"Processing {len(unique_urls)} unique URLs.")

    # Clear failed_urls.txt at the start of a fresh processing run
    if not force_reprocess and os.path.exists(FAILED_URLS_LOG):
        open(FAILED_URLS_LOG, 'w').close() # Clear file

    for original_url in tqdm(unique_urls, desc="Fetching Articles", unit="article"):
        normalized_url = normalize_url(original_url)
        url_hash = get_url_hash(original_url)
        filename = f"{url_hash}.json"
        filepath = os.path.join(RAW_ARTICLES_DIR, filename)

        # Check if URL was already processed successfully and not forcing reprocess
        if normalized_url in processed_urls_metadata and not force_reprocess:
            existing_info = processed_urls_metadata[normalized_url]
            if existing_info['has_text'] and not existing_info['status'].startswith('failed_'):
                # print(f"Skipping already successfully processed URL: {original_url}")
                skipped_this_run.append(original_url)
                continue

        # If forcing reprocess and file exists, remove it first
        if force_reprocess and os.path.exists(filepath):
            os.remove(filepath)
            # print(f"Removed existing file for reprocessing: {original_url}")

        # Fetch article data
        article_data = get_article_text_with_retries(original_url)

        if article_data["text"]:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(article_data, f, ensure_ascii=False, indent=4)
                successfully_processed_this_run += 1
            except Exception as e:
                failed_this_run.append({
                    "url": original_url,
                    "reason": f"file_save_error: {type(e).__name__}: {e}",
                    "status": article_data.get("status", "unknown")
                })
                with open(FAILED_URLS_LOG, 'a', encoding='utf-8') as f:
                    f.write(original_url + '\n')
        else:
            failed_this_run.append({
                "url": original_url,
                "reason": article_data.get("status", "no_text_retrieved")
            })
            with open(FAILED_URLS_LOG, 'a', encoding='utf-8') as f:
                f.write(original_url + '\n')

    # Print summary
    print(f"\n--- Processing Summary ---")
    print(f"  Successfully fetched & saved: {successfully_processed_this_run}")
    print(f"  Skipped (already processed): {len(skipped_this_run)}")
    print(f"  Failed (this run): {len(failed_this_run)}")

    if failed_this_run:
        print(f"\nFirst 5 Failed URLs (details in {FAILED_URLS_LOG}):")
        for failed_info in failed_this_run[:5]:
            print(f"  URL: {failed_info['url']}")
            print(f"    Reason: {failed_info['reason']}")
        if len(failed_this_run) > 5:
            print(f"  ... and {len(failed_this_run) - 5} more.")

# Loading and Merging Article Data

def load_all_article_json_data(json_dir: str) -> dict:
    """
    Load all article JSON files and create a mapping from normalized URL to full article data.
    """
    articles = {}
    json_files = list(os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json'))

    print(f"Loading {len(json_files)} article files for merging...")

    for json_file in tqdm(json_files, desc="Loading articles"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                article_data = json.load(f)

            # Prioritize normalized_url from saved data, otherwise normalize on the fly
            normalized_url = article_data.get('normalized_url') or normalize_url(article_data.get('url', ''))
            if normalized_url:
                articles[normalized_url] = article_data

        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    print(f"Loaded full data for {len(articles)} articles.")
    return articles

def merge_csv_with_articles(csv_path: str, json_dir: str, output_path: str) -> pd.DataFrame:
    """
    Merge CSV data with article text from JSON files.
    """
    print(f"\n--- Starting Data Merging ({csv_path} + {json_dir}) ---")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV.")

    articles_full_data = load_all_article_json_data(json_dir)

    # Initialize new columns
    df['article_title'] = ''
    df['article_text'] = ''
    df['article_source'] = ''
    df['article_status'] = ''
    df['article_retrieved_at'] = ''
    df['has_article_text'] = False

    matched_count = 0
    unmatched_urls_in_csv = []

    print("Merging article data with DataFrame...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Merging data"):
        source_url = row.get('url')
        if pd.isna(source_url):
            df.at[idx, 'article_status'] = 'no_url_in_csv'
            continue

        normalized_source_url = normalize_url(source_url)
        article_data = articles_full_data.get(normalized_source_url)

        if article_data:
            df.at[idx, 'article_title'] = article_data.get('title', '')
            df.at[idx, 'article_text'] = article_data.get('text', '')
            df.at[idx, 'article_source'] = article_data.get('source', '')
            df.at[idx, 'article_status'] = article_data.get('status', '')
            df.at[idx, 'article_retrieved_at'] = article_data.get('retrieved_at', '')
            df.at[idx, 'has_article_text'] = bool(article_data.get('text', '').strip())
            matched_count += 1
        else:
            df.at[idx, 'article_status'] = 'not_found_in_json'
            unmatched_urls_in_csv.append(source_url)

    print(f"\n--- Merging Statistics ---")
    print(f"  Total rows in CSV: {len(df)}")
    print(f"  Matched with article data: {matched_count}")
    print(f"  Unmatched in JSON: {len(df) - matched_count}")
    print(f"  Rows with article text after merge: {df['has_article_text'].sum()}")

    if unmatched_urls_in_csv:
        print(f"\nFirst 5 URLs from CSV not found in JSON data:")
        for url in unmatched_urls_in_csv[:5]:
            print(f"  {url}")
        if len(unmatched_urls_in_csv) > 5:
            print(f"  ... and {len(unmatched_urls_in_csv) - 5} more.")

    if output_path:
        print(f"\nSaving merged data to {output_path}...")
        df.to_csv(output_path, index=False)
        print("Saved successfully!")

    return df

def analyze_merge_results(df: pd.DataFrame):
    """
    Analyze the results of the merge operation, providing insights into data quality.
    """
    print("\n" + "="*50)
    print("MERGE RESULTS ANALYSIS")
    print("="*50)

    total_rows = len(df)
    if total_rows == 0:
        print("No data to analyze.")
        return

    has_text_count = df['has_article_text'].sum()
    no_text_count = total_rows - has_text_count

    print(f"Total entries: {total_rows}")
    print(f"Entries with article text: {has_text_count} ({has_text_count/total_rows*100:.1f}%)")
    print(f"Entries without article text: {no_text_count} ({no_text_count/total_rows*100:.1f}%)")

    print(f"\nBreakdown by article_status:")
    status_counts = df['article_status'].value_counts(dropna=False) # Include NaN for completeness
    for status, count in status_counts.items():
        status_label = "NaN (No Status)" if status is None else status
        print(f"  - {status_label}: {count} ({count/total_rows*100:.1f}%)")

    # Text length statistics for articles that successfully got text
    text_lengths = df[df['has_article_text']]['article_text'].str.len()
    if not text_lengths.empty:
        print(f"\nArticle text length statistics (for entries with text):")
        print(f"  Mean: {text_lengths.mean():.0f} characters")
        print(f"  Median: {text_lengths.median():.0f} characters")
        print(f"  Min: {text_lengths.min()} characters")
        print(f"  Max: {text_lengths.max()} characters")
    else:
        print("\nNo articles with text found for length analysis.")

    # Show some examples of entries without text
    no_text_examples = df[~df['has_article_text']].head(5)
    if not no_text_examples.empty:
        print("\nFirst 5 entries without fetched article text:")
        for _, row in no_text_examples.iterrows():
            print(f"  URL: {row.get('url', 'N/A')}")
            print(f"  Status: {row.get('article_status', 'N/A')}")
            print(f"  Original source: {row.get('source', 'N/A')}\n")

# Main Execution Flow 
def main():
    """Main function to orchestrate the script execution."""

    # 1. Load original CSV and filter Media Cloud URLs
    print("Loading original story pull CSV...")
    df_original = pd.read_csv(CSV_INPUT_PATH)
    mediacloud_df = df_original[df_original['source'] == 'media-cloud']
    urls_to_fetch = mediacloud_df['url'].tolist()
    print(f"Found {len(urls_to_fetch)} Media Cloud URLs to process.")

    # Save filtered Media Cloud URLs to a separate CSV
    mediacloud_df.to_csv(MC_URLS_CSV_PATH, index=False)
    print(f"Media Cloud URLs saved to {MC_URLS_CSV_PATH}")

    # 2. Fetch and save articles
    print("\n--- Starting initial article fetching ---")
    fetch_and_save_articles(urls_to_fetch)
    print("--- Initial article fetching complete ---")

    # 3. Reprocess failed URLs (from the log file generated by the previous step)
    if os.path.exists(FAILED_URLS_LOG) and os.path.getsize(FAILED_URLS_LOG) > 0:
        with open(FAILED_URLS_LOG, 'r', encoding='utf-8') as f:
            failed_urls_from_log = [line.strip() for line in f if line.strip()]

        if failed_urls_from_log:
            print(f"\n--- Reprocessing {len(failed_urls_from_log)} previously failed URLs ---")
            # Clear the failed URLs log for the re-processing run
            open(FAILED_URLS_LOG, 'w').close()
            fetch_and_save_articles(failed_urls_from_log, force_reprocess=True)
            print("--- Reprocessing failed URLs complete ---")
        else:
            print("\nNo URLs in failed_urls.txt to reprocess.")
    else:
        print(f"\nNo {FAILED_URLS_LOG} file found or it is empty. No URLs to reprocess.")

    # 4. Merge fetched data back into the Media Cloud URLs DataFrame
    merged_df = merge_csv_with_articles(MC_URLS_CSV_PATH, RAW_ARTICLES_DIR, MERGED_OUTPUT_CSV)

    # 5. Analyze the final merged results
    analyze_merge_results(merged_df)

if __name__ == "__main__":
    main()
