import datetime as dt
import logging
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from importlib.metadata import version
from pathlib import Path
from typing import List

import mediacloud.api
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from mc_classifier_pipeline import utils
from mc_classifier_pipeline.prediction import ModelPredictor, cleanup_memory
from mc_classifier_pipeline.utils import detect_model_framework


utils.configure_logging()
logger = logging.getLogger(__name__)


load_dotenv()
api_key = os.getenv("MC_API_KEY")
try:
    search_api = mediacloud.api.SearchApi(api_key)
    directory_api = mediacloud.api.DirectoryApi(api_key)
    logger.info(f"Using Media Cloud python client v{version('mediacloud')}")
except Exception as e:
    raise RuntimeError(f"Failed to initialize Media Cloud APIs: {e}")


def build_inference_parser():
    """Build argument parser for inference script."""

    description = """
Run inference on articles from URLs using trained models.

Examples:
    # Basic usage
    python -m src.mc_classifier_pipeline.inference \\
        --url-file url_list.txt \\
        --model-dir experiments/project_1/20250806_103847/models/20250806_123513_000

    # With custom parameters
    python -m src.mc_classifier_pipeline.inference \\
        --url-file my_urls.txt \\
        --model-dir models/bert_model \\
        --output-file my_predictions.csv \\
        --batch-size 16 \\
        --start-date 2025-01-01 \\
        --end-date 2025-06-01
    """

    parser = ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("--url-file", type=Path, required=True, help="Path to the file with list of urls")

    parser.add_argument("--model-dir", type=Path, required=True, help="Path to the trained model")
    parser.add_argument(
        "--output-file",
        type=Path,
        default="predictions.csv",
        help="Output CSV file for predictions (default: predictions.csv)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model inference (default: 32)")
    parser.add_argument(
        "--start-date", type=str, default="2025-01-01", help="Start date for Media Cloud search (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, default="2025-06-10", help="End date for Media Cloud search (YYYY-MM-DD)"
    )

    return parser


def parse_args():
    """Parse command-line arguments for inference."""
    parser = build_inference_parser()
    return parser.parse_args()


def predict_labels_hf(
    model_dir: str,
    texts: List[str],
    batch_size: int = 32,
) -> List[str]:
    """
    Predict labels using HuggingFace model (legacy function for backward compatibility).
    """
    predictor = ModelPredictor(model_dir)
    try:
        return predictor.predict(texts, batch_size)
    finally:
        predictor.cleanup()


def predict_labels_sklearn(
    model_dir: str,
    texts: List[str],
) -> List[str]:
    """
    Predict labels using sklearn model (legacy function for backward compatibility).
    """
    predictor = ModelPredictor(model_dir)
    try:
        return predictor.predict(texts)
    finally:
        predictor.cleanup()


def generate_predictions(model_dir, df, batch_size):
    """
    Generate predictions using appropriate framework and add to DataFrame.

    Args:
        model_dir: Path to trained model
        df: DataFrame with 'text' column
        batch_size: Batch size for inference
    """
    framework = detect_model_framework(model_dir)
    texts = list(df["text"])
    y_pred = None

    if framework == "hf":
        y_pred = predict_labels_hf(
            model_dir,
            texts,
            batch_size=batch_size,
        )
    elif framework == "sklearn":
        y_pred = predict_labels_sklearn(model_dir, texts)
    else:
        raise RuntimeError(f"Unknown framework tag for {model_dir}: {framework}")

    if y_pred is not None:
        df["prediction"] = y_pred
    cleanup_memory()


def validate_dates(start_date, end_date):
    """Validate that date range is logical and not in future."""

    today = dt.date.today()
    if end_date < start_date:
        raise ValueError("End date must be after start date.")
    if start_date > today or end_date > today:
        raise ValueError("Dates cannot be in the future.")


def process_urls(urls, start_date_str="2025-01-01", end_date_str="2025-06-10"):
    """
    Fetch article content from Media Cloud using URLs.

    Args:
        urls: List of article URLs
        start_date_str: Start date (YYYY-MM-DD)
        end_date_str: End date (YYYY-MM-DD)

    Returns:
        DataFrame with columns ['id', 'url', 'text']
    """
    # Parse date strings
    try:
        start_date = dt.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = dt.datetime.strptime(end_date_str, "%Y-%m-%d").date()
        validate_dates(start_date, end_date)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

    articles = []
    failed_count = 0

    for url in tqdm(urls, desc="Fetching articles"):
        try:
            my_query = f'url:"{url}"'
            results = search_api.story_list(my_query, start_date, end_date)
            if results and len(results[0]) > 0:
                story_id = results[0][0]["id"]
                articles.append(search_api.story(story_id))
            else:
                logger.warning(f"No story found for URL: {url}")
                failed_count += 1
        except Exception as e:
            logger.error(f"Failed to fetch story for URL {url}: {e}")
            failed_count += 1
            continue

    logger.info(f"Successfully fetched {len(articles)} articles, {failed_count} failed")

    if not articles:
        raise ValueError("No articles were successfully fetched")

    df = pd.DataFrame(articles)[["id", "url", "text"]]
    return df


def main():
    args = parse_args()

    # Validate inputs
    if not args.url_file.exists():
        raise FileNotFoundError(f"URL file not found: {args.url_file}")

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    # Read and validate URLs
    with open(args.url_file, "r") as f:
        urls = [url.strip() for url in f if url.strip()]

    if not urls:
        raise ValueError("No valid URLs found in the file")

    logger.info(f"Processing {len(urls)} URLs with model: {args.model_dir}")

    df = process_urls(urls, args.start_date, args.end_date)
    # Add model info to output
    df["model_used"] = str(args.model_dir)
    df["prediction_timestamp"] = dt.datetime.now().isoformat()

    generate_predictions(args.model_dir, df, args.batch_size)

    # Save with logging
    logger.info(f"Number of failed articles: {len(urls) - len(df)}")
    df.to_csv(args.output_file, index=False)
    logger.info(f"Predictions saved to: {args.output_file}")
    logger.info(f"Processed {len(df)} articles successfully")

    # Print summary
    if "prediction" in df.columns:
        prediction_counts = df["prediction"].value_counts()
        logger.info(f"Prediction distribution:\n{prediction_counts}")


if __name__ == "__main__":
    main()
