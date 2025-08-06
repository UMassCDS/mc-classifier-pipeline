from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
import gc
import logging
from pathlib import Path
import mediacloud.api
from dotenv import load_dotenv
import os
from importlib.metadata import version
import datetime as dt
import json
import pandas as pd
from typing import List, Optional
from tqdm import tqdm

from mc_classifier_pipeline import utils

# Configure logging
utils.configure_logging()
logger = logging.getLogger(__name__)


load_dotenv()
api_key = os.getenv("MC_API_KEY")
search_api = mediacloud.api.SearchApi(api_key)
directory_api = mediacloud.api.DirectoryApi(api_key)
f"Using Media Cloud python client v{version('mediacloud')}"



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
    
    parser = ArgumentParser(
        description=description,
        formatter_class=RawDescriptionHelpFormatter  
    )

    parser.add_argument(
        "--url-file",
        type=Path,
        required=True,
        help = "Path to the file with list of urls"
    )

    parser.add_argument(
        "--model-dir",
        type = Path,
        required=True,
        help = "Path to the trained model"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default="predictions.csv",
        help="Output CSV file for predictions (default: predictions.csv)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model inference (default: 32)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-01-01",
        help="Start date for Media Cloud search (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", 
        type=str,
        default="2025-06-10",
        help="End date for Media Cloud search (YYYY-MM-DD)"
    )

    return parser

def parse_args():
    parser = build_inference_parser()
    return parser.parse_args()

def detect_framework(path):
    meta_path = os.path.join(path, "metadata.json")
    framework = None
    if os.path.exists(meta_path):
        logger.info("Opening metadata file")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            fw = str(meta.get("framework", "")).strip().lower()
            if fw in {"hf", "transformers"}:
                framework = "hf"
                logger.debug(f"Detected HuggingFace model from metadata: {path}")
            elif fw in {"sk", "sklearn", "scikit-learn"}:
                framework = "sklearn"
                logger.debug(f"Detected sklearn model from metadata: {path}")
        except Exception as e:
            logger.warning(f"Could not read metadata.json in {path}: {e}")

    # If framework not detected from metadata, try file-based detection
    if not framework:
        has_config = os.path.exists(os.path.join(path, "config.json"))
        has_model_pkl = os.path.exists(os.path.join(path, "model.pkl"))
        has_vectorizer = os.path.exists(os.path.join(path, "vectorizer.pkl"))

        if has_config:
            framework = "hf"
            logger.debug(f"Detected HuggingFace model from config.json: {path}")
        elif has_model_pkl and has_vectorizer:
            framework = "sklearn"
            logger.debug(f"Detected sklearn model from model.pkl + vectorizer.pkl: {path}")
        else:
            logger.warning(f"Could not determine framework for {path}")

    return framework

def predict_labels_hf(
    model_dir: str,
    texts: List[str],
    max_length: Optional[int] = None,
    batch_size: int = 32,
) -> List[str]:
    """
    Hugging Face inference: load tokenizer+model from `model_dir`,
    run batched inference on `texts`, and inverse-transform to string labels.
    """
    from mc_classifier_pipeline.bert_recipe import BERTTextClassifier

    logger.debug(f"Loading HuggingFace model from: {model_dir}")
    # Use BERTTextClassifier from bert_recipe for HF model predictions
    classifier = BERTTextClassifier.load_for_inference(model_path=model_dir)
    logger.debug(f"Running predictions on {len(texts)} texts with batch_size={batch_size}")
    predictions = classifier.predict(texts=texts, return_probabilities=False)

    # Explicitly delete the classifier to free memory
    del classifier
    logger.debug("HuggingFace model cleaned up from memory")

    return predictions

def predict_labels_sklearn(
    model_dir: str,
    texts: List[str],
) -> List[str]:
    """Use SKNaiveBayesTextClassifier for sklearn predictions."""
    from mc_classifier_pipeline.sk_naive_bayes_recipe import SKNaiveBayesTextClassifier  # Import from correct module

    logger.debug(f"Loading sklearn model from: {model_dir}")
    classifier = SKNaiveBayesTextClassifier.load_for_inference(model_path=model_dir)
    logger.debug(f"Running sklearn predictions on {len(texts)} texts")
    predictions = classifier.predict(texts=texts, return_probabilities=False)

    # Explicitly delete the classifier to free memory
    del classifier
    logger.debug("Sklearn model cleaned up from memory")

    return predictions


def _cleanup_memory():
    """Clean up memory after model evaluation"""

    logger.debug("Starting memory cleanup...")
    # Force garbage collection
    gc.collect()

    # Clear GPU memory if available
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory cache cleared")
    except ImportError:
        # torch not available, skip GPU cleanup
        logger.debug("PyTorch not available, skipping GPU cleanup")

    logger.debug("Memory cleanup completed")


def generate_predictions(model_dir, df, batch_size):
    framework = detect_framework(model_dir)
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
    
    if(y_pred is not None):
        df["prediction"] = y_pred
    _cleanup_memory()


def process_urls(urls, start_date_str="2025-01-01", end_date_str="2025-06-10"):
    # Parse date strings
    try:
        start_date = dt.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = dt.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
    

    articles = []
    failed_count = 0

    for url in tqdm(urls, desc="Fetching articles"):
        try:
            my_query = f'url:"{url}"'
            results = search_api.story_list(my_query, start_date, end_date)
            if results and len(results[0]) > 0:
                story_id = results[0][0]['id']
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
    df.to_csv(args.output_file, index=False)
    logger.info(f"Predictions saved to: {args.output_file}")
    logger.info(f"Processed {len(df)} articles successfully")

    # Print summary
    if "prediction" in df.columns:
        prediction_counts = df["prediction"].value_counts()
        logger.info(f"Prediction distribution:\n{prediction_counts}")
    

if __name__ == "__main__":
    main()

#"experiments/project_1/20250806_103847/models/20250806_123513_000"