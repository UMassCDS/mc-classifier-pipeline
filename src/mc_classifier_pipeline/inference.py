from argparse import ArgumentParser
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
    parser = ArgumentParser(

    )

    parser.add_argument(
        "--url-file",
        type=Path,
        help = "Path to the file with list of urls"
    )

    parser.add_argument(
        "--model-dir",
        type = Path,
        help = "Path to the trained model"
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
            logger.info(f"framework is {fw}")
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


def generate_predictions(model_dir, df):
    framework = detect_framework(model_dir)
    texts = list(df["text"])
    y_pred = None

    if framework == "hf":
        y_pred = predict_labels_hf(
            model_dir,
            texts,
            max_length=None,  # used if provided; otherwise per-model metadata/default
            batch_size=32,
        )
    elif framework == "sklearn":
        y_pred = predict_labels_sklearn(model_dir, texts)
    else:
        raise RuntimeError(f"Unknown framework tag for {model_dir}: {framework}")
    
    if(y_pred is not None):
        df["prediction"] = y_pred
    _cleanup_memory()


def process_urls(urls):
    articles = []
    #urls = urls[:5]
    for url in urls:
        my_query = f'url:"{url}"'
        start_date = dt.date(2025, 1, 1)
        end_date = dt.date(2025, 6, 10)
        results = search_api.story_list(my_query, start_date, end_date)
        if(results and len(results[0]) > 0):
            story_id = results[0][0]['id']
            articles.append(search_api.story(story_id))

    df = pd.DataFrame(articles)[["id", "url", "text"]]
    return df
    



def main():
    args = parse_args()

    url_file = args.url_file
    urls = []
    with open(url_file, "r") as f:
        urls = [url.strip() for url in f]

    df = process_urls(urls)
    generate_predictions(args.model_dir, df)

    df.to_csv("predictions.csv", index=False)
    





if __name__ == "__main__":
    main()

#"experiments/project_1/20250806_103847/models/20250806_123513_000"