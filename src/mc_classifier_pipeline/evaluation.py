import json
import argparse
import gc
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


# Configure Logging
from mc_classifier_pipeline.utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


# IO helpers
def load_test_data(
    experiment_dir: Path,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    """
    Load test.csv and validate required columns.
    """
    test_path = experiment_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Could not find test.csv at: {test_path}")
    df = pd.read_csv(test_path)
    for col in (text_column, label_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in test.csv")
    return df[[text_column, label_column]].copy()


def discover_model_dirs(models_root: Path) -> List[Dict[str, str]]:
    """
    Scan <models_root> for valid model directories.

    A directory qualifies if it has:
      - Hugging Face:  config.json  AND  label_encoder.pkl
      - scikit-learn:  model.pkl AND vectorizer.pkl AND label_encoder.pkl

    Returns a list of dicts: [{"path": "<abs_path>", "framework": "hf"|"sklearn", "name": "<dir_name>"}]
    """
    logger.info(f"Scanning for models in: {models_root}")
    if not models_root.is_dir():
        raise FileNotFoundError(f"Models folder not found: {models_root}")

    found: List[Dict[str, str]] = []
    for model_path in sorted(models_root.iterdir()):
        if not model_path.is_dir():
            continue

        framework: Optional[str] = None

        # First try to detect framework from metadata.json
        meta_path = model_path / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                fw = str(meta.get("framework", "")).strip().lower()
                if fw in {"hf", "transformers"}:
                    framework = "hf"
                    logger.debug(f"Detected HuggingFace model from metadata: {model_path}")
                elif fw in {"sk", "sklearn", "scikit-learn"}:
                    framework = "sklearn"
                    logger.debug(f"Detected sklearn model from metadata: {model_path}")
            except Exception as e:
                logger.warning(f"Could not read metadata.json in {model_path}: {e}")

        # If framework not detected from metadata, try file-based detection
        if not framework:
            has_config = (model_path / "config.json").exists()
            has_model_pkl = (model_path / "model.pkl").exists()
            has_vectorizer = (model_path / "vectorizer.pkl").exists()

            if has_config:
                framework = "hf"
                logger.debug(f"Detected HuggingFace model from config.json: {model_path}")
            elif has_model_pkl and has_vectorizer:
                framework = "sklearn"
                logger.debug(f"Detected sklearn model from model.pkl + vectorizer.pkl: {model_path}")
            else:
                logger.warning(f"Could not determine framework for {model_path}")
                continue

        found.append({"path": str(model_path), "framework": framework, "name": model_path.name})

    if not found:
        logger.error(f"No valid model folders found under: {models_root}")
        raise RuntimeError(f"No valid model folders found under: {models_root}")

    logger.info(f"Found {len(found)} valid models: {[m['name'] for m in found]}")
    return found


# Predictions
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


# Metrics
def compute_weighted_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, F1 with average='weighted',
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


# Main eval
def evaluate_models(
    experiment_dir: Path,
    text_column: str,
    label_column: str,
    best_metric: str,
    batch_size: int,
    max_length: Optional[int],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate all models under <experiment_dir>/models against test.csv using weighted metrics.
    Produces a leaderboard DataFrame and a summary dict.
    """
    logger.info(f"Starting model evaluation for experiment: {experiment_dir}")
    models_root = experiment_dir / "models"
    test_df = load_test_data(experiment_dir, text_column, label_column)
    logger.info(f"Loaded test data with {len(test_df)} samples")
    model_dirs = discover_model_dirs(models_root)

    texts = test_df[text_column].astype(str).tolist()
    y_true = test_df[label_column].astype(str).tolist()

    rows = []
    per_model_metrics: Dict[str, Dict] = {}
    success_count = 0
    failure_count = 0

    logger.info(f"Evaluating {len(model_dirs)} models with weighted metrics (framework-aware)")
    for item in tqdm(model_dirs):
        mdir, framework, name = item["path"], item["framework"], item["name"]
        logger.info(f"Evaluating model: {name} (framework: {framework})")
        try:
            if framework == "hf":
                y_pred = predict_labels_hf(
                    mdir,
                    texts,
                    max_length=max_length,  # used if provided; otherwise per-model metadata/default
                    batch_size=batch_size,
                )
            elif framework == "sklearn":
                y_pred = predict_labels_sklearn(mdir, texts)
            else:
                raise RuntimeError(f"Unknown framework tag for {mdir}: {framework}")

            metrics = compute_weighted_metrics(y_true, y_pred)
            logger.info(
                f"Model {name} evaluation completed - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
            )

            row = {
                "model_name": name,
                "model_path": mdir,
                "framework": framework,
                **metrics,
            }
            rows.append(row)
            per_model_metrics[name] = row.copy()
            success_count += 1

            # Clean up memory after successful evaluation
            _cleanup_memory()

        except Exception as e:
            logger.exception(f"Failed evaluating {mdir}: {e}")
            logger.error(f"Model {name} evaluation failed with error: {str(e)}")
            row = {
                "model_name": name,
                "model_path": mdir,
                "framework": framework,
                "error": str(e),
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
            }
            rows.append(row)
            per_model_metrics[name] = {"error": str(e), "framework": framework}
            failure_count += 1

            # Clean up memory even after failures
            _cleanup_memory()

    logger.info("All model evaluations completed")
    logger.info(
        f"Evaluation summary: {success_count} models succeeded, {failure_count} models failed out of {len(model_dirs)} total"
    )

    if failure_count > 0:
        failed_models = [row["model_name"] for row in rows if "error" in row]
        logger.warning(f"Failed models: {', '.join(failed_models)}")

    if success_count > 0:
        successful_models = [row["model_name"] for row in rows if "error" not in row]
        logger.info(f"Successful models: {', '.join(successful_models)}")

    results = pd.DataFrame(rows)

    # Sort leaderboard by chosen metric (descending)
    valid_metrics = ["accuracy", "precision", "recall", "f1"]
    if best_metric not in valid_metrics:
        raise ValueError(f"--best-metric must be one of: {', '.join(valid_metrics)}")

    logger.info(f"Sorting results by {best_metric} metric")
    results = results.sort_values(by=best_metric, ascending=False, na_position="last").reset_index(drop=True)

    # Determine best model (first valid row)
    best_row = results.iloc[0].to_dict() if not results.empty else None
    if best_row:
        logger.info(
            f"Best model: {best_row['model_name']} ({best_row['framework']}) - {best_metric}: {best_row[best_metric]:.4f}"
        )
    else:
        logger.warning("No valid models found in evaluation results")
    summary = {
        "evaluated_at": datetime.now().isoformat(),
        "experiment_dir": str(experiment_dir),
        "models_root": str(models_root),
        "test_size": len(test_df),
        "text_column": text_column,
        "label_column": label_column,
        "best_metric": best_metric,
        "total_models": len(model_dirs),
        "successful_models": success_count,
        "failed_models": failure_count,
        "best_model": {
            "model_name": best_row.get("model_name") if best_row else None,
            "model_path": best_row.get("model_path") if best_row else None,
            "framework": best_row.get("framework") if best_row else None,
            "metrics": {k: best_row.get(k) for k in ("accuracy", "precision", "recall", "f1")} if best_row else None,
        },
        "metrics_per_model": per_model_metrics,
    }

    logger.info(
        f"Evaluation summary created: {success_count} successful, {failure_count} failed, {len(results)} total models"
    )
    return results, summary


def write_outputs(models_root: Path, results: pd.DataFrame, summary: Dict):
    """
    Write results.csv and evaluation_summary.json into the models/ folder.
    """
    models_root.mkdir(parents=True, exist_ok=True)
    results_path = models_root / "results.csv"
    summary_path = models_root / "evaluation_summary.json"

    results.to_csv(results_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Leaderboard written: {results_path}")
    logger.info(f"Summary written: {summary_path}")


# CLI
def build_argparser(add_help: bool = True):
    p = argparse.ArgumentParser(
        description="Evaluate all trained models in an experiment folder (weighted metrics; supports HF and sklearn).",
        add_help=add_help,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--experiment-dir", required=True, help="Path to experiment folder containing test.csv and models/")
    p.add_argument("--text-column", default="text", help="Text column name in test.csv")
    p.add_argument("--label-column", default="label", help="Label column name in test.csv")
    p.add_argument(
        "--best-metric",
        default="f1",
        choices=["accuracy", "precision", "recall", "f1"],
        help="Metric used to select the best model (default: f1)",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for HF models (ignored for sklearn)")
    p.add_argument("--max-length", type=int, default=None, help="Max sequence length for HF models (optional)")
    return p


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    return build_argparser().parse_args()


def main(args: Optional[argparse.Namespace] = None):
    logger.info("Starting evaluation script")
    if args is None:
        args = parse_args()

    # Convert experiment_dir to Path object
    experiment_dir = Path(args.experiment_dir)

    logger.info(
        f"Evaluation parameters - experiment_dir: {experiment_dir}, "
        f"text_column: {args.text_column}, label_column: {args.label_column}, "
        f"best_metric: {args.best_metric}, batch_size: {args.batch_size}"
    )

    results, summary = evaluate_models(
        experiment_dir=experiment_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        best_metric=args.best_metric,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    models_root = experiment_dir / "models"
    write_outputs(models_root, results, summary)
    logger.info("Evaluation script completed successfully")


if __name__ == "__main__":
    main()
