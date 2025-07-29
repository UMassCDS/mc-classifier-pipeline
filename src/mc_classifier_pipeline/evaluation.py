import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Configure logging
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


logger = logging.getLogger(__name__)


# IO helpers 
def load_test_data(
    experiment_dir: str,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    """
    Load test.csv and validate required columns.
    """
    test_path = os.path.join(experiment_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not find test.csv at: {test_path}")
    df = pd.read_csv(test_path)
    for col in (text_column, label_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in test.csv")
    return df[[text_column, label_column]].copy()


def discover_model_dirs(models_root: str) -> List[str]:
    """
    Return subdirectories under models_root that look like saved HF models.
    We require at least a label_encoder.pkl and a HF config.json.
    """
    if not os.path.isdir(models_root):
        raise FileNotFoundError(f"Models folder not found: {models_root}")

    model_dirs = []
    for name in sorted(os.listdir(models_root)):
        path = os.path.join(models_root, name)
        if not os.path.isdir(path):
            continue
        has_encoder = os.path.exists(os.path.join(path, "label_encoder.pkl"))
        has_config = os.path.exists(os.path.join(path, "config.json"))
        if has_encoder and has_config:
            model_dirs.append(path)
        else:
            logger.warning(f"Skipping {path} (missing label_encoder.pkl or config.json)")
    if not model_dirs:
        raise RuntimeError(f"No valid model folders found under: {models_root}")
    return model_dirs


#  Predictions 
@torch.no_grad()
def predict_labels_for_model(
    model_dir: str,
    texts: List[str],
    max_length: Optional[int] = None,
    batch_size: int = 32,
) -> List[str]:
    """
    Load tokenizer+model from `model_dir`, run batched inference on `texts`,
    then map predicted ids back to string labels using label_encoder.pkl.
    """
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load label encoder
    enc_path = os.path.join(model_dir, "label_encoder.pkl")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"label_encoder.pkl missing in {model_dir}")
    label_encoder = joblib.load(enc_path)

    # Try to get max_length from metadata if not provided
    if max_length is None:
        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            max_length = meta.get("hyperparameters", {}).get("max_length", 512)
        else:
            max_length = 512

    predictions: List[int] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,           # dynamic padding per batch
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        batch_pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        predictions.extend(batch_pred)

    # Map ids -> string labels
    pred_labels = label_encoder.inverse_transform(np.array(predictions)) if len(predictions) else np.array([])

    return pred_labels.tolist()


# Metrics
def compute_weighted_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, F1 with average='weighted',
    matching the training script's compute_metrics behavior.
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


#  Main eval 
def evaluate_models(
    experiment_dir: str,
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
    models_root = os.path.join(experiment_dir, "models")
    test_df = load_test_data(experiment_dir, text_column, label_column)
    model_dirs = discover_model_dirs(models_root)

    texts = test_df[text_column].astype(str).tolist()
    y_true = test_df[label_column].astype(str).tolist()

    rows = []
    per_model_metrics: Dict[str, Dict] = {}

    logger.info(f"Evaluating {len(model_dirs)} models with weighted metrics (no pos_label)")
    for mdir in tqdm(model_dirs, desc="Models"):
        name = os.path.basename(mdir.rstrip("/"))
        try:
            # Predictions
            y_pred = predict_labels_for_model(
                mdir,
                texts,
                max_length=max_length,
                batch_size=batch_size,
            )

            # Weighted metrics (decision quality, multi-class friendly)
            metrics = compute_weighted_metrics(y_true, y_pred)

            row = {
                "model_name": name,
                "model_path": mdir,
                **metrics,
            }
            rows.append(row)
            per_model_metrics[name] = row.copy()

        except Exception as e:
            logger.exception(f"Failed evaluating {mdir}: {e}")
            # Record a failed row with NaNs so you can see it in results.csv
            row = {
                "model_name": name,
                "model_path": mdir,
                "error": str(e),
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
            }
            rows.append(row)
            per_model_metrics[name] = {"error": str(e)}

    results = pd.DataFrame(rows)

    # Sort leaderboard by chosen metric (descending)
    valid_metrics = ["accuracy", "precision", "recall", "f1"]
    if best_metric not in valid_metrics:
        raise ValueError(f"--best-metric must be one of: {', '.join(valid_metrics)}")

    results = results.sort_values(by=best_metric, ascending=False, na_position="last").reset_index(drop=True)

    # Determine best model (first valid row)
    best_row = results.iloc[0].to_dict() if not results.empty else None
    summary = {
        "evaluated_at": datetime.now().isoformat(),
        "experiment_dir": experiment_dir,
        "models_root": models_root,
        "test_size": len(test_df),
        "text_column": text_column,
        "label_column": label_column,
        "best_metric": best_metric,
        "best_model": {
            "model_name": best_row.get("model_name") if best_row else None,
            "model_path": best_row.get("model_path") if best_row else None,
            "metrics": {
                k: best_row.get(k) for k in ("accuracy", "precision", "recall", "f1")
            } if best_row else None,
        },
        "metrics_per_model": per_model_metrics,
    }

    return results, summary


def write_outputs(models_root: str, results: pd.DataFrame, summary: Dict):
    """
    Write results.csv and evaluation_summary.json into the models/ folder.
    """
    os.makedirs(models_root, exist_ok=True)
    results_path = os.path.join(models_root, "results.csv")
    summary_path = os.path.join(models_root, "evaluation_summary.json")

    results.to_csv(results_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Leaderboard written: {results_path}")
    logger.info(f"Summary written: {summary_path}")


# ----------------------------- CLI -------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="Evaluate all trained models in an experiment folder (weighted metrics, like training script).",
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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=None, help="Override max sequence length for tokenization")
    return p


def main():
    configure_logging()
    args = build_argparser().parse_args()

    results, summary = evaluate_models(
        experiment_dir=args.experiment_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        best_metric=args.best_metric,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    models_root = os.path.join(args.experiment_dir, "models")
    write_outputs(models_root, results, summary)


if __name__ == "__main__":
    main()
