import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Configure Logging 
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


def discover_model_dirs(models_root: str) -> List[Dict[str, str]]:
    """
    Scan <models_root> for valid model directories.

    A directory qualifies if it has:
      - Hugging Face:  config.json  AND  label_encoder.pkl
      - scikit-learn:  model.pkl AND vectorizer.pkl AND label_encoder.pkl

    Returns a list of dicts: [{"path": "<abs_path>", "framework": "hf"|"sklearn", "name": "<dir_name>"}]
    """
    if not os.path.isdir(models_root):
        raise FileNotFoundError(f"Models folder not found: {models_root}")

    found: List[Dict[str, str]] = []
    for name in sorted(os.listdir(models_root)):
        path = os.path.join(models_root, name)
        if not os.path.isdir(path):
            continue

        has_label_encoder = os.path.exists(os.path.join(path, "label_encoder.pkl"))
        is_hf = os.path.exists(os.path.join(path, "config.json"))
        is_sk = (
            os.path.exists(os.path.join(path, "model.pkl"))
            and os.path.exists(os.path.join(path, "vectorizer.pkl"))
        )

        if has_label_encoder and is_hf:
            found.append({"path": path, "framework": "hf", "name": name})
        elif has_label_encoder and is_sk:
            found.append({"path": path, "framework": "sklearn", "name": name})
        else:
            logger.warning(
                f"Skipping {path} (missing required files: "
                f"{'label_encoder.pkl ' if not has_label_encoder else ''}"
                f"{'' if is_hf or is_sk else 'and neither HF nor sklearn artifacts detected'})"
            )

    if not found:
        raise RuntimeError(f"No valid model folders found under: {models_root}")

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
    # Lazy imports so sklearn-only runs don't require transformers/torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Label encoder
    enc_path = os.path.join(model_dir, "label_encoder.pkl")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"label_encoder.pkl missing in {model_dir}")
    label_encoder = joblib.load(enc_path)

    # Determine max_length
    if max_length is None:
        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            max_length = meta.get("hyperparameters", {}).get("max_length", 512)
        else:
            max_length = 512

    preds_int: List[int] = []

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
        with torch.no_grad():
            logits = model(**enc).logits
            batch_pred = logits.argmax(dim=-1).cpu().numpy().tolist()
            preds_int.extend(batch_pred)

    # ids -> string labels
    pred_labels = label_encoder.inverse_transform(np.array(preds_int)) if len(preds_int) else np.array([])
    return pred_labels.tolist()


def predict_labels_sklearn(
    model_dir: str,
    texts: List[str],
) -> List[str]:
    """
    scikit-learn inference: load vectorizer+model+label_encoder from `model_dir`,
    predict class ids, and inverse-transform to string labels.
    """
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    X = vectorizer.transform(texts)
    pred_ids = model.predict(X)
    pred_labels = label_encoder.inverse_transform(pred_ids)
    return pred_labels.tolist()


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

    logger.info(f"Evaluating {len(model_dirs)} models with weighted metrics (framework-aware)")
    for item in model_dirs:
        mdir, framework, name = item["path"], item["framework"], item["name"]
        try:
            if framework == "hf":
                y_pred = predict_labels_hf(
                    mdir,
                    texts,
                    max_length=max_length,    # used if provided; otherwise per-model metadata/default
                    batch_size=batch_size,
                )
            elif framework == "sklearn":
                y_pred = predict_labels_sklearn(mdir, texts)
            else:
                raise RuntimeError(f"Unknown framework tag for {mdir}: {framework}")

            metrics = compute_weighted_metrics(y_true, y_pred)

            row = {
                "model_name": name,
                "model_path": mdir,
                "framework": framework,
                **metrics,
            }
            rows.append(row)
            per_model_metrics[name] = row.copy()

        except Exception as e:
            logger.exception(f"Failed evaluating {mdir}: {e}")
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
            "framework": best_row.get("framework") if best_row else None,
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


# CLI 
def build_argparser():
    p = argparse.ArgumentParser(
        description="Evaluate all trained models in an experiment folder (weighted metrics; supports HF and sklearn).",
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
