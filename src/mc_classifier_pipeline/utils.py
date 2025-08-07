"""A module for important set-up and configuration functionality, but doesn't implement the library's key features."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def configure_logging():
    """A helper method that configures logging, usable by any script in this library."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s : %(asctime)s : %(name)s : %(message)s",
    )


def validate_environment_variables(required_vars: List[str]) -> Dict[str, str]:
    """
    Validate that required environment variables are set.

    Args:
        required_vars: List of environment variable names to check

    Returns:
        Dictionary mapping variable names to their values

    Raises:
        ValueError: If any required variables are missing
    """
    missing_vars = []
    values = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            values[var] = value

    if missing_vars:
        raise ValueError(f"Missing env variables: {', '.join(missing_vars)}. Set them in your .env file.")

    return values


def load_data_splits(
    project_folder: str, text_column: str = "text", label_column: str = "label"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test data from CSV files.

    Args:
        project_folder: Path to folder containing train.csv and test.csv
        text_column: Name of the text column
        label_column: Name of the label column

    Returns:
        Tuple of (train_df, test_df)

    Raises:
        FileNotFoundError: If train.csv or test.csv not found
        ValueError: If required columns not found
    """
    train_path = os.path.join(project_folder, "train.csv")
    test_path = os.path.join(project_folder, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Validate required columns
    for df_name, df in [("train", train_df), ("test", test_df)]:
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"Required columns '{text_column}' and '{label_column}' not found in {df_name} data")

    return train_df, test_df


def compute_classification_metrics(y_true: List, y_pred: List, average: str = "weighted") -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class metrics

    Returns:
        Dictionary containing accuracy, precision, recall, and f1 scores
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def detect_model_framework(model_dir: str) -> Optional[str]:
    """
    Detect ML framework (HuggingFace or sklearn) from model directory.

    Args:
        model_dir: Path to model directory

    Returns:
        'hf' for HuggingFace, 'sklearn' for scikit-learn, or None if detection fails
    """
    meta_path = os.path.join(model_dir, "metadata.json")

    # First try to detect framework from metadata.json
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            fw = str(meta.get("framework", "")).strip().lower()
            if fw in {"hf", "transformers"}:
                return "hf"
            elif fw in {"sk", "sklearn", "scikit-learn"}:
                return "sklearn"
        except Exception:
            pass

    # Fallback to file-based detection
    has_config = os.path.exists(os.path.join(model_dir, "config.json"))
    has_model_pkl = os.path.exists(os.path.join(model_dir, "model.pkl"))
    has_vectorizer = os.path.exists(os.path.join(model_dir, "vectorizer.pkl"))
    has_label_encoder = os.path.exists(os.path.join(model_dir, "label_encoder.pkl"))

    if has_config and has_label_encoder:
        return "hf"
    elif has_model_pkl and has_vectorizer and has_label_encoder:
        return "sklearn"

    return None


def create_timestamp_dir(parent: Path) -> Path:
    """
    Create a unique timestamped directory under `parent`, like 20250729_153015_001

    Args:
        parent: Parent directory to create timestamped subdirectory in

    Returns:
        Path to the created directory

    Raises:
        RuntimeError: If unable to create unique directory after 1000 attempts
    """
    from datetime import datetime

    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    MAX_DIRECTORY_ATTEMPTS = 1000

    for i in range(MAX_DIRECTORY_ATTEMPTS):
        suffix = f"_{i:03d}"
        p = parent / f"{base}{suffix}"
        if not p.exists():
            p.mkdir(parents=True, exist_ok=False)
            return p

    raise RuntimeError("Failed to create a unique timestamped directory")


def save_metadata(metadata: Dict[str, Any], output_path: Path) -> None:
    """
    Save metadata to JSON file, handling path serialization.

    Args:
        metadata: Dictionary to save
        output_path: Path to save the JSON file
    """

    def convert_paths(obj):
        """Convert Path objects to strings for JSON serialization"""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        return obj

    serializable_metadata = convert_paths(metadata)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metadata, f, indent=2, ensure_ascii=False, default=str)
