import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .bert_recipe import BERTTextClassifier                
from .sk_naive_bayes_recipe import SKNaiveBayesTextClassifier   


from .evaluation import evaluate_models, write_outputs


# Logging 
from .utils import configure_logging  




# Recipes registry 
"""
We map a user-provided `model_type` string to:
- a constructor callable
- which keys in `model_params` are constructor args vs. training hyperparams
- a human-readable slug (used only in logs; folder names are timestamps)
"""

MODEL_REGISTRY = {
    "BertFineTune": {
        "constructor": BERTTextClassifier,
        "constructor_keys": {"model_name"},           # pass these to __init__
        "slug": "bert",
        "framework": "hf",
    },
    "SklearnMultinomialNaiveBayes": {
        "constructor": SKNaiveBayesTextClassifier,
        "constructor_keys": set(),                    # sklearn constructor takes no args
        "slug": "sk-nb",
        "framework": "sklearn",
    },
}


# Helper: parse recipes 
def load_recipes(recipes_json: Optional[str], recipes_file: Optional[str]) -> List[Dict[str, Any]]:
    """
    Load the recipes list from either a JSON string or a JSON file.
    Each item should look like:
      {"model_type": "BertFineTune", "model_params": {"model_name": "bert-base-uncased", "learning_rate": 3e-5}}
    """
    if recipes_file:
        with open(recipes_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif recipes_json:
        data = json.loads(recipes_json)
    else:
        raise ValueError("You must provide --recipes or --recipes-file")

    if not isinstance(data, list):
        raise ValueError("Recipes must be a JSON list of recipe objects")

    for i, item in enumerate(data):
        if not isinstance(item, dict) or "model_type" not in item:
            raise ValueError(f"Recipe #{i} must be an object with a 'model_type' key")
        item.setdefault("model_params", {})
        if not isinstance(item["model_params"], dict):
            raise ValueError(f"Recipe #{i} 'model_params' must be an object")
    return data


# Helper: timestamp dir 
def make_timestamp_dir(parent: Path) -> Path:
    """
    Create a unique timestamped directory under `parent`, like 20250729_153015_001
    """
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Avoid collisions if called multiple times in the same second.
    for i in range(1000):
        suffix = f"_{i:03d}"
        p = parent / f"{base}{suffix}"
        if not p.exists():
            p.mkdir(parents=True, exist_ok=False)
            return p
    raise RuntimeError("Failed to create a unique timestamped directory")


# Training logic 
def split_params(model_type: str, model_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split model_params into constructor kwargs and training hyperparams,
    based on the registry for the given model_type.
    """
    spec = MODEL_REGISTRY[model_type]
    constructor_keys = spec["constructor_keys"]
    constructor_kwargs = {k: v for k, v in model_params.items() if k in constructor_keys}
    hyperparams = {k: v for k, v in model_params.items() if k not in constructor_keys}
    return constructor_kwargs, hyperparams


def train_single_model(
    model_type: str,
    project_dir: Path,
    models_root: Path,
    text_column: str,
    label_column: str,
    model_params: Dict[str, Any],
) -> Path:
    """
    Train one model given a recipe, write it into its own timestamped folder, and return that folder path.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: {list(MODEL_REGISTRY)}")

    spec = MODEL_REGISTRY[model_type]
    constructor_kwargs, hyperparams = split_params(model_type, model_params)
    constructor = spec["constructor"]
    slug = spec["slug"]
    framework = spec["framework"]

    out_dir = make_timestamp_dir(models_root)  # timestamp-only naming

    # Build the classifier instance
    clf = constructor(**constructor_kwargs)

    # Call its train method
    metadata = clf.train(
        project_folder=str(project_dir),
        save_path=str(out_dir),
        text_column=text_column,
        label_column=label_column,
        hyperparams=hyperparams if hyperparams else None,
    )

    # Ensure metadata carries framework tag
    if isinstance(metadata, dict):
        metadata.setdefault("framework", framework)
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    return out_dir


#  Evaluation integration 
def run_evaluation(
    project_dir: Path,
    best_metric: str,
    text_column: str,
    label_column: str,
    batch_size: int = 32,
    max_length: Optional[int] = None,
) -> None:
    """
    Run evaluation (weighted metrics) across all model folders under <project_dir>/models
    and write results.csv + evaluation_summary.json into that models/ root.
    """
    
    
    results, summary = evaluate_models(
        experiment_dir=str(project_dir),
        text_column=text_column,
        label_column=label_column,
        best_metric=best_metric,
        batch_size=batch_size,
        max_length=max_length,
    )

    models_root = project_dir / "models"
    write_outputs(str(models_root), results, summary)

    best = summary.get("best_model", {})


# CLI
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train multiple model recipes on the same dataset split, then evaluate them."
    )
    p.add_argument(
        "--project-dir",
        required=True,
        help="Path to a folder containing train.csv and test.csv (e.g., experiments/project_42/20250728_113000)",
    )
    p.add_argument(
        "--recipes",
        help="JSON string for a list of recipes. Example: "
             "'[{\"model_type\":\"BertFineTune\",\"model_params\":{\"model_name\":\"bert-base-uncased\",\"learning_rate\":3e-5}}, "
             "{\"model_type\":\"SklearnMultinomialNaiveBayes\",\"model_params\":{\"alpha\":0.2}}]'",
    )
    p.add_argument("--recipes-file", help="Path to a JSON file that contains the list of recipes")
    p.add_argument(
        "--best-metric",
        default="f1",
        choices=["accuracy", "precision", "recall", "f1"],
        help="Metric used to select the best model in evaluation (default: f1)",
    )
    p.add_argument("--text-column", default="text", help="Text column name in train/test CSV (default: text)")
    p.add_argument("--label-column", default="label", help="Label column name in train/test CSV (default: label)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for HF models during eval (default: 32)")
    p.add_argument("--max-length", type=int, default=None, help="Max sequence length for HF models during eval")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    project_dir = Path(args.project_dir).resolve()
    models_root = project_dir / "models"
    log_file = models_root / "training.log"
    #configure_logging(log_file)

    # Validate input files
    train_csv = project_dir / "train.csv"
    test_csv = project_dir / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("train.csv and/or test.csv not found in project-dir")

    # Load recipes
    recipes = load_recipes(args.recipes, args.recipes_file)

    # Ensure models root exists
    models_root.mkdir(parents=True, exist_ok=True)

    # Train each recipe
    trained_dirs: List[Path] = []
    for idx, rec in enumerate(recipes, 1):
        model_type = rec["model_type"]
        model_params = rec.get("model_params", {}) or {}

        out_dir = train_single_model(
            model_type=model_type,
            project_dir=project_dir,
            models_root=models_root,
            text_column=args.text_column,
            label_column=args.label_column,
            model_params=model_params,
        )
        trained_dirs.append(out_dir)

    run_evaluation(
        project_dir=project_dir,
        best_metric=args.best_metric,
        text_column=args.text_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
