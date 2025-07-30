import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from mc_classifier_pipeline.bert_recipe import BERTTextClassifier
from mc_classifier_pipeline.sk_naive_bayes_recipe import SKNaiveBayesTextClassifier


def build_trainer_parser(add_help=True):
    """
    Build the argument parser for the trainer.
    """

    parser = argparse.ArgumentParser(
        description="Train multiple model recipes from the same train and test split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Path to a folder containing train.csv and test.csv (e.g., experiments/project_42/20250728_113000)",
    )
    parser.add_argument("--models-config", required=True, help="Path to a models config file with predefined model configurations")
    parser.add_argument("--model-names", nargs="+", help="Specific model names to train from the models config (if not specified, trains all)")
    parser.add_argument("--text-column", default="text", help="Text column name in train/test CSV (default: text)")
    parser.add_argument("--label-column", default="label", help="Label column name in train/test CSV (default: label)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for HF models during eval (default: 32)")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length for HF models during eval")

    return parser


def parse_args():
    return build_trainer_parser().parse_args()

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


def load_models_config(config_file: str, selected_models: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Load models from a configuration file.
    
    Args:
        config_file: Path to JSON config file with model definitions
        selected_models: Optional list of model names to train (if None, trains all)
    
    Returns:
        List of model configuration dictionaries
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    if "models" not in config:
        raise ValueError("Config file must have a 'models' key with list of model definitions")
    
    models = config["models"]
    
    # Filter by selected model names if provided
    if selected_models:
        models = [m for m in models if m.get("name") in selected_models]
        missing = set(selected_models) - {m.get("name") for m in models}
        if missing:
            raise ValueError(f"Models not found in config: {missing}")
    
    # Validate each model config
    for model in models:
        if "model_type" not in model:
            raise ValueError(f"Model {model.get('name', 'unnamed')} missing 'model_type'")
    
    return models


def train_model_from_config(
    model_config: Dict[str, Any],
    experiment_dir: Path,
    models_root: Path,
    text_column: str,
    label_column: str,
) -> Path:
    """
    Train a single model directly from its configuration.
    """
    model_type = model_config["model_type"]
    model_params = model_config.get("model_params", {})
    config_name = model_config.get("name", "unnamed_model")
    
    print(f"Training {config_name} ({model_type})...")
    
    # Create classifier directly based on type
    if model_type == "BertFineTune":
        model_name = model_params.pop("model_name", "bert-base-uncased")
        clf = BERTTextClassifier(model_name=model_name)
        framework = "hf"
    elif model_type == "SklearnMultinomialNaiveBayes":
        clf = SKNaiveBayesTextClassifier()
        framework = "sklearn"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    out_dir = make_timestamp_dir(models_root)

    # Train the model
    metadata = clf.train(
        project_folder=str(experiment_dir),
        save_path=str(out_dir),
        text_column=text_column,
        label_column=label_column,
        hyperparams=model_params if model_params else None,
    )

    # Save metadata with config name and framework
    if isinstance(metadata, dict):
        metadata.setdefault("framework", framework)
        metadata.setdefault("config_name", config_name)
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # Clean up
    del clf
    gc.collect()
    
    print(f"  → Saved to {out_dir}")
    return out_dir


def train_all_models(experiment_dir, model_configs, text_column, label_column):
    """Train all models from their configurations."""
    models_root = experiment_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    trained_dirs: List[Path] = []
    for config in model_configs:
        out_dir = train_model_from_config(
            model_config=config,
            experiment_dir=experiment_dir,
            models_root=models_root,
            text_column=text_column,
            label_column=label_column,
        )
        trained_dirs.append(out_dir)
    
    return trained_dirs


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()

    # Validate input files
    train_csv = experiment_dir / "train.csv"
    test_csv = experiment_dir / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("train.csv and/or test.csv not found in experiment-dir")
    
    # Load model configurations
    print(f"Loading models from config: {args.models_config}")
    if args.model_names:
        print(f"Training selected models: {args.model_names}")
    model_configs = load_models_config(args.models_config, args.model_names)

    print(f"Will train {len(model_configs)} models")
    for i, config in enumerate(model_configs):
        print(f"  {i+1}. {config.get('name', 'unnamed')} ({config['model_type']})")

    # Train all models
    train_all_models(experiment_dir, model_configs, args.text_column, args.label_column)


if __name__ == "__main__":
    main()