"""
This orchestrator supports 6 classification scenarios based on Label Studio configuration
and target label specification. The pipeline automatically determines the appropriate
training strategy and model types based on the annotation config structure.

SUPPORTED SCENARIOS:

1. SINGLE-LABEL CATEGORY (No Target Specified)
    Label Studio Config:
        <Choices name="sentiment" choice="single">
        <Choice value="Positive"/><Choice value="Negative"/><Choice value="Neutral"/>
        </Choices>

    Preprocessing: Extracts mixed single labels ["Positive", "Negative", "Neutral"]
    Training: 3 binary classifiers (Positive vs not-Positive, Negative vs not-Negative, Neutral vs not-Neutral)
    Use Case: Topic classification, sentiment analysis where only one label applies

2. MULTIPLE SINGLE-LABEL CATEGORIES (No Target Specified)
    Label Studio Config:
        <Choices name="sentiment" choice="single">...</Choices>
        <Choices name="topic" choice="single">...</Choices>

    Preprocessing: Extracts all labels from both categories in single-label format
    Training: N binary classifiers (one for each unique label across all categories)
    Use Case: Multiple classification tasks with mutually exclusive choices

3. MULTI-LABEL CATEGORY (No Target Specified)
    Label Studio Config:
        <Choices name="tags" choice="multiple">
        <Choice value="Opinion"/><Choice value="Analysis"/><Choice value="Breaking News"/>
        </Choices>

    Preprocessing: Extracts in multi-label format [["Opinion", "Analysis"], ["Breaking News"]]
    Training: 1 multi-label classifier handling all tag labels simultaneously
    Use Case: Content tagging where multiple labels can apply to one article

4. MULTIPLE MULTI-LABEL CATEGORIES (No Target Specified)
    Label Studio Config:
        <Choices name="tags" choice="multiple">...</Choices>
        <Choices name="emotions" choice="multiple">...</Choices>

    Preprocessing: Extracts labels by category, preserving structure
    Training: 2 separate multi-label classifiers (one for tags, one for emotions)
    Use Case: Multiple independent multi-label tasks (e.g., content tags + emotional tone)

5. MIXED CATEGORIES (No Target Specified)
    Label Studio Config:
        <Choices name="sentiment" choice="single">...</Choices>
        <Choices name="tags" choice="multiple">...</Choices>

    Preprocessing: Extracts everything in categorized format
    Training: Binary classifiers for single-label categories + multi-label classifiers for multi-label categories
    Use Case: Complex annotation projects with both exclusive and non-exclusive labeling

6. TARGETED CLASSIFICATION (Target Specified)
    Command: --target-label sentiment (category name)
            --target-label tags (category name)
            --target-label Positive (specific label)

    Preprocessing: Extracts only specified target labels/category
    Training: Depends on target type:
        - Single-label category → Binary classifiers for each label in category
        - Multi-label category → One multi-label classifier for category
        - Specific label → One binary classifier for that label
    Use Case: Focused training on specific classification task

TECHNICAL DETAILS:

Model Types Generated:
- Binary Classifier: Uses BCEWithLogitsLoss, outputs 0/1 for single label presence
- Multi-label Classifier: Uses BCEWithLogitsLoss, outputs probability vector for multiple labels
- Multi-class Classifier: Uses CrossEntropyLoss, outputs one choice from multiple options

Data Formats:
- Single-label: "Positive", "Sports" (string labels)
- Multi-label: ["Opinion", "Analysis"] (list of labels)
- Categorized: {"sentiment": ["Positive"], "tags": ["Opinion", "Analysis"]} (structured labels)

Output Structure:
- Each training job produces a separate model in timestamped directories
- Models are named: {model_type}_{job_name} (e.g., bert_binary_Positive, bert_multilabel_tags)
- Training summary includes job types, target labels, and performance metrics
"""

import argparse
import logging
import os
import json
from pathlib import Path
import pandas as pd
from mc_classifier_pipeline.utils import configure_logging
import mc_classifier_pipeline.preprocessing as preproc
import mc_classifier_pipeline.trainer as trainer
import mc_classifier_pipeline.evaluation as evaluator

configure_logging()
logger = logging.getLogger(__name__)
# Configuration and constants
LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN")


def parse_cli():
    # Create a preliminary parser just to check if --experiment-dir is provided
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--experiment-dir", required=False, default=None)

    # Parse known args to see if experiment-dir is provided
    known_args, _ = preliminary_parser.parse_known_args()

    # Build parent parsers conditionally
    parent_parsers = [
        trainer.build_trainer_parser(add_help=False),
        evaluator.build_argparser(add_help=False),
    ]

    # Only include preprocessing parser if we're not using existing experiment-dir
    if not known_args.experiment_dir:
        parent_parsers.insert(0, preproc.build_argument_parser(add_help=False))

    # Create the main argument parser
    parser = argparse.ArgumentParser(
        description="End-to-end Preprocessing -> Model Evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
        parents=parent_parsers,
        epilog="""
            Examples:
            python -m mc_classifier_pipeline.model_orchestrator --project-id 1 --train-ratio 0.7  --output-dir experiments --target-label 'Analysis' --models-config configs/quick_test.json
            python -m mc_classifier_pipeline.model_orchestrator  --experiment-dir src/mc_classifier_pipeline/experiments/project_1/20250811_092812 --target-label 'Analysis' --models-config configs/quick_test.json
            python -m mc_classifier_pipeline.model_orchestrator --project-id 10 --train-ratio 0.8 --output-dir experiments --experiment-name climate_sentiment_v1 --random-seed 123 --models-config configs/quick_test.json
        """,
    )
    parser.add_argument(
        "--experiment-dir",
        required=False,
        default=None,
        help="Experiment directory (if provided, preprocessing will be skipped)",
    )

    logger.info("Parsed command line arguments...")
    return parser.parse_args()


def validate_experiment_dir(experiment_dir: Path, args: argparse.Namespace) -> None:
    """Validate that the experiment directory exists and has expected structure."""

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    if not experiment_dir.is_dir():
        raise NotADirectoryError(f"Experiment path is not a directory: {experiment_dir}")

    # Check for expected dataset files and metadata
    expected_files = ["train.csv", "test.csv", "metadata.json"]
    missing_files = []

    for file_name in expected_files:
        file_path = experiment_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        raise FileNotFoundError(f"Missing expected files in experiment directory {experiment_dir}: {missing_files}")

    # Validate metadata.json is valid JSON
    try:
        with open(experiment_dir / "metadata.json", "r") as f:
            json.load(f)
        logger.info("metadata.json validated successfully")
    except json.JSONDecodeError as e:
        raise ValueError(f"metadata.json is not valid JSON: {e}")
    except Exception as e:
        raise ValueError(f"Could not read metadata.json: {e}")

    # Validate that required columns exist in the datasets
    if hasattr(args, "text_column") and hasattr(args, "label_column"):
        try:
            train_df = pd.read_csv(experiment_dir / "train.csv")
            required_columns = [args.text_column, args.label_column]
            missing_columns = [col for col in required_columns if col not in train_df.columns]

            if missing_columns:
                raise ValueError(
                    f"Missing required columns in train.csv: {missing_columns}. "
                    f"Available columns: {list(train_df.columns)}"
                )
        except Exception as e:
            logger.warning(f"Could not validate dataset columns: {e}")

    logger.info(f"Experiment directory validation passed: {experiment_dir}")


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations and dependencies."""

    # If experiment_dir is provided, we don't need preprocessing-specific args
    if args.experiment_dir:
        # Check that we have the minimum required args for training/evaluation
        required_for_training = ["models_config", "text_column", "label_column"]
        missing_args = [arg for arg in required_for_training if not hasattr(args, arg) or getattr(args, arg) is None]

        if missing_args:
            raise ValueError(f"When using --experiment-dir, the following arguments are required: {missing_args}")
    else:
        # Check that we have required args for preprocessing
        preprocessing_required = ["project_id"]  # Add other required preprocessing args here
        missing_args = [arg for arg in preprocessing_required if not hasattr(args, arg) or getattr(args, arg) is None]

        if missing_args:
            raise ValueError(
                f"When not using --experiment-dir, the following preprocessing arguments are required: {missing_args}"
            )


def main() -> None:  # entry-point in pyproject.toml
    args = parse_cli()

    # Validate argument combinations
    try:
        validate_args(args)
    except ValueError as e:
        logger.error(f"Argument validation failed: {e}")
        raise

    # Determine experiment directory
    if args.experiment_dir:
        # Use provided experiment directory, skip preprocessing
        experiment_dir = Path(args.experiment_dir)
        logger.info(f"Using provided experiment directory: {experiment_dir}")

        # Validate the directory exists and has expected structure
        try:
            validate_experiment_dir(experiment_dir, args)
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            logger.error(f"Experiment directory validation failed: {e}")
            raise
    else:
        # Run preprocessing to generate experiment directory
        try:
            logger.info("Running preprocessing step")
            experiment_dir_str = preproc.run_preprocessing_pipeline(args)
            experiment_dir = Path(experiment_dir_str)
            logger.info("Preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    # Prepare trainer arguments
    trainer_args = argparse.Namespace(
        experiment_dir=str(experiment_dir),  # Convert to string for trainer compatibility
        models_config=args.models_config,
        model_names=args.model_names,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    # Run training step
    try:
        logger.info("Running training step")
        trainer.main(trainer_args)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Prepare evaluator arguments
    evaluator_args = argparse.Namespace(
        experiment_dir=str(experiment_dir),  # Convert to string for evaluator compatibility
        text_column=args.text_column,
        label_column=args.label_column,
        best_metric=args.best_metric,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Run evaluation step
    try:
        logger.info("Running evaluation step")
        evaluator.main(evaluator_args)
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    # Re-load env vars in case running as script
    LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
    LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN")
    main()
