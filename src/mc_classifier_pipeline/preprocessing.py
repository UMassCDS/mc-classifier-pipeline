import argparse
import datetime as dt
import json
import logging
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union


import pandas as pd
from label_studio_sdk.client import LabelStudio
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mc_classifier_pipeline import utils


# Configure logging
utils.configure_logging()
logger = logging.getLogger(__name__)

# Configuration and constants
LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN")


def validate_environment_variables() -> Tuple[str, str]:
    missing_vars = []
    if not LABEL_STUDIO_HOST:
        missing_vars.append("LABEL_STUDIO_HOST")
    if not LABEL_STUDIO_TOKEN:
        missing_vars.append("LABEL_STUDIO_TOKEN")

    if missing_vars:
        raise ValueError(f"Missing env variables: {', '.join(missing_vars)}. set them in your .env file.")

    return LABEL_STUDIO_HOST, LABEL_STUDIO_TOKEN


def is_multi_label_from_config(label_config: str) -> bool:
    """
    Parse the Label Studio XML config and return True if multi-label, else False.
    """
    try:
        root = ET.fromstring(label_config)
        choices = root.find(".//Choices")
        if choices is not None:
            return choices.attrib.get("choice", "single") == "multiple"
    except Exception:
        pass
    return False


def extract_labels_by_category_from_config(label_config):
    """Extract labels grouped by category from Label Studio config"""

    root = ET.fromstring(label_config)
    label_categories = {}

    for choices in root.findall(".//Choices"):
        category_name = choices.get("name")
        choice_values = [choice.get("value") for choice in choices.findall("Choice")]
        # Check multi-label for THIS specific choices element
        is_multi_label = choices.get("choice", "single") == "multiple"

        if category_name and choice_values:
            label_categories[category_name] = {"labels": choice_values, "is_multi_label": is_multi_label}

    return label_categories


def get_project_info(client: LabelStudio, project_id: int) -> Dict[str, Any]:
    """
    Get project information and labeling configuration.

    Args:
        client: Label Studio client instance
        project_id: Label Studio project ID

    Returns:
        Dictionary containing project information
    """
    logger.info(f"Fetching project information for project {project_id}")

    try:
        project = client.projects.get(id=project_id)
        project_info = {
            "id": project.id,
            "title": project.title,
            "description": getattr(project, "description", ""),
            "label_config": project.label_config,
            "created_at": getattr(project, "created_at", ""),
            "created_by": getattr(project, "created_by", ""),
        }

        logger.info(f"Successfully retrieved project: '{project_info['title']}'")
        return project_info

    except Exception as e:
        logger.error(f"Failed to fetch project {project_id}: {e}")
        raise


def download_tasks_and_annotations(client: LabelStudio, project_id: int) -> List[Dict[str, Any]]:
    """
    Download all tasks and their annotations from a Label Studio project.

    Args:
        client: Label Studio client instance
        project_id: Label Studio project ID

    Returns:
        List of tasks with their annotations
    """
    logger.info(f"Downloading tasks and annotations from project {project_id}")

    try:
        tasks = client.tasks.list(project=project_id, include="id,data,annotations,predictions")

        tasks_with_annotations = []
        annotated_count = 0

        for task in tqdm(tasks.items, desc="Processing tasks"):
            task_dict = {"id": task.id, "data": task.data, "annotations": []}

            # Get annotations for this task
            if hasattr(task, "annotations") and task.annotations:
                for annotation in task.annotations:
                    annotation_dict = {
                        "id": annotation.get("id"),
                        "result": annotation.get("result"),
                        "completed_by": annotation.get("completed_by"),
                        "created_at": annotation.get("created_at", ""),
                        "was_cancelled": annotation.get("was_cancelled", False),
                    }
                    task_dict["annotations"].append(annotation_dict)
                if task_dict["annotations"]:
                    annotated_count += 1

            tasks_with_annotations.append(task_dict)

        logger.info(f"Downloaded {len(tasks_with_annotations)} total tasks, {annotated_count} with annotations")

        return tasks_with_annotations

    except Exception as e:
        logger.error(f"Failed to download tasks from project {project_id}: {e}")
        raise


def extract_text_and_labels(
    tasks: List[Dict[str, Any]],
    target_label: Optional[Union[str, List[str]]] = None,
    is_multi_label: bool = False,
    label_categories: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract text and labels from Label Studio tasks for text classification.

    Args:
        tasks: List of tasks with annotations
        target_label: Specific label(s) to target - can be a category name, specific label, or list of labels
        is_multi_label: Boolean indicating if this is a multi-label classification task
        label_categories: Optional label categories info for preserving structure

    Returns:
        List of records with text and labels suitable for classification
    """
    logger.info(f"Extracting text and labels from {len(tasks)} tasks")
    if target_label:
        logger.info(f"Targeting label(s): '{target_label}'")

    records = []
    skipped_count = 0

    for task in tqdm(tasks, desc="Extracting labels"):
        # Skip tasks without annotations
        if not task.get("annotations") or len(task["annotations"]) == 0:
            skipped_count += 1
            continue

        # Get text from task data
        text = task["data"].get("text", "").strip()
        if not text:
            skipped_count += 1
            continue

        # Process each annotation (in case of multiple annotators)
        for annotation in task["annotations"]:
            if annotation.get("was_cancelled", False):
                continue

            # Extract labels from annotation results
            labels = []
            for result in annotation.get("result", []):
                if result.get("type") in ["choices", "textarea"]:
                    # Handle choice-type annotations
                    if "value" in result and "choices" in result["value"]:
                        labels.extend(result["value"]["choices"])
                    # Handle text-type annotations
                    elif "value" in result and "text" in result["value"]:
                        labels.append(result["value"]["text"])
                    # Handle other value formats
                    elif "value" in result:
                        value = result["value"]
                        if isinstance(value, list):
                            labels.extend(value)
                        elif isinstance(value, str):
                            labels.append(value)

            # Filter by target label(s) if specified
            if target_label:
                if isinstance(target_label, list):
                    # target_label is a list of specific labels to keep
                    labels = [label for label in labels if label in target_label]
                else:
                    # target_label is a single string - could be specific label or category
                    # Keep existing string matching logic for backward compatibility
                    labels = [
                        label for label in labels if isinstance(label, str) and target_label.lower() in label.lower()
                    ]

            # Create record if we have labels
            if labels:
                if is_multi_label:
                    # For multi-label, preserve category structure when no target specified
                    if target_label is None and label_categories:
                        # Group labels by category to preserve structure
                        categorized_labels = {}
                        label_to_category = {}
                        for cat_name, cat_info in label_categories.items():
                            for label_name in cat_info["labels"]:
                                label_to_category[label_name] = cat_name

                        for label_name in labels:
                            if label_name in label_to_category:
                                cat_name = label_to_category[label_name]
                                if cat_name not in categorized_labels:
                                    categorized_labels[cat_name] = []
                                categorized_labels[cat_name].append(label_name)

                        label = categorized_labels
                    else:
                        # Regular multi-label (target specified or simple case)
                        label = labels
                else:
                    label = labels[0] if labels else None

                if label is not None:
                    record = {
                        "text": text,
                        "label": label,
                        "task_id": task["id"],
                        "annotation_id": annotation["id"],
                        "story_id": task["data"].get("story_id", ""),
                        "title": task["data"].get("title", ""),
                        "url": task["data"].get("url", ""),
                        "language": task["data"].get("language", ""),
                        "publish_date": task["data"].get("publish_date", ""),
                        "annotated_by": annotation.get("completed_by", ""),
                        "annotated_at": annotation.get("created_at", ""),
                    }
                    records.append(record)

    logger.info(f"Extracted {len(records)} labeled records from {len(tasks)} tasks")
    logger.info(f"Skipped {skipped_count} tasks (no annotations or no text)")

    if records:
        if isinstance(records[0]["label"], str):
            # Simple string labels
            label_counts = Counter(record["label"] for record in records)
        elif isinstance(records[0]["label"], dict):
            # Categorized labels format: {"sentiment": ["Positive"], "tags": ["Opinion"]}
            all_labels = []
            for record in records:
                label_dict = record["label"]
                for category, category_labels in label_dict.items():
                    if isinstance(category_labels, list):
                        all_labels.extend(category_labels)
                    else:
                        all_labels.append(category_labels)
            label_counts = Counter(all_labels)
        else:
            # Handle multi-label tasks (list format)
            def flatten_labels(labels):
                flat = []
                for label in labels:
                    if isinstance(label, list):
                        flat.extend(flatten_labels(label))
                    else:
                        flat.append(label)
                return flat

            all_labels = []
            for record in records:
                label = record["label"]
                if isinstance(label, list):
                    all_labels.extend(flatten_labels(label))
                else:
                    all_labels.append(label)
            label_counts = Counter(all_labels)

        logger.info(f"Label distribution: {dict(label_counts)}")

    return records


def create_train_test_split(
    records: List[Dict[str, Any]], train_ratio: float, random_state: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create stratified train/test split maintaining label balance.

    Args:
        records: List of labeled records
        train_ratio: Proportion of data for training (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_records, test_records)
    """
    logger.info(f"Creating train/test split with ratio {train_ratio:.2f}")

    if not records:
        logger.warning("No records to split")
        return [], []

    df = pd.DataFrame(records)

    # Handle multi-label case by using first label for stratification
    if isinstance(records[0]["label"], list):
        logger.warning("Multi-label detected. Using first label for stratification.")
        stratify_labels = df["label"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x))
    else:
        stratify_labels = df["label"]

    label_counts = stratify_labels.value_counts()
    min_samples = label_counts.min()

    if min_samples < 2:
        logger.warning(
            f"Some labels have only {min_samples} sample(s). "
            "Cannot perform stratified split. Using random split instead."
        )
        stratify = None
    else:
        stratify = stratify_labels

    try:
        train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=random_state, stratify=stratify)

        train_records = train_df.to_dict("records")
        test_records = test_df.to_dict("records")

        logger.info(f"Split complete: {len(train_records)} training, {len(test_records)} test samples")

        return train_records, test_records

    except Exception as e:
        logger.error(f"Failed to create train/test split: {e}")
        raise


def save_data_splits(
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    output_dir: Path,
    project_id: int,
    experiment_name: Optional[str] = None,
) -> Path:
    """
    Save train and test data to CSV files in a timestamped experiment folder.

    Args:
        train_records: Training data records
        test_records: Test data records
        output_dir: Base output directory
        experiment_name: Optional experiment name (timestamp used if not provided)

    Returns:
        Path to the created experiment directory

    Raises:
        OSError: If directory creation or file writing fails
        ValueError: If data cannot be converted to DataFrame
    """
    root_dir = Path(output_dir)

    # Use timestamp as experiment_name if not provided
    if experiment_name is None:
        experiment_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        experiment_timestamp = experiment_name

    # Build the directory path as: root/experiments/project_<project_id>/<experiment_timestamp>
    experiment_dir = root_dir / f"project_{project_id}" / experiment_timestamp
    try:
        experiment_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create experiment directory {experiment_dir}: {e}")
        raise
    logger.info(f"Saving data splits to {experiment_dir}")
    try:
        if train_records:
            train_df = pd.DataFrame(train_records)
            train_file = experiment_dir / "train.csv"
            train_df.to_csv(train_file, index=False)
            logger.info(f"Training data saved: {train_file} ({len(train_records)} records)")

        if test_records:
            test_df = pd.DataFrame(test_records)
            test_file = experiment_dir / "test.csv"
            test_df.to_csv(test_file, index=False)
            logger.info(f"Test data saved: {test_file} ({len(test_records)} records)")
    except Exception as e:
        logger.error(f"Failed to save data splits: {e}")
        raise
    return experiment_dir


def create_metadata(
    project_info: Dict[str, Any],
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    args: argparse.Namespace,
    label_categories: Dict[str, Any],
    overall_is_multi_label: bool,
    actual_is_multi_label: bool,
) -> Dict[str, Any]:
    """
    Create metadata dictionary tracking experiment details.

    Args:
        project_info: Label Studio project information
        train_records: Training data records
        test_records: Test data records
        args: Command line arguments
        label_categories: Label categories extracted from config
        overall_is_multi_label: Overall multi-label setting
        actual_is_multi_label: The actual multi-label setting used for extraction

    Returns:
        Metadata dictionary
    """

    # Calculate label distributions
    def count_labels_by_category(records, label_categories):
        """Count labels grouped by category"""
        # Create reverse mapping: label -> category
        label_to_category = {}
        for category, info in label_categories.items():
            for label in info["labels"]:
                label_to_category[label] = category

        category_counts = {category: Counter() for category in label_categories.keys()}

        for record in records:
            label = record["label"]

            if isinstance(label, dict):
                # Categorized labels (Scenarios 4 & 5)
                for cat_name, cat_labels in label.items():
                    if cat_name in category_counts:
                        for label in cat_labels:
                            category_counts[cat_name][label] += 1
            else:
                # Regular labels (string or list)
                labels_to_count = [label] if isinstance(label, str) else label

                for label in labels_to_count:
                    if label in label_to_category:
                        category = label_to_category[label]
                        category_counts[category][label] += 1

        return {cat: dict(counts) for cat, counts in category_counts.items() if counts}

    train_labels = count_labels_by_category(train_records, label_categories) if train_records else {}
    test_labels = count_labels_by_category(test_records, label_categories) if test_records else {}

    # Compute Label Studio task id range
    all_records = (train_records or []) + (test_records or [])
    task_ids = [r["task_id"] for r in all_records if "task_id" in r]
    task_id_range = [min(task_ids), max(task_ids)] if task_ids else None

    metadata = {
        "experiment": {
            "created_at": dt.datetime.now().isoformat(),
            "script_version": "1.1.0",
            "data_seed": getattr(args, "random_seed", 42),
        },
        "data_split": {
            "train_ratio": args.train_ratio,
            "test_ratio": 1.0 - args.train_ratio,
            "train_samples": len(train_records),
            "test_samples": len(test_records),
            "total_samples": len(train_records) + len(test_records),
            "stratified": True,  # We attempt stratification by default
        },
        "classification_task": {
            "target_label": getattr(args, "target_label", None),
            "task_type": "multi_label_classification" if actual_is_multi_label else "single_label_classification",
            "label_categories": label_categories,
            "overall_is_multi_label": overall_is_multi_label,
            "train_label_distribution": train_labels,
            "test_label_distribution": test_labels,
        },
        "label_studio": {
            "project_id": args.project_id,
            "project_title": project_info.get("title", ""),
            "project_description": project_info.get("description", ""),
            "annotation_config": project_info.get("label_config", ""),
            "data_downloaded_at": dt.datetime.now().isoformat(),
            "task_id_range": task_id_range,
        },
        "command_line_args": vars(args),
    }

    return metadata


def save_metadata(metadata: Dict[str, Any], experiment_dir: Path) -> None:
    """
    Save experiment metadata to JSON file.

    Args:
        metadata: Metadata dictionary
        experiment_dir: Experiment directory path

    Raises:
        OSError: If file writing fails
        TypeError: If metadata cannot be serialized to JSON
    """
    metadata_file = experiment_dir / "metadata.json"

    # Convert Path objects to strings in metadata
    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(i) for i in obj]
        return obj

    metadata = convert_paths(metadata)

    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Metadata saved: {metadata_file}")
    except (OSError, TypeError) as e:
        logger.error(f"Failed to save metadata to {metadata_file}: {e}")
        raise


def build_argument_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """
    Build the argument parser for the preprocessing script.

    Args:
        add_help: Whether to add the default help argument

    Returns:
        Argument parser instance
    """
    parser = argparse.ArgumentParser(
        description="Create train/test splits from Label Studio annotations for text classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python -m mc_classifier_pipeline.preprocessing --project-id 10 --train-ratio 0.8 --output-dir experiments
            python -m mc_classifier_pipeline.preprocessing --project-id 10 --train-ratio 0.7 --target-label \"Solutions Journalism\" --output-dir experiments
            python -m mc_classifier_pipeline.preprocessing --project-id 10 --train-ratio 0.8 --output-dir experiments --experiment-name climate_sentiment_v1 --random-seed 123
        """,
        add_help=add_help,
    )

    parser.add_argument(
        "--project-id", type=int, required=True, help="Label Studio project ID to download annotations from"
    )

    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Proportion of data for training (0.0 to 1.0). Default: 0.8"
    )

    parser.add_argument(
        "--target-label", type=str, help="Specific label to target if annotation config supports multiple choices"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="experiments",
        help="Base output directory for experiment folders (default: 'experiments')",
    )

    parser.add_argument(
        "--experiment-name", type=str, help="Optional experiment name (timestamp used if not provided)"
    )

    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducible data splits. Default: 42"
    )

    return parser


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    return build_argument_parser().parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If arguments are invalid
    """
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(f"Train ratio must be between 0.0 and 1.0, got {args.train_ratio}")

    if args.project_id <= 0:
        raise ValueError(f"Project ID must be positive, got {args.project_id}")


def run_preprocessing_pipeline(args: Optional[argparse.Namespace] = None) -> Path:
    """
    Main function to execute the preprocessing pipeline.

    Args:
        args: Optional parsed arguments (for testing/integration)

    Returns:
        Path to the created experiment directory
    """
    if args is None:
        args = parse_args()

    # Validate arguments
    validate_args(args)

    logger.info("Starting Label Studio preprocessing pipeline")
    logger.info(f"Project ID: {args.project_id}")
    logger.info(f"Train ratio: {args.train_ratio}")
    logger.info(f"Output directory: {args.output_dir}")

    # Validate environment variables
    host, token = validate_environment_variables()

    # Initialize Label Studio client
    logger.info("Initializing Label Studio client")
    try:
        client = LabelStudio(base_url=host, api_key=token)
    except Exception as e:
        logger.error(f"Failed to initialize Label Studio client: {e}")
        raise

    # Get project information
    project_info = get_project_info(client, args.project_id)

    # Extract label categories from config (this gives you per-category multi-label info)
    label_categories = extract_labels_by_category_from_config(project_info["label_config"])
    logger.info(f"Label categories found: {list(label_categories.keys())}")
    for category, info in label_categories.items():
        logger.info(f"  {category}: {info['labels']} ({'multi-label' if info['is_multi_label'] else 'single-label'})")

    # Determine overall multi-label setting
    overall_is_multi_label = is_multi_label_from_config(project_info["label_config"])
    logger.info(f"Overall task type: {'multi-label' if overall_is_multi_label else 'single-label'}")

    # Download tasks and annotations
    tasks = download_tasks_and_annotations(client, args.project_id)

    # Resolve target_label and determine appropriate settings
    if args.target_label and args.target_label in label_categories:
        # User specified a category like 'sentiment'
        category_info = label_categories[args.target_label]
        target_labels = category_info["labels"]  # ['Positive', 'Negative', 'Neutral']
        is_multi_label = category_info["is_multi_label"]
        logger.info(f"Using category '{args.target_label}': {target_labels}")
        logger.info(f"Category is {'multi-label' if is_multi_label else 'single-label'}")

        # Extract records with the specific labels from this category
        records = extract_text_and_labels(tasks, target_labels, is_multi_label, label_categories)

    elif args.target_label:
        # User specified a specific label like 'Positive'
        logger.info(f"Using specific label: '{args.target_label}'")
        is_multi_label = False
        records = extract_text_and_labels(tasks, args.target_label, is_multi_label, label_categories)

    else:
        # No target specified - use overall settings
        logger.info("No target label specified, extracting all labels")
        is_multi_label = overall_is_multi_label
        records = extract_text_and_labels(tasks, None, is_multi_label, label_categories)

    if not records:
        logger.error("No labeled records found. Cannot create train/test split.")
        raise ValueError("No labeled records found in Label Studio project")

    # Create train/test split
    train_records, test_records = create_train_test_split(records, args.train_ratio, args.random_seed)

    # Save data splits
    experiment_dir = save_data_splits(
        train_records, test_records, args.output_dir, args.project_id, args.experiment_name
    )

    # Create and save metadata (now includes label categories)
    metadata = create_metadata(
        project_info, train_records, test_records, args, label_categories, overall_is_multi_label, is_multi_label
    )
    save_metadata(metadata, experiment_dir)

    logger.info("Preprocessing pipeline completed successfully")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Training samples: {len(train_records)}")
    logger.info(f"Test samples: {len(test_records)}")

    return experiment_dir


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    # Re-load env vars in case running as script
    LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
    LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN")
    _ = run_preprocessing_pipeline()
