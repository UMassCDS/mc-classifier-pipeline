import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mc_classifier_pipeline.utils import configure_logging
import mc_classifier_pipeline.preprocessing as preproc
import mc_classifier_pipeline.trainer as trainer
import mc_classifier_pipeline.evaluation as evaluator

configure_logging()
logger = logging.getLogger(__name__)

def parse_experiment_metadata(experiment_dir: Path) -> Dict[str, Any]:
    """
    Parse experiment metadata JSON and extract key variables.
    
    Args:
        experiment_dir: Path to the experiment directory containing metadata.json
        
    Returns:
        dict: Parsed experiment variables
    """
    experiment_path = experiment_dir
    metadata_file = experiment_path / "metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    # Load JSON metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        # Handle target_label - infer from annotation config if null
        target_label = metadata['classification_task']['target_label']
        if target_label is None:
            # Try to infer from annotation config - look for the single Choices element
            annotation_config = metadata['label_studio']['annotation_config']
            import re
            choices_matches = re.findall(r'<Choices[^>]*name="([^"]*)"', annotation_config)
            if choices_matches:
                target_label = choices_matches[0]  # Use the first (and likely only) Choices name
            else:
                raise ValueError("target_label is null and cannot infer label column from annotation config")
    
    # Extract experiment variables
    experiment_vars = {
        # Experiment info
        'created_at': metadata['experiment']['created_at'],
        'script_version': metadata['experiment']['script_version'],
        'data_seed': metadata['experiment']['data_seed'],
        
        # Data split info
        'train_ratio': metadata['data_split']['train_ratio'],
        'test_ratio': metadata['data_split']['test_ratio'],
        'train_samples': metadata['data_split']['train_samples'],
        'test_samples': metadata['data_split']['test_samples'],
        'total_samples': metadata['data_split']['total_samples'],
        'stratified': metadata['data_split']['stratified'],
        
        # Classification task info
        'target_label': metadata['classification_task']['target_label'],
        'task_type': metadata['classification_task']['task_type'],
        'train_label_distribution': metadata['classification_task']['train_label_distribution'],
        'test_label_distribution': metadata['classification_task']['test_label_distribution'],
        'unique_labels': metadata['classification_task']['unique_labels'],
        
        # Label Studio info
        'project_id': metadata['label_studio']['project_id'],
        'project_title': metadata['label_studio']['project_title'],
        'project_description': metadata['label_studio']['project_description'],
        'data_downloaded_at': metadata['label_studio']['data_downloaded_at'],
        'task_id_range': metadata['label_studio']['task_id_range'],
        
        # Command line args
        'original_project_id': metadata['command_line_args']['project_id'],
        'original_train_ratio': metadata['command_line_args']['train_ratio'],
        'original_target_label': metadata['command_line_args']['target_label'],
        'output_dir': metadata['command_line_args']['output_dir'],
        'experiment_name': metadata['command_line_args']['experiment_name'],
        'random_seed': metadata['command_line_args']['random_seed'],
        
        # Derived paths
        'experiment_dir': str(experiment_path),
        'train_csv_path': str(experiment_path / "train.csv"),
        'test_csv_path': str(experiment_path / "test.csv"),
        'metadata_path': str(metadata_file)
    }

    return experiment_vars


def parse_cli():
    # Create the main argument parser for the pipeline
    parser = argparse.ArgumentParser(
        description="End-to-end Preprocessing -> Model Evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
        # Use argument parsers from doc_retriever and label_studio_uploader as parents
        parents=[
            preproc.build_argument_parser(add_help=False),
            trainer.build_trainer_parser(add_help=False),
            evaluator.build_argparser()
        ],
    )
    parser.add_argument('--experiment-dir', required=False, default=None,
                       help='Experiment directory (auto-generated from preprocessing if not provided)')
    logger.info("Parsed command line arguments...")
    return parser.parse_args()

def main() -> None:  # entry‑point in pyproject.toml
    args = parse_cli()

    try:
        logger.info("Running preprocessing step")
        preproc.run_preprocessing_pipeline(args)
        logger.info("Preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

    experiment_dir = preproc.get_experiment_dir(args.project_id, args.output_dir, args.experiment_name)
    args.experiment_dir = experiment_dir
    experiment_vars = parse_experiment_metadata(experiment_dir)

    models = trainer.load_models_config(args.models_config, args.model_names)
    trainer.train_all_models(args.experiment_dir, models, args.text_column, args.label_column)

    results, summary = evaluator.evaluate_models(
        experiment_dir=str(args.experiment_dir),  # Convert to string
        text_column=args.text_column,
        label_column=args.label_column,
        best_metric=args.best_metric,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    
    models_root = str(os.path.join(args.experiment_dir, "models"))  # Convert to string
    evaluator.write_outputs(models_root, results, summary)


if __name__ == "__main__":
    main()