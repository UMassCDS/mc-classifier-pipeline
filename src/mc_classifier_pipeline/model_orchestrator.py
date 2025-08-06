import argparse
import logging

from mc_classifier_pipeline.utils import configure_logging
import mc_classifier_pipeline.preprocessing as preproc
import mc_classifier_pipeline.trainer as trainer
import mc_classifier_pipeline.evaluation as evaluator

configure_logging()
logger = logging.getLogger(__name__)


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
            evaluator.build_argparser(),
        ],
    )
    parser.add_argument(
        "--experiment-dir",
        required=False,
        default=None,
        help="Experiment directory (auto-generated from preprocessing if not provided)",
    )
    logger.info("Parsed command line arguments...")
    return parser.parse_args()


def main() -> None:  # entry‑point in pyproject.toml
    args = parse_cli()

    try:
        logger.info("Running preprocessing step")
        experiment_dir = preproc.run_preprocessing_pipeline(args)
        logger.info("Preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

    trainer_args = argparse.Namespace(
        experiment_dir=experiment_dir,
        models_config=args.models_config,
        model_names=args.model_names,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    try:
        logger.info("Running training step")
        trainer.main(trainer_args)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    evaluator_args = argparse.Namespace(
        experiment_dir=str(experiment_dir),
        text_column=args.text_column,
        label_column=args.label_column,
        best_metric=args.best_metric,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    try:
        logger.info("Running evaluation step")
        evaluator.main(evaluator_args)
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
