"""
End-to-end pipeline to retrieve documents and upload them to Label Studio.

Steps performed:
  1) Retrieve Media Cloud stories matching a query and write outputs:
     - Raw article JSON in `data/raw_articles/`
     - Optional CSV summary via `--output`
     - Optional Label Studio task JSON via `--label-studio-tasks`
  2) Upload the generated tasks JSON to a specified Label Studio project.

Example:
  python -m mc_classifier_pipeline.run_pipeline \
    --query "election" \
    --start-date 2025-06-01 \
    --end-date 2025-06-30 \
    --project_id 2 \
    --label-studio-tasks data/labelstudio_tasks.json
"""

from __future__ import annotations

import argparse
import logging

from mc_classifier_pipeline.utils import configure_logging
import mc_classifier_pipeline.doc_retriever as dr
import mc_classifier_pipeline.label_studio_uploader as lsu

configure_logging()
logger = logging.getLogger(__name__)


def parse_cli():
    # Create the main argument parser for the pipeline
    parser = argparse.ArgumentParser(
        description="End-to-end MediaCloud -> Label Studio pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
        # Use argument parsers from doc_retriever and label_studio_uploader as parents
        parents=[
            dr.build_arg_parser(add_help=False),
            lsu.build_uploader_parser(add_help=False),
        ],
    )
    logger.info("Parsed command line arguments...")
    return parser.parse_args()


def main() -> None:  # entry‑point in pyproject.toml
    args = parse_cli()

    # Retrieve articles & build JSON file
    logger.info("Running doc_retriever step")
    dr.main(args)

    # Upload tasks to Label Studio
    logger.info("Running label_studio_uploader step")
    lsu.main(args)


if __name__ == "__main__":
    main()
