"""
End-to-end pipeline:
  1. Query documents -> JSON tasks
  2. Push those tasks to a chosen Label Studio project
Usage example:
  python -m mc_classifier_pipeline.run_pipeline.py --query "election" --start 2025-06-01 --end 2025-06-30 --project_id 2
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
    parser = argparse.ArgumentParser(
        description="End-to-end MediaCloud -> Label Studio pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
