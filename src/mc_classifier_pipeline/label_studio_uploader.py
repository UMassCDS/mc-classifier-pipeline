import argparse
import os
import logging
from pathlib import Path

import requests
from dotenv import load_dotenv
from . import utils

load_dotenv()

LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN")
PROJECT_ID = os.getenv("LABEL_STUDIO_PROJECT_ID")

# Configure logging
utils.configure_logging()
logger = logging.getLogger(__name__)

missing_vars = [k for k in ["LABEL_STUDIO_HOST", "LABEL_STUDIO_TOKEN", "LABEL_STUDIO_PROJECT_ID"] if not globals().get(k)]
if missing_vars:
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}. Please set them in your .env file.")
    raise SystemExit(1)


def upload_tasks(task_file: Path):
    """
    Uploads a JSON task file to Label Studio via API.

    Args:
        task_file: Path to the formatted Label Studio tasks JSON file
    """
    url = f"{LABEL_STUDIO_HOST.rstrip('/')}/api/projects/{PROJECT_ID}/import"
    headers = {"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}

    try:
        with open(task_file, "rb") as f:
            files = {"file": f}
            response = requests.post(url, headers=headers, files=files)
    except requests.RequestException as e:
        logger.error("Request failed: %s", e)
        raise SystemExit(1) from e

    if response.ok:
        logger.info("Successfully uploaded '%s' to project %s.", task_file.name, PROJECT_ID)
    else:
        logger.error("Upload failed (%s): %s", response.status_code, response.text)
        raise SystemExit(1)


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Upload formatted Label Studio tasks JSON to a specified Label Studio project via API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Upload the default tasks file
            python src/mc_classifier_pipeline/LS_uploader.py
            # Upload a custom tasks file
            python src/mc_classifier_pipeline/LS_uploader.py data/custom_tasks.json

        """,
    )

    parser.add_argument(
        "task_file",
        type=Path,
        nargs="?",
        default=Path("data/labelstudio_tasks.json"),
        help="Path to the Label Studio tasks JSON file (default: data/labelstudio_tasks.json)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if not args.task_file.exists():
        logger.error("Task file '%s' not found.", args.task_file)
        raise SystemExit(1)

    logger.info("Uploading tasks from '%s' to Label Studio project %s...", args.task_file, PROJECT_ID)
    upload_tasks(args.task_file)


if __name__ == "__main__":
    main()
