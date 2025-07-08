import argparse
import os
import logging
from pathlib import Path

import requests
from dotenv import load_dotenv
from mc_classifier_pipeline.utils import configure_logging

load_dotenv()
# Configure logging
configure_logging()
logger = logging.getLogger(__name__)




def upload_tasks(task_file: Path, project_id: int, label_studio_host: str, label_studio_token: str):
    """
    Uploads a JSON task file to Label Studio via API.

    Args:
        task_file: Path to the formatted Label Studio tasks JSON file
    """
    logger.info("Uploading tasks from '%s' to Label Studio project %s...", task_file, project_id)
    url = f"{label_studio_host.rstrip('/')}/api/projects/{project_id}/import"
    headers = {"Authorization": f"Token {label_studio_token}"}

    try:
        with open(task_file, "rb") as f:
            files = {"file": f}
            response = requests.post(url, headers=headers, files=files)
    except requests.RequestException as e:
        logger.error("Request failed: %s", e)
        raise e

    if response.ok:
        logger.info("Successfully uploaded '%s' to project %s.", task_file.name, project_id)
    else:
        logger.error("Upload failed (%s): %s", response.status_code, response.text)
        raise requests.RequestException(f"Upload failed with response {response.status_code}: {response.text}")


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Upload formatted Label Studio tasks JSON to a specified Label Studio project via API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Upload the default tasks file to Label Studio project with id 4
            python src/mc_classifier_pipeline/label_studio_uploader.py --project_id 4
            # Upload a custom tasks file to Label Studio project with id 100
            python src/mc_classifier_pipeline/label_studio_uploader.py data/custom_tasks.json -p 100
        """,
    )

    parser.add_argument(
        "task_file",
        type=Path,
        nargs="?",
        default=Path("data/labelstudio_tasks.json"),
        help="Path to the Label Studio tasks JSON file (default: data/labelstudio_tasks.json)",
    )
    parser.add_argument(
        "--project_id",
        "-p",
        type=int,
        required=True,
        help="The project id for the Label Studio project where tasks will be added",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    label_studio_host = os.getenv("LABEL_STUDIO_HOST")
    label_studio_token = os.getenv("LABEL_STUDIO_TOKEN")
    missing_vars = []
    if not label_studio_host:
        missing_vars.append("LABEL_STUDIO_HOST")
    if not label_studio_token:
        missing_vars.append("LABEL_STUDIO_TOKEN")

    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}. Please set them in your .env file.")
        raise SystemExit(1)
    if not args.task_file.exists():
        logger.error("Task file '%s' not found.", args.task_file)
        raise SystemExit(1)
    upload_tasks(args.task_file, args.project_id, label_studio_host, label_studio_token)


if __name__ == "__main__":
    main()
