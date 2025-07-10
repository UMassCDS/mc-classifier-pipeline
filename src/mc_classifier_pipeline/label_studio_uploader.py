import argparse
import os
import logging
from pathlib import Path
import json
from io import BytesIO

import requests
from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio

from mc_classifier_pipeline.utils import configure_logging

load_dotenv()
# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


def upload_tasks(tasks: list, project_id: int, label_studio_host: str, label_studio_token: str):
    """
    Upload a list of Label Studio tasks (already filtered) to the specified project.
    """
    logger.info("Uploading %d tasks to Label Studio project %s...", len(tasks), project_id)
    if not tasks:
        logger.info("No tasks to upload after filtering; skipping API call.")
        return

    url = f"{label_studio_host.rstrip('/')}/api/projects/{project_id}/import"
    headers = {"Authorization": f"Token {label_studio_token}"}

    # Serialize tasks once, stream from memory
    payload = json.dumps(tasks, ensure_ascii=False).encode()
    buffer = BytesIO(payload)
    files = {"file": ("tasks.json", buffer, "application/json")}

    try:
        response = requests.post(url, headers=headers, files=files)
    except requests.RequestException as e:
        raise e

    if response.ok:
        logger.info("Successfully uploaded %d tasks.", len(tasks))
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
        raise ValueError(
            f"Missing environment variables: {', '.join(missing_vars)}. Please set them in your .env file."
        )

    ls_client = LabelStudio(base_url=label_studio_host, api_key=label_studio_token)

    if not args.task_file.exists():
        raise FileNotFoundError(f"Task file '{args.task_file}' not found.")

    try:
        with open(args.task_file, encoding="utf-8") as f:
            all_tasks = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Task file is not valid JSON: %s", e)
        raise

    logger.info("Task file contains %d tasks.", len(all_tasks))

    project_tasks = ls_client.tasks.list(project=args.project_id, include="id,data")

    existing_story_ids = set([t.data.get("story_id", "") for t in project_tasks.items])

    tasks_to_upload = []
    for t in all_tasks:
        sid = t.get("data", {}).get("story_id")
        if sid is None:
            logger.warning("Task missing story_id – uploading anyway.")
            tasks_to_upload.append(t)
            continue
        if sid not in existing_story_ids:
            tasks_to_upload.append(t)

    if not tasks_to_upload:
        logger.info("All tasks already uploaded to project %s; nothing to do.", args.project_id)
        return

    upload_tasks(tasks_to_upload, args.project_id, label_studio_host, label_studio_token)


if __name__ == "__main__":
    main()
