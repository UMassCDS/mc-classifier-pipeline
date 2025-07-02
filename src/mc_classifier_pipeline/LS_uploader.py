import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN")
PROJECT_ID = os.getenv("LABEL_STUDIO_PROJECT_ID")

if not LABEL_STUDIO_HOST or not LABEL_STUDIO_TOKEN or not PROJECT_ID:
    sys.stderr.write(
        "ERROR: Please set LABEL_STUDIO_HOST, LABEL_STUDIO_TOKEN, and LABEL_STUDIO_PROJECT_ID in your .env file.\n"
    )
    sys.exit(1)


def upload_tasks(task_file: Path):
    """
    Uploads a JSON task file to Label Studio via API.

    Args:
        task_file: Path to the Label Studio–formatted tasks JSON file
    """
    url = f"{LABEL_STUDIO_HOST.rstrip('/')}" \
          f"/api/projects/{PROJECT_ID}/import"
    headers = {"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}

    with open(task_file, "rb") as f:
        files = {"file": f}
        response = requests.post(url, headers=headers, files=files)

    if response.ok:
        print(f"✔ Successfully uploaded '{task_file.name}' to project {PROJECT_ID}.")
    else:
        sys.stderr.write(
            f"❌ Upload failed ({response.status_code}): {response.text}\n"
        )
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload Label Studio–formatted tasks JSON to a specified Label Studio project via API."
    )
    parser.add_argument(
        "task_file",
        type=Path,
        nargs="?",
        default=Path("data/labelstudio_tasks.json"),
        help="Path to the Label Studio tasks JSON file (default: data/labelstudio_tasks.json)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.task_file.exists():
        sys.stderr.write(f"ERROR: Task file '{args.task_file}' not found.\n")
        sys.exit(1)

    print(f"Uploading tasks from '{args.task_file}' to Label Studio project {PROJECT_ID}...")
    upload_tasks(args.task_file)


if __name__ == "__main__":
    main()
