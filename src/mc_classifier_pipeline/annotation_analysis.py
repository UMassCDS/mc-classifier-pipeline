import argparse
import os
import logging
import pandas as pd
from dotenv import load_dotenv
from collections import defaultdict, Counter
from typing import List, Dict, Any
from label_studio_sdk.client import LabelStudio
from sklearn.metrics import cohen_kappa_score

# Configure logging
from utils import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


load_dotenv()

LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN")


def validate_environment_variables():
    if not LABEL_STUDIO_HOST or not LABEL_STUDIO_TOKEN:
        raise ValueError("Missing LABEL_STUDIO_HOST or LABEL_STUDIO_TOKEN environment variables.")
    return LABEL_STUDIO_HOST, LABEL_STUDIO_TOKEN


def get_project_tasks(client: LabelStudio, project_id: int) -> List[Dict[str, Any]]:
    logger.info(f"Downloading tasks and annotations from project {project_id}")
    tasks = client.tasks.list(project=project_id, include="id,data,annotations")
    return tasks.items


def extract_annotations(tasks: List[Any]) -> List[Dict[str, Any]]:
    records = []
    for task in tasks:
        text = task.data.get("text", "").strip()
        if not text or not hasattr(task, "annotations"):
            continue
        for annotation in task.annotations:
            if annotation.get("was_cancelled", False):
                continue
            labels = []
            for result in annotation.get("result", []):
                if result.get("type") == "choices" and "choices" in result.get("value", {}):
                    labels.extend(result["value"]["choices"])
            if labels:
                records.append(
                    {
                        "task_id": task.id,
                        "annotator": annotation.get("completed_by", ""),
                        "label": labels[0] if len(labels) == 1 else labels,
                        "text": text,
                    }
                )
    return records


def calculate_agreement(records: List[Dict[str, Any]]) -> pd.DataFrame:
    # Group annotations by task
    task_annotations = defaultdict(list)
    for r in records:
        task_annotations[r["task_id"]].append((r["annotator"], r["label"]))
    # Only consider tasks with multiple annotators
    multi_annotated = {tid: annots for tid, annots in task_annotations.items() if len(annots) > 1}
    agreement_data = []
    for tid, annots in multi_annotated.items():
        annotators, labels = zip(*annots)

        # Convert any list labels to strings for hashing
        def label_to_str(label):
            if isinstance(label, list):
                return ",".join(map(str, label))
            return str(label)

        labels_str = [label_to_str(label) for label in labels]
        # If more than 2 annotators, calculate pairwise kappa
        kappas = []
        for i in range(len(labels_str)):
            for j in range(i + 1, len(labels_str)):
                # Only calculate kappa for single-label tasks
                if "," in labels_str[i] or "," in labels_str[j]:
                    continue
                kappas.append(cohen_kappa_score([labels_str[i]], [labels_str[j]]))
        majority_label = Counter(labels_str).most_common(1)[0][0]
        agreement_data.append(
            {
                "task_id": tid,
                "num_annotators": len(annotators),
                "labels": labels,
                "majority_label": majority_label,
                "pairwise_kappa_mean": round(sum(kappas) / len(kappas), 3) if kappas else None,
                "all_agree": len(set(labels_str)) == 1,
            }
        )
    return pd.DataFrame(agreement_data)


def calculate_overall_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    annotator_counts = Counter(r["annotator"] for r in records)
    label_counts = Counter(r["label"] for r in records if isinstance(r["label"], str))
    num_tasks = len(set(r["task_id"] for r in records))
    num_annotations = len(records)
    return {
        "num_tasks": num_tasks,
        "num_annotations": num_annotations,
        "num_annotators": len(annotator_counts),
        "label_distribution": dict(label_counts),
        "annotations_per_annotator": dict(annotator_counts),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate inter-annotator agreement and metrics from Label Studio annotations"
    )

    parser.add_argument(
        "--project-id", type=int, required=False, default=1, help="Label Studio project ID (default: 1)"
    )
    parser.add_argument("--output-csv", type=str, default="data/agreement_metrics.csv", help="Output CSV file")
    args = parser.parse_args()

    host, token = validate_environment_variables()
    client = LabelStudio(base_url=host, api_key=token)
    tasks = get_project_tasks(client, args.project_id)
    records = extract_annotations(tasks)
    agreement_df = calculate_agreement(records)

    # Sort by task_id
    agreement_df = agreement_df.sort_values("task_id")
    agreement_df.to_csv(args.output_csv, index=False)
    logger.info(f"Agreement metrics saved to {args.output_csv}")


if __name__ == "__main__":
    main()
