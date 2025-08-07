import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
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
    """Download tasks and annotations from Label Studio project."""
    try:
        logger.info(f"Downloading tasks and annotations from project {project_id}")
        tasks = client.tasks.list(project=project_id, include="id,data,annotations")
        logger.info(f"Retrieved {len(tasks.items)} tasks from project {project_id}")
        return tasks.items
    except Exception as e:
        logger.error(f"Error retrieving tasks from project {project_id}: {e}")
        raise


def extract_annotations(tasks: List[Any]) -> List[Dict[str, Any]]:
    """Extract and validate single-label annotations from tasks."""
    records = []
    skipped_tasks = 0

    for task in tasks:
        text = task.data.get("text", "").strip()
        if not text:
            skipped_tasks += 1
            continue

        if not hasattr(task, "annotations") or not task.annotations:
            skipped_tasks += 1
            continue

        for annotation in task.annotations:
            # Skip cancelled or rejected annotations
            if annotation.get("was_cancelled", False) or annotation.get("was_cancelled", False):
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
                        "label": labels[0],
                        "text": text,
                        "annotation_time": annotation.get("created_at", ""),
                        "lead_time": annotation.get("lead_time", 0),
                    }
                )

    logger.info(f"Extracted {len(records)} annotations from {len(tasks)} tasks (skipped {skipped_tasks} tasks)")
    return records


def calculate_agreement(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Calculate inter-annotator agreement metrics."""
    if not records:
        logger.warning("No annotations found for agreement calculation")
        return pd.DataFrame()

    # Group annotations by task
    task_annotations = defaultdict(list)
    for r in records:
        # Handle single-label cases only
        label = str(r["label"])
        task_annotations[r["task_id"]].append((r["annotator"], label))

    # Only consider tasks with multiple annotators
    multi_annotated = {tid: annots for tid, annots in task_annotations.items() if len(annots) > 1}

    if not multi_annotated:
        logger.warning("No tasks with multiple annotators found")
        return pd.DataFrame()

    agreement_data = []
    for tid, annots in multi_annotated.items():
        annotators, labels = zip(*annots)

        # Calculate agreement metrics
        unique_labels = set(labels)
        num_annotators = len(annotators)

        # Simple agreement (percentage of annotators who agree)
        most_common_label, most_common_count = Counter(labels).most_common(1)[0]
        simple_agreement = most_common_count / num_annotators

        # Calculate pairwise kappa only for 2 annotators
        pairwise_kappa = None
        if (
            num_annotators == 2 and len(unique_labels) > 1
        ):  # Only calculate if there are exactly 2 annotators and disagreement
            try:
                pairwise_kappa = cohen_kappa_score([labels[0]], [labels[1]])
                if np.isnan(pairwise_kappa):
                    pairwise_kappa = None
            except Exception as e:
                logger.debug(f"Could not calculate kappa for task {tid}: {e}")
                pairwise_kappa = None

        agreement_data.append(
            {
                "task_id": tid,
                "num_annotators": num_annotators,
                "labels": list(labels),
                "annotators": list(annotators),
                "unique_labels": len(unique_labels),
                "majority_label": most_common_label,
                "simple_agreement": round(simple_agreement, 3),
                "pairwise_kappa": round(pairwise_kappa, 3) if pairwise_kappa is not None else None,
                "all_agree": len(unique_labels) == 1,
                "disagreement_level": "high"
                if simple_agreement < 0.5
                else "medium"
                if simple_agreement < 0.8
                else "low",
            }
        )

    return pd.DataFrame(agreement_data)


def calculate_overall_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive project-level metrics."""
    if not records:
        return {}

    # Basic counts
    annotator_counts = Counter(r["annotator"] for r in records)
    task_ids = set(r["task_id"] for r in records)

    # Label distribution
    label_counts = Counter(str(r["label"]) for r in records)

    # Annotation timing analysis
    lead_times = [r.get("lead_time", 0) for r in records if r.get("lead_times", 0) > 0]

    # Tasks with multiple annotators
    task_annotation_counts = Counter(r["task_id"] for r in records)
    multi_annotated_tasks = sum(1 for count in task_annotation_counts.values() if count > 1)

    return {
        "project_summary": {
            "total_tasks": len(task_ids),
            "total_annotations": len(records),
            "unique_annotators": len(annotator_counts),
            "tasks_with_multiple_annotators": multi_annotated_tasks,
            "average_annotations_per_task": round(len(records) / len(task_ids), 2) if task_ids else 0,
        },
        "annotator_stats": {
            "annotations_per_annotator": dict(annotator_counts),
            "most_active_annotator": annotator_counts.most_common(1)[0] if annotator_counts else None,
            "least_active_annotator": annotator_counts.most_common()[-1] if annotator_counts else None,
        },
        "label_distribution": dict(label_counts),
        "timing_stats": {
            "average_lead_time": round(np.mean(lead_times), 2) if lead_times else None,
            "median_lead_time": round(np.median(lead_times), 2) if lead_times else None,
            "min_lead_time": min(lead_times) if lead_times else None,
            "max_lead_time": max(lead_times) if lead_times else None,
        },
    }


def calculate_annotator_consistency(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Calculate consistency metrics for each annotator."""
    if not records:
        return pd.DataFrame()

    annotator_data = defaultdict(lambda: {"annotations": [], "labels": [], "tasks": set()})

    for r in records:
        annotator = r["annotator"]
        label = r["label"]
        task_id = r["task_id"]

        annotator_data[annotator]["annotations"].append(r)
        annotator_data[annotator]["labels"].append(label)
        annotator_data[annotator]["tasks"].add(task_id)

    consistency_data = []
    for annotator, data in annotator_data.items():
        labels = data["labels"]
        label_counts = Counter()

        for label in labels:
            label_counts[str(label)] += 1

        # Calculate label distribution consistency
        total_annotations = len(labels)
        most_common_label, most_common_count = label_counts.most_common(1)[0]
        consistency_ratio = most_common_count / total_annotations if total_annotations > 0 else 0

        consistency_data.append(
            {
                "annotator": annotator,
                "total_annotations": total_annotations,
                "unique_tasks": len(data["tasks"]),
                "unique_labels_used": len(label_counts),
                "most_common_label": most_common_label,
                "consistency_ratio": round(consistency_ratio, 3),
                "label_distribution": dict(label_counts),
            }
        )

    return pd.DataFrame(consistency_data)


def generate_analysis_report(
    agreement_df: pd.DataFrame,
    overall_metrics: Dict[str, Any],
    annotator_consistency_df: pd.DataFrame,
    output_dir: str,
):
    """Generate comprehensive analysis report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save agreement metrics
    agreement_file = os.path.join(output_dir, f"agreement_metrics_{timestamp}.csv")
    agreement_df.to_csv(agreement_file, index=False)
    logger.info(f"Agreement metrics saved to {agreement_file}")

    # Save overall metrics
    metrics_file = os.path.join(output_dir, f"overall_metrics_{timestamp}.json")
    with open(metrics_file, "w") as f:
        json.dump(overall_metrics, f, indent=2, default=str)
    logger.info(f"Overall metrics saved to {metrics_file}")

    # Save annotator consistency
    if not annotator_consistency_df.empty:
        consistency_file = os.path.join(output_dir, f"annotator_consistency_{timestamp}.csv")
        annotator_consistency_df.to_csv(consistency_file, index=False)
        logger.info(f"Annotator consistency saved to {consistency_file}")

    # Generate summary report
    summary_file = os.path.join(output_dir, f"analysis_summary_{timestamp}.txt")
    with open(summary_file, "w") as f:
        f.write("ANNOTATION ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        # Project summary
        f.write("PROJECT SUMMARY:\n")
        f.write("-" * 20 + "\n")
        for key, value in overall_metrics.get("project_summary", {}).items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")

        # Agreement summary
        if not agreement_df.empty:
            f.write("AGREEMENT SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Tasks with multiple annotators: {len(agreement_df)}\n")
            f.write(f"Tasks with perfect agreement: {agreement_df['all_agree'].sum()}\n")
            f.write(f"Average simple agreement: {agreement_df['simple_agreement'].mean():.3f}\n")
            if agreement_df["pairwise_kappa"].notna().any():
                f.write(f"Average pairwise kappa: {agreement_df['pairwise_kappa'].mean():.3f}\n")
            f.write("\n")

        # Annotator summary
        f.write("ANNOTATOR SUMMARY:\n")
        f.write("-" * 20 + "\n")
        annotator_stats = overall_metrics.get("annotator_stats", {})
        if annotator_stats.get("most_active_annotator"):
            f.write(
                f"Most active annotator: {annotator_stats['most_active_annotator'][0]} ({annotator_stats['most_active_annotator'][1]} annotations)\n"
            )
        f.write("\n")

    logger.info(f"Analysis summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate comprehensive inter-annotator agreement and metrics from single-label Label Studio annotations"
    )

    parser.add_argument(
        "--project-id", type=int, required=False, default=1, help="Label Studio project ID (default: 1)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/annotation_analysis", help="Output directory for analysis files"
    )
    parser.add_argument("--include-text", action="store_true", help="Include text content in output files")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        host, token = validate_environment_variables()
        client = LabelStudio(base_url=host, api_key=token)

        # Get and process data
        tasks = get_project_tasks(client, args.project_id)
        records = extract_annotations(tasks)

        if not records:
            logger.error("No valid annotations found. Please check your project configuration.")
            return

        # Calculate metrics
        agreement_df = calculate_agreement(records)
        overall_metrics = calculate_overall_metrics(records)
        annotator_consistency_df = calculate_annotator_consistency(records)

        # Generate comprehensive report
        generate_analysis_report(agreement_df, overall_metrics, annotator_consistency_df, args.output_dir)

        # Print summary to console
        logger.info("=== ANALYSIS COMPLETE ===")
        logger.info(f"Total annotations processed: {overall_metrics['project_summary']['total_annotations']}")
        logger.info(
            f"Tasks with multiple annotators: {overall_metrics['project_summary']['tasks_with_multiple_annotators']}"
        )
        if not agreement_df.empty:
            logger.info(f"Average agreement: {agreement_df['simple_agreement'].mean():.3f}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
