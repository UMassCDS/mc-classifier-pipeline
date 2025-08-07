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

from utils import configure_logging

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
    """Extract text-classification annotations from Label Studio tasks."""
    records = []
    skipped = 0

    for task in tasks:
        text = (getattr(task, "data", {}) or {}).get("text", "")
        if isinstance(text, str):
            text = text.strip()
        else:
            text = str(text) if text is not None else ""
        if not text:
            skipped += 1
            continue

        ann_list = getattr(task, "annotations", None) or []
        if not ann_list:
            skipped += 1
            continue

        for annotation in ann_list:
            if annotation.get("was_cancelled") or annotation.get("cancelled") or annotation.get("skipped"):
                continue

            result_items = annotation.get("result", []) or []
            cls_results = [
                r
                for r in result_items
                if r.get("type") == "choices" and isinstance(r.get("value", {}).get("choices"), list)
            ]
            if not cls_results:
                continue

            labels_by_control = {}
            meta = {}

            for r in cls_results:
                control_name = r.get("from_name") or "label"
                value = r.get("value", {}) or {}
                labels = [str(v) for v in value.get("choices", [])]
                if labels:
                    existing = labels_by_control.get(control_name, [])
                    for lbl in labels:
                        if lbl not in existing:
                            existing.append(lbl)
                    labels_by_control[control_name] = existing

                # Extract meta
                if isinstance(r.get("meta"), dict):
                    meta.update(r["meta"])
                if "score" in r:
                    meta.setdefault("score", r.get("score"))
                if "confidence" in value:
                    meta.setdefault("confidence", value.get("confidence"))
                if "explanation" in value:
                    meta.setdefault("explanation", value.get("explanation"))

            if labels_by_control:
                records.append(
                    {
                        "task_id": getattr(task, "id", None),
                        "annotator": annotation.get("completed_by", annotation.get("created_by", "")),
                        "text": text,
                        "labels": labels_by_control,
                        "meta": meta,
                        "annotation_time": annotation.get("created_at", ""),
                        "lead_time": annotation.get("lead_time", 0.0) or 0.0,
                    }
                )

    logger.info(f"Extracted {len(records)} annotations from {len(tasks)} tasks (skipped {skipped} tasks)")
    return records


def analyze_annotations(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Comprehensive analysis of annotations - combines all metrics."""
    if not records:
        return {}

    # Basic counts
    annotator_counts = Counter(r["annotator"] for r in records)
    task_ids = set(r["task_id"] for r in records)
    task_annotation_counts = Counter(r["task_id"] for r in records)
    multi_annotated_tasks = sum(1 for count in task_annotation_counts.values() if count > 1)

    # Label distributions
    overall_counter = Counter()
    control_to_counter = defaultdict(Counter)
    annotator_to_control_counter = defaultdict(lambda: defaultdict(Counter))
    controls = set()
    labels_by_control = defaultdict(set)

    for r in records:
        annotator = r.get("annotator")
        labels_by_control_dict = r.get("labels", {}) or {}
        for control, labels in labels_by_control_dict.items():
            controls.add(control)
            for lbl in labels:
                lbl_str = str(lbl)
                overall_counter[lbl_str] += 1
                control_to_counter[control][lbl_str] += 1
                annotator_to_control_counter[annotator][control][lbl_str] += 1
                labels_by_control[control].add(lbl_str)

    # Agreement analysis
    task_to_annots = defaultdict(list)
    for r in records:
        task_to_annots[r["task_id"]].append(r)

    multi_annotated = {tid: annots for tid, annots in task_to_annots.items() if len(annots) > 1}
    agreement_data = []

    for tid, annots in multi_annotated.items():
        task_controls = set()
        for a in annots:
            task_controls.update((a.get("labels") or {}).keys())

        per_control_agreement = {}
        per_control_all_agree = {}

        for control in sorted(task_controls):
            selections = []
            for a in annots:
                labels_list = (a.get("labels") or {}).get(control, []) or []
                canon = tuple(sorted(set(str(x) for x in labels_list)))
                selections.append(canon)

            if selections:
                counts = Counter(selections)
                most_common_set, most_common_count = counts.most_common(1)[0]
                ratio = most_common_count / len(selections)
                per_control_agreement[control] = round(ratio, 3)
                per_control_all_agree[control] = len(counts) == 1

        valid_agreements = [v for v in per_control_agreement.values() if not np.isnan(v)]
        overall_agreement = round(float(np.mean(valid_agreements)), 3) if valid_agreements else None
        any_disagreement = any(not v for v in per_control_all_agree.values()) if per_control_all_agree else False

        agreement_data.append(
            {
                "task_id": tid,
                "num_annotators": len(annots),
                "per_control_simple_agreement": per_control_agreement,
                "per_control_all_agree": per_control_all_agree,
                "overall_simple_agreement": overall_agreement,
                "has_disagreement": any_disagreement,
            }
        )

    # Timing analysis
    lead_times = [r.get("lead_time", 0) for r in records if (r.get("lead_time") or 0) > 0]

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
        "label_distributions": {
            "overall": dict(overall_counter),
            "per_control": {c: dict(cnt) for c, cnt in control_to_counter.items()},
            "per_annotator": {
                ann: {c: dict(cnt) for c, cnt in controls.items()}
                for ann, controls in annotator_to_control_counter.items()
            },
        },
        "schema": {
            "controls": sorted(controls),
            "labels_by_control": {c: sorted(list(s)) for c, s in labels_by_control.items()},
        },
        "agreement": agreement_data,
        "timing_stats": {
            "average_lead_time": round(np.mean(lead_times), 2) if lead_times else None,
            "median_lead_time": round(np.median(lead_times), 2) if lead_times else None,
            "min_lead_time": min(lead_times) if lead_times else None,
            "max_lead_time": max(lead_times) if lead_times else None,
        },
    }


def save_analysis_results(analysis: Dict[str, Any], output_dir: str):
    """Save analysis results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save agreement metrics
    if analysis.get("agreement"):
        agreement_df = pd.DataFrame(analysis["agreement"])
        agreement_file = os.path.join(output_dir, f"agreement_metrics_{timestamp}.csv")
        agreement_df.to_csv(agreement_file, index=False)
        logger.info(f"Agreement metrics saved to {agreement_file}")

    # Save overall analysis
    analysis_file = os.path.join(output_dir, f"analysis_{timestamp}.json")
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"Complete analysis saved to {analysis_file}")

    # Generate summary report
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_file, "w") as f:
        f.write("ANNOTATION ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        # Project summary
        f.write("PROJECT SUMMARY:\n")
        f.write("-" * 20 + "\n")
        for key, value in analysis.get("project_summary", {}).items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")

        # Agreement summary
        if analysis.get("agreement"):
            f.write("AGREEMENT SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Tasks with multiple annotators: {len(analysis['agreement'])}\n")
            overall_agreements = [
                a.get("overall_simple_agreement")
                for a in analysis["agreement"]
                if a.get("overall_simple_agreement") is not None
            ]
            if overall_agreements:
                f.write(f"Average overall simple agreement: {np.mean(overall_agreements):.3f}\n")
            f.write("\n")

        # Schema summary
        f.write("LABEL SCHEMA:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Controls: {', '.join(analysis.get('schema', {}).get('controls', []))}\n")
        f.write("\n")

    logger.info(f"Summary report saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Label Studio text-classification annotations")
    parser.add_argument("--project-id", type=int, default=1, help="Label Studio project ID (default: 1)")
    parser.add_argument("--output-dir", type=str, default="data/annotation_analysis", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        host, token = validate_environment_variables()
        client = LabelStudio(base_url=host, api_key=token)

        tasks = get_project_tasks(client, args.project_id)
        records = extract_annotations(tasks)

        if not records:
            logger.error("No valid annotations found. Please check your project configuration.")
            return

        analysis = analyze_annotations(records)
        save_analysis_results(analysis, args.output_dir)

        # Print summary to console
        logger.info("=== ANALYSIS COMPLETE ===")
        logger.info(f"Total annotations processed: {analysis['project_summary']['total_annotations']}")
        logger.info(f"Tasks with multiple annotators: {analysis['project_summary']['tasks_with_multiple_annotators']}")
        if analysis.get("agreement"):
            overall_agreements = [
                a.get("overall_simple_agreement")
                for a in analysis["agreement"]
                if a.get("overall_simple_agreement") is not None
            ]
            if overall_agreements:
                logger.info(f"Average overall simple agreement: {np.mean(overall_agreements):.3f}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
