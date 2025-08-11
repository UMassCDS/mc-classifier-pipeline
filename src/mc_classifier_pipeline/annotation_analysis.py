"""
Enhanced Label Studio Annotation Analysis Module

This module provides comprehensive analysis of text-classification annotations from Label Studio projects,
including advanced inter-annotator agreement metrics and label filtering capabilities.

Key Features:
    - Advanced Agreement Metrics: Krippendorff's Alpha, Gwet's AC1, Cohen's Kappa
    - Label Filtering: Focus analysis on specific labels of interest
    - Comprehensive Output: CSV metrics, JSON data, and human-readable summaries
    - Multi-annotator Support: Handles projects with varying numbers of annotators per task
    - Single-Label Focus: Optimized for single-label classification tasks

Environment Variables:
    LABEL_STUDIO_HOST: Your Label Studio instance URL
    LABEL_STUDIO_TOKEN: Your Label Studio API token

Dependencies:
    irrCAC: Required for accurate agreement metric calculations
    label-studio-sdk: For Label Studio API integration
    numpy, pandas: For data processing and analysis

Usage:
    python -m mc_classifier_pipeline.annotation_analysis --project-id 1 --labels positive negative
"""

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from irrCAC.raw import CAC
from label_studio_sdk.client import LabelStudio

from mc_classifier_pipeline.utils import configure_logging, validate_label_studio_env

configure_logging()
logger = logging.getLogger(__name__)


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

            if labels_by_control:
                records.append(
                    {
                        "task_id": getattr(task, "id", None),
                        "annotator": annotation.get("completed_by", annotation.get("created_by", "")),
                        "text": text,
                        "labels": labels_by_control,
                        "annotation_time": annotation.get("created_at", ""),
                        "lead_time": annotation.get("lead_time", 0.0) or 0.0,
                    }
                )

    logger.info(f"Extracted {len(records)} annotations from {len(tasks)} tasks (skipped {skipped} tasks)")
    return records


def calculate_overall_agreement_metrics(multi_annotated_tasks: Dict[int, List[Dict[str, Any]]], 
                                      control: str) -> Dict[str, float]:
    """
    Calculate overall agreement metrics across multiple tasks for a specific control.
    Uses correct data structure for irrCAC: rows = subjects (tasks), columns = raters (annotators).
    
    Args:
        multi_annotated_tasks: Dictionary mapping task_id to list of annotations
        control: The control/field name to analyze
    
    Returns:
        Dictionary containing overall agreement metrics
    """
    if not multi_annotated_tasks:
        return {"krippendorff_alpha": np.nan, "gwet_ac1": np.nan, "cohen_kappa": np.nan}
    
    try:
        # Build data matrix: tasks (subjects) as rows, annotators (raters) as columns
        all_annotators = set()
        
        # First pass: collect all annotators and filter tasks with sufficient annotations
        valid_tasks = {}
        for task_id, annotations in multi_annotated_tasks.items():
            task_annotators = []
            task_labels = []
            
            for annotation in annotations:
                annotator_id = annotation.get("annotator", "unknown")
                labels = annotation.get("labels", {}).get(control, [])
                
                # Take only first label (single-label classification)
                if isinstance(labels, list) and len(labels) > 0:
                    label = str(labels[0])
                    task_annotators.append(annotator_id)
                    task_labels.append(label)
                    all_annotators.add(annotator_id)
            
            # Only include tasks with at least 2 valid annotations
            if len(task_labels) >= 2:
                valid_tasks[task_id] = list(zip(task_annotators, task_labels))
        
        if len(valid_tasks) == 0:
            logger.warning(f"No tasks with multiple valid annotations for control '{control}'")
            return {"krippendorff_alpha": np.nan, "gwet_ac1": np.nan, "cohen_kappa": np.nan}
        
        # Sort annotators for consistent column ordering
        all_annotators = sorted(all_annotators)
        
        # Build the rating matrix
        rating_matrix = []
        for task_id, annotator_label_pairs in valid_tasks.items():
            # Create row for this task
            task_row = [np.nan] * len(all_annotators)  # Initialize with NaN for missing ratings
            
            for annotator, label in annotator_label_pairs:
                annotator_idx = all_annotators.index(annotator)
                task_row[annotator_idx] = label
            
            rating_matrix.append(task_row)
        
        # Convert to pandas DataFrame
        data = pd.DataFrame(rating_matrix, columns=all_annotators)
        
        # Check if there's sufficient variation in the data
        all_labels = []
        for _, row in data.iterrows():
            valid_labels = [v for v in row.values if pd.notna(v)]
            all_labels.extend(valid_labels)
        
        unique_labels = set(all_labels)
        if len(unique_labels) < 2:
            logger.warning(f"Insufficient label variation for control '{control}' (only {len(unique_labels)} unique labels)")
            return {"krippendorff_alpha": np.nan, "gwet_ac1": np.nan, "cohen_kappa": np.nan}
        
        logger.info(f"Computing agreement metrics for control '{control}': {len(data)} tasks, {len(all_annotators)} annotators, {len(unique_labels)} unique labels")
        
        metrics = {}
        
        # Calculate Krippendorff's Alpha
        try:
            krippendorff_cac = CAC(data, weights="identity", categories=sorted(unique_labels))
            krippendorff_result = krippendorff_cac.krippendorff()
            alpha_value = krippendorff_result.get("est", {}).get("coefficient_value", np.nan)
            metrics["krippendorff_alpha"] = float(alpha_value) if not pd.isna(alpha_value) else np.nan
            logger.info(f"Krippendorff's Alpha for '{control}': {metrics['krippendorff_alpha']:.3f}")
        except Exception as e:
            logger.warning(f"Error calculating Krippendorff's alpha for '{control}': {e}")
            metrics["krippendorff_alpha"] = np.nan
        
        # Calculate Gwet's AC1
        try:
            gwet_cac = CAC(data, weights="identity", categories=sorted(unique_labels))
            gwet_result = gwet_cac.gwet()
            ac1_value = gwet_result.get("est", {}).get("coefficient_value", np.nan)
            metrics["gwet_ac1"] = float(ac1_value) if not pd.isna(ac1_value) else np.nan
            logger.info(f"Gwet's AC1 for '{control}': {metrics['gwet_ac1']:.3f}")
        except Exception as e:
            logger.warning(f"Error calculating Gwet's AC1 for '{control}': {e}")
            metrics["gwet_ac1"] = np.nan
        
        # Calculate Cohen's Kappa (only if exactly 2 annotators)
        if len(all_annotators) == 2:
            try:
                cohen_cac = CAC(data, weights="identity", categories=sorted(unique_labels))
                cohen_result = cohen_cac.cohen()
                kappa_value = cohen_result.get("est", {}).get("coefficient_value", np.nan)
                metrics["cohen_kappa"] = float(kappa_value) if not pd.isna(kappa_value) else np.nan
                logger.info(f"Cohen's Kappa for '{control}': {metrics['cohen_kappa']:.3f}")
            except Exception as e:
                logger.warning(f"Error calculating Cohen's Kappa for '{control}': {e}")
                metrics["cohen_kappa"] = np.nan
        else:
            metrics["cohen_kappa"] = np.nan
            logger.info(f"Cohen's Kappa not applicable for '{control}' (requires exactly 2 annotators, found {len(all_annotators)})")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating overall agreement metrics for '{control}': {e}")
        return {"krippendorff_alpha": np.nan, "gwet_ac1": np.nan, "cohen_kappa": np.nan}


def calculate_simple_agreement_metrics(task_annotations: List[Dict[str, Any]], control: str) -> Dict[str, Any]:
    """
    Calculate simple agreement metrics for a specific control/task combination.
    
    Args:
        task_annotations: List of annotation records for a single task
        control: The control/field name to analyze
    
    Returns:
        Dictionary containing simple agreement metrics
    """
    if len(task_annotations) < 2:
        return {
            "num_annotators": len(task_annotations),
            "agreement_ratio": np.nan,
            "all_agree": False,
            "most_common_label": None,
            "label_distribution": {}
        }
    
    # Extract labels for this control
    labels = []
    for annotation in task_annotations:
        control_labels = annotation.get("labels", {}).get(control, [])
        if isinstance(control_labels, list) and len(control_labels) > 0:
            labels.append(str(control_labels[0]))  # Take first label only
        else:
            labels.append(None)  # Missing annotation
    
    # Remove None values for agreement calculation
    valid_labels = [label for label in labels if label is not None]
    
    if not valid_labels:
        return {
            "num_annotators": len(task_annotations),
            "agreement_ratio": np.nan,
            "all_agree": False,
            "most_common_label": None,
            "label_distribution": {}
        }
    
    label_counts = Counter(valid_labels)
    most_common_label, most_common_count = label_counts.most_common(1)[0]
    agreement_ratio = most_common_count / len(valid_labels)
    all_agree = len(label_counts) == 1
    
    return {
        "num_annotators": len(valid_labels),
        "agreement_ratio": round(agreement_ratio, 3),
        "all_agree": all_agree,
        "most_common_label": most_common_label,
        "label_distribution": dict(label_counts)
    }


def analyze_annotations(
    records: List[Dict[str, Any]], required_labels: List[str]
) -> Dict[str, Any]:
    """
    Comprehensive analysis of annotations - combines all metrics.

    Args:
        records: List of annotation records
        required_labels: List of labels that must be considered; all others are filtered out
    """
    if not records:
        return {}

    # Filter by required_labels
    filtered_records = []
    for record in records:
        # Check if any label in any control matches required_labels
        has_required_label = False
        filtered_labels = {}
        
        for control, labels in (record.get("labels", {}) or {}).items():
            filtered_control_labels = [label for label in labels if label in required_labels]
            if filtered_control_labels:
                filtered_labels[control] = filtered_control_labels
                has_required_label = True
        
        if has_required_label:
            record_copy = record.copy()
            record_copy["labels"] = filtered_labels
            filtered_records.append(record_copy)
    
    records = filtered_records
    logger.info(f"Filtered to {len(records)} annotations with required labels: {required_labels}")

    if not records:
        logger.error("No annotations remain after filtering. Check your label requirements.")
        return {}

    # Basic counts
    annotator_counts = Counter(r["annotator"] for r in records)
    task_ids = set(r["task_id"] for r in records)
    task_annotation_counts = Counter(r["task_id"] for r in records)
    multi_annotated_tasks_count = sum(1 for count in task_annotation_counts.values() if count > 1)

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

    # Group annotations by task
    task_to_annots = defaultdict(list)
    for r in records:
        task_to_annots[r["task_id"]].append(r)

    multi_annotated = {tid: annots for tid, annots in task_to_annots.items() if len(annots) > 1}
    logger.info(f"Found {len(multi_annotated)} tasks with multiple annotators")
    
    # Per-task simple agreement analysis
    agreement_data = []
    all_controls = sorted(controls)
    
    for task_id, annotations in multi_annotated.items():
        per_control_metrics = {}
        
        for control in all_controls:
            metrics = calculate_simple_agreement_metrics(annotations, control)
            per_control_metrics[control] = metrics
        
        # Calculate overall agreement across all controls for this task
        valid_agreements = [
            metrics["agreement_ratio"] 
            for metrics in per_control_metrics.values() 
            if not pd.isna(metrics["agreement_ratio"])
        ]
        overall_agreement = round(float(np.mean(valid_agreements)), 3) if valid_agreements else None
        
        any_disagreement = any(
            not metrics["all_agree"] 
            for metrics in per_control_metrics.values() 
            if metrics["num_annotators"] > 1
        )
        
        agreement_data.append({
            "task_id": task_id,
            "num_annotators": len(annotations),
            "per_control_metrics": per_control_metrics,
            "overall_simple_agreement": overall_agreement,
            "has_disagreement": any_disagreement,
        })
    
    # Calculate overall agreement metrics across all tasks using advanced methods
    logger.info("Computing advanced agreement metrics across all tasks...")
    overall_agreement_metrics = {}
    for control in all_controls:
        logger.info(f"Processing control: {control}")
        metrics = calculate_overall_agreement_metrics(multi_annotated, control)
        if any(not pd.isna(v) for v in metrics.values()):
            overall_agreement_metrics[control] = metrics

    # Timing analysis
    lead_times = [r.get("lead_time", 0) for r in records if (r.get("lead_time") or 0) > 0]

    return {
        "project_summary": {
            "total_tasks": len(task_ids),
            "total_annotations": len(records),
            "unique_annotators": len(annotator_counts),
            "tasks_with_multiple_annotators": multi_annotated_tasks_count,
            "average_annotations_per_task": round(len(records) / len(task_ids), 2) if task_ids else 0,
            "required_labels": required_labels,
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
        "overall_agreement_metrics": overall_agreement_metrics,
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
        agreement_records = []
        for task_data in analysis["agreement"]:
            base_record = {
                "task_id": task_data["task_id"],
                "num_annotators": task_data["num_annotators"],
                "overall_simple_agreement": task_data.get("overall_simple_agreement"),
                "has_disagreement": task_data.get("has_disagreement")
            }
            
            # Add per-control metrics
            for control, metrics in task_data.get("per_control_metrics", {}).items():
                record = base_record.copy()
                record["control"] = control
                record.update({f"control_{k}": v for k, v in metrics.items()})
                agreement_records.append(record)
        
        if agreement_records:
            agreement_df = pd.DataFrame(agreement_records)
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

        # Simple agreement summary
        if analysis.get("agreement"):
            f.write("SIMPLE AGREEMENT SUMMARY:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Tasks with multiple annotators: {len(analysis['agreement'])}\n")
            overall_agreements = [
                a.get("overall_simple_agreement")
                for a in analysis["agreement"]
                if a.get("overall_simple_agreement") is not None
            ]
            if overall_agreements:
                f.write(f"Average overall simple agreement: {np.mean(overall_agreements):.3f}\n")
                f.write(f"Min simple agreement: {min(overall_agreements):.3f}\n")
                f.write(f"Max simple agreement: {max(overall_agreements):.3f}\n")
            
            disagreement_count = sum(1 for a in analysis["agreement"] if a.get("has_disagreement", False))
            f.write(f"Tasks with disagreement: {disagreement_count}\n")
            f.write("\n")

        # Advanced agreement metrics summary
        if analysis.get("overall_agreement_metrics"):
            f.write("ADVANCED AGREEMENT METRICS (Overall):\n")
            f.write("-" * 40 + "\n")
            for control, metrics in analysis["overall_agreement_metrics"].items():
                f.write(f"\nControl: {control}\n")
                for metric_name, value in metrics.items():
                    if not pd.isna(value):
                        f.write(f"  {metric_name.replace('_', ' ').title()}: {value:.3f}\n")
                        
                        # Add interpretation
                        if metric_name in ["krippendorff_alpha", "gwet_ac1", "cohen_kappa"]:
                            if value >= 0.8:
                                interpretation = "Excellent"
                            elif value >= 0.6:
                                interpretation = "Good"
                            elif value >= 0.4:
                                interpretation = "Moderate"
                            elif value >= 0.2:
                                interpretation = "Fair"
                            elif value >= 0.0:
                                interpretation = "Poor"
                            else:
                                interpretation = "Worse than random"
                            f.write(f"    ({interpretation} agreement)\n")
            f.write("\n")

        # Schema summary
        f.write("LABEL SCHEMA:\n")
        f.write("-" * 20 + "\n")
        schema = analysis.get("schema", {})
        f.write(f"Controls: {', '.join(schema.get('controls', []))}\n")
        for control, labels in schema.get("labels_by_control", {}).items():
            f.write(f"  {control}: {', '.join(labels)}\n")
        f.write("\n")

        # Label distribution summary
        f.write("LABEL DISTRIBUTION:\n")
        f.write("-" * 20 + "\n")
        overall_dist = analysis.get("label_distributions", {}).get("overall", {})
        total_labels = sum(overall_dist.values()) if overall_dist else 0
        for label, count in sorted(overall_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_labels) * 100 if total_labels > 0 else 0
            f.write(f"  {label}: {count} ({percentage:.1f}%)\n")

    logger.info(f"Summary report saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Label Studio text-classification annotations with advanced agreement metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis of all annotations in project 1
    python -m mc_classifier_pipeline.annotation_analysis --project-id 1 --labels positive negative neutral
    
    # Focus on specific labels 
    python -m mc_classifier_pipeline.annotation_analysis --project-id 1 --labels positive negative
    
    # Custom output directory
    python -m mc_classifier_pipeline.annotation_analysis --project-id 1 --labels relevant irrelevant --output-dir ./my_analysis

Output Files:
    - agreement_metrics_[timestamp].csv: Detailed per-task agreement metrics
    - analysis_[timestamp].json: Complete analysis data in machine-readable format  
    - summary_[timestamp].txt: Human-readable summary with key insights

Agreement Metrics:
    - Krippendorff's Alpha: Multi-annotator agreement, handles missing data well
    - Gwet's AC1: More stable with imbalanced label distributions
    - Cohen's Kappa: Classic pairwise agreement (requires exactly 2 annotators)

Agreement Interpretation:
    0.8-1.0: Excellent, 0.6-0.8: Good, 0.4-0.6: Moderate, 0.2-0.4: Fair, 0.0-0.2: Poor, <0.0: Worse than random
        """,
    )
    parser.add_argument("--project-id", type=int, default=1, help="Label Studio project ID (default: 1)")
    parser.add_argument(
        "--output-dir", type=str, default="data/annotation_analysis", help="Output directory for analysis results"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="Required labels to include in analysis; all others are filtered out",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        host, token = validate_label_studio_env()
        client = LabelStudio(base_url=host, api_key=token)

        tasks = get_project_tasks(client, args.project_id)
        records = extract_annotations(tasks)

        if not records:
            logger.error("No valid annotations found. Please check your project configuration.")
            return

        analysis = analyze_annotations(records, args.labels)
        
        if not analysis:
            logger.error("Analysis failed or returned empty results.")
            return
        
        save_analysis_results(analysis, args.output_dir)

        # Print summary to console
        logger.info("=== ANALYSIS COMPLETE ===")
        project_summary = analysis.get("project_summary", {})
        logger.info(f"Total annotations processed: {project_summary.get('total_annotations', 0)}")
        logger.info(f"Total tasks: {project_summary.get('total_tasks', 0)}")
        logger.info(f"Tasks with multiple annotators: {project_summary.get('tasks_with_multiple_annotators', 0)}")
        logger.info(f"Required labels: {', '.join(project_summary.get('required_labels', []))}")
        
        # Print simple agreement summary
        if analysis.get("agreement"):
            overall_agreements = [
                a.get("overall_simple_agreement")
                for a in analysis["agreement"]
                if a.get("overall_simple_agreement") is not None
            ]
            if overall_agreements:
                logger.info(f"Average simple agreement: {np.mean(overall_agreements):.3f}")

        # Print advanced metrics summary
        if analysis.get("overall_agreement_metrics"):
            logger.info("\n=== ADVANCED AGREEMENT METRICS ===")
            for control, metrics in analysis["overall_agreement_metrics"].items():
                logger.info(f"\nControl: {control}")
                for metric_name, value in metrics.items():
                    if not pd.isna(value):
                        logger.info(f"  {metric_name.replace('_', ' ').title()}: {value:.3f}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()