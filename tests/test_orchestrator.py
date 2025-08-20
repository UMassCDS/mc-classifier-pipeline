import subprocess
import tempfile
import os
import pytest
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import glob
import pandas as pd
import json

# Load environment variables from .env if present
load_dotenv()

LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN")

try:
    from label_studio_sdk.client import LabelStudio
except ImportError:
    LabelStudio = None

PROJECT_ID = 1
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def get_labelstudio_config(project_id):
    if not LABEL_STUDIO_HOST or not LABEL_STUDIO_TOKEN:
        pytest.skip("LABEL_STUDIO_HOST and LABEL_STUDIO_TOKEN must be set in environment or .env file.")
    if LabelStudio is None:
        pytest.skip("label_studio_sdk is not installed.")
    client = LabelStudio(base_url=LABEL_STUDIO_HOST, api_key=LABEL_STUDIO_TOKEN)
    project = client.projects.get(id=project_id)
    return project.label_config


def parse_labelstudio_config(xml_str):
    root = ET.fromstring(xml_str)
    categories = {}
    for choices in root.findall(".//Choices"):
        name = choices.attrib["name"]
        choice_type = choices.attrib.get("choice", "single")
        categories[name] = choice_type  # 'single' or 'multiple'
    return categories


def get_all_targets(config_xml):
    categories = parse_labelstudio_config(config_xml)
    targets = set(categories.keys())  # category names
    # Parse all label values from each Choices block
    import xml.etree.ElementTree as ET

    root = ET.fromstring(config_xml)
    for choices in root.findall(".//Choices"):
        for choice in choices.findall("Choice"):
            val = choice.attrib.get("value")
            if val:
                targets.add(val)
    return sorted(targets)


def get_representative_targets(config_xml):
    categories = parse_labelstudio_config(config_xml)
    single_label_cat = next((k for k, v in categories.items() if v == "single"), None)
    multi_label_cat = next((k for k, v in categories.items() if v == "multiple"), None)
    # Find a label from each
    root = ET.fromstring(config_xml)
    single_label_value = None
    multi_label_value = None
    for choices in root.findall(".//Choices"):
        if choices.attrib.get("name") == single_label_cat and single_label_value is None:
            first = choices.find("Choice")
            if first is not None:
                single_label_value = first.attrib.get("value")
        if choices.attrib.get("name") == multi_label_cat and multi_label_value is None:
            first = choices.find("Choice")
            if first is not None:
                multi_label_value = first.attrib.get("value")
    # Build the list
    targets = []
    if single_label_cat:
        targets.append(("single_label_category", single_label_cat))
    if multi_label_cat:
        targets.append(("multi_label_category", multi_label_cat))
    if single_label_value:
        targets.append(("single_label_value", single_label_value))
    if multi_label_value:
        targets.append(("multi_label_value", multi_label_value))
    return targets


def validate_output_artifacts_and_evaluation(output_dir, project_id):
    import os

    models_dir = os.path.join(output_dir, f"project_{project_id}")
    metadata_files = glob.glob(os.path.join(models_dir, "*", "models", "*", "metadata.json"), recursive=True)
    assert metadata_files, f"No model metadata.json found in {models_dir}"

    # BERT (HuggingFace) required files
    required_bert_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "training_args.bin",
        "metadata.json",
    ]
    bert_model_files = ["model.safetensors", "pytorch_model.bin"]  # Accept either

    # Naive Bayes required files
    required_nb_files = ["model.pkl", "vectorizer.pkl", "metadata.json"]
    # For NB, require EITHER label_encoder.pkl (single-label) OR label_binarizer.pkl (multi-label)

    for meta_path in metadata_files:
        model_dir = os.path.dirname(meta_path)
        # Load metadata to check if Optuna was used
        with open(meta_path, "r") as f:
            meta = json.load(f)
        optuna_enabled = False
        if "hyperparameter_optimization" in meta:
            optuna_enabled = meta["hyperparameter_optimization"].get("enabled", False)
        # Check for optuna_study.pkl if Optuna was enabled
        if optuna_enabled:
            optuna_study_path = os.path.join(model_dir, "optuna_study.pkl")
            assert os.path.isfile(optuna_study_path), f"optuna_study.pkl not found in {model_dir} (Optuna was enabled)"
        # BERT check
        if os.path.isfile(os.path.join(model_dir, "config.json")) and os.path.isfile(
            os.path.join(model_dir, "tokenizer.json")
        ):
            missing = [f for f in required_bert_files if not os.path.isfile(os.path.join(model_dir, f))]
            has_bert_model = any(os.path.isfile(os.path.join(model_dir, f)) for f in bert_model_files)
            assert not missing and has_bert_model, (
                f"BERT model missing files in {model_dir}: {missing + ([] if has_bert_model else bert_model_files)}"
            )
        # Naive Bayes check
        if os.path.isfile(os.path.join(model_dir, "model.pkl")):
            missing = [f for f in required_nb_files if not os.path.isfile(os.path.join(model_dir, f))]
            has_label_encoder = os.path.isfile(os.path.join(model_dir, "label_encoder.pkl"))
            has_label_binarizer = os.path.isfile(os.path.join(model_dir, "label_binarizer.pkl"))
            assert not missing and (has_label_encoder or has_label_binarizer), (
                f"Naive Bayes model missing files in {model_dir}: {missing + ([] if has_label_encoder or has_label_binarizer else ['label_encoder.pkl or label_binarizer.pkl'])}"
            )
            # Optionally check for optuna_study.pkl if present (not required)

    # Evaluation results (find latest experiment dir)
    if metadata_files:
        project_dir = os.path.join(output_dir, f"project_{project_id}")
        experiment_dirs = sorted([d for d in glob.glob(os.path.join(project_dir, "*/")) if os.path.isdir(d)])
        assert experiment_dirs, f"No experiment directories found in {project_dir}"
        latest_experiment = experiment_dirs[-1]
        models_root = os.path.join(latest_experiment, "models")
        glob.glob(os.path.join(models_root, "*/"))
        results_csv = os.path.join(models_root, "results.csv")
        eval_summary = os.path.join(models_root, "evaluation_summary.json")
        assert os.path.isfile(results_csv), f"results.csv not found in {models_root}"
        assert os.path.isfile(eval_summary), f"evaluation_summary.json not found in {models_root}"
        results = pd.read_csv(results_csv)
        assert not results.empty, f"results.csv is empty in {models_root}"
        assert "accuracy" in results.columns, f"results.csv missing accuracy column in {models_root}"


@pytest.mark.parametrize(
    "optuna_flag, optuna_trials, models_config",
    [("False", None, "configs/quick_test.json"), ("True", None, "configs/quick_test_optuna.json")],
    ids=["no_optuna", "with_optuna"],
)
def test_scenario1_single_label(optuna_flag, optuna_trials, models_config):
    """
    Scenario 1: SINGLE-LABEL CATEGORY (No Target Specified)
    - Label Studio config: One <Choices> with choice="single" (e.g., sentiment)
    - Preprocessing: Extracts mixed single labels (e.g., ["Positive", "Negative", "Neutral"])
    - Training: 3 binary classifiers (one for each label)
    - Use Case: Topic classification, sentiment analysis where only one label applies
    """
    config_xml = get_labelstudio_config(PROJECT_ID)
    categories = parse_labelstudio_config(config_xml)
    single_label_cats = [k for k, v in categories.items() if v == "single"]
    if not single_label_cats:
        pytest.skip("No single-label categories for scenario 1.")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "single-label_category")
        cmd = [
            "python",
            "-m",
            "mc_classifier_pipeline.model_orchestrator",
            "--project-id",
            str(PROJECT_ID),
            "--output-dir",
            output_dir,
            "--models-config",
            models_config,
            "--text-column",
            TEXT_COLUMN,
            "--label-column",
            LABEL_COLUMN,
            "--train-ratio",
            "0.7",
        ]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0, (
            f"Scenario 1 failed\nSTDOUT: {result.stdout.decode()}\nSTDERR: {result.stderr.decode()}"
        )
        validate_output_artifacts_and_evaluation(output_dir, PROJECT_ID)


@pytest.mark.parametrize(
    "optuna_flag, optuna_trials, models_config",
    [("False", None, "configs/quick_test.json"), ("True", None, "configs/quick_test_optuna.json")],
    ids=["no_optuna", "with_optuna"],
)
def test_scenario2_multiple_single_label(optuna_flag, optuna_trials, models_config):
    """
    Scenario 2: MULTIPLE SINGLE-LABEL CATEGORIES (No Target Specified)
    - Label Studio config: Multiple <Choices> with choice="single"
    - Preprocessing: Extracts all labels from both categories in single-label format
    - Training: N binary classifiers (one for each unique label across all categories)
    - Use Case: Multiple classification tasks with mutually exclusive choices
    """
    config_xml = get_labelstudio_config(PROJECT_ID)
    categories = parse_labelstudio_config(config_xml)
    single_label_cats = [k for k, v in categories.items() if v == "single"]
    if len(single_label_cats) <= 1:
        pytest.skip("Not enough single-label categories for scenario 2.")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "multiple_single-label_categories")
        cmd = [
            "python",
            "-m",
            "mc_classifier_pipeline.model_orchestrator",
            "--project-id",
            str(PROJECT_ID),
            "--output-dir",
            output_dir,
            "--models-config",
            models_config,
            "--text-column",
            TEXT_COLUMN,
            "--label-column",
            LABEL_COLUMN,
            "--train-ratio",
            "0.7",
        ]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0, (
            f"Scenario 2 failed\nSTDOUT: {result.stdout.decode()}\nSTDERR: {result.stderr.decode()}"
        )
        validate_output_artifacts_and_evaluation(output_dir, PROJECT_ID)


@pytest.mark.parametrize(
    "optuna_flag, optuna_trials, models_config",
    [("False", None, "configs/quick_test.json"), ("True", None, "configs/quick_test_optuna.json")],
    ids=["no_optuna", "with_optuna"],
)
def test_scenario3_multi_label(optuna_flag, optuna_trials, models_config):
    """
    Scenario 3: MULTI-LABEL CATEGORY (No Target Specified)
    - Label Studio config: One <Choices> with choice="multiple" (e.g., tags)
    - Preprocessing: Extracts in multi-label format (e.g., [["Opinion", "Analysis"], ["Breaking News"]])
    - Training: 1 multi-label classifier handling all tag labels simultaneously
    - Use Case: Content tagging where multiple labels can apply to one article
    """
    config_xml = get_labelstudio_config(PROJECT_ID)
    categories = parse_labelstudio_config(config_xml)
    multi_label_cats = [k for k, v in categories.items() if v == "multiple"]
    if not multi_label_cats:
        pytest.skip("No multi-label categories for scenario 3.")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "multi-label_category")
        cmd = [
            "python",
            "-m",
            "mc_classifier_pipeline.model_orchestrator",
            "--project-id",
            str(PROJECT_ID),
            "--output-dir",
            output_dir,
            "--models-config",
            models_config,
            "--text-column",
            TEXT_COLUMN,
            "--label-column",
            LABEL_COLUMN,
            "--train-ratio",
            "0.7",
        ]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0, (
            f"Scenario 3 failed\nSTDOUT: {result.stdout.decode()}\nSTDERR: {result.stderr.decode()}"
        )
        validate_output_artifacts_and_evaluation(output_dir, PROJECT_ID)


@pytest.mark.parametrize(
    "optuna_flag, optuna_trials, models_config",
    [("False", None, "configs/quick_test.json"), ("True", None, "configs/quick_test_optuna.json")],
    ids=["no_optuna", "with_optuna"],
)
def test_scenario4_multiple_multi_label(optuna_flag, optuna_trials, models_config):
    """
    Scenario 4: MULTIPLE MULTI-LABEL CATEGORIES (No Target Specified)
    - Label Studio config: Multiple <Choices> with choice="multiple"
    - Preprocessing: Extracts labels by category, preserving structure
    - Training: 2 separate multi-label classifiers (one for each category)
    - Use Case: Multiple independent multi-label tasks (e.g., content tags + emotional tone)
    """
    config_xml = get_labelstudio_config(PROJECT_ID)
    categories = parse_labelstudio_config(config_xml)
    multi_label_cats = [k for k, v in categories.items() if v == "multiple"]
    if len(multi_label_cats) <= 1:
        pytest.skip("Not enough multi-label categories for scenario 4.")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "multiple_multi-label_categories")
        cmd = [
            "python",
            "-m",
            "mc_classifier_pipeline.model_orchestrator",
            "--project-id",
            str(PROJECT_ID),
            "--output-dir",
            output_dir,
            "--models-config",
            models_config,
            "--text-column",
            TEXT_COLUMN,
            "--label-column",
            LABEL_COLUMN,
            "--train-ratio",
            "0.7",
        ]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0, (
            f"Scenario 4 failed\nSTDOUT: {result.stdout.decode()}\nSTDERR: {result.stderr.decode()}"
        )
        validate_output_artifacts_and_evaluation(output_dir, PROJECT_ID)


@pytest.mark.parametrize(
    "optuna_flag, optuna_trials, models_config",
    [("False", None, "configs/quick_test.json"), ("True", None, "configs/quick_test_optuna.json")],
    ids=["no_optuna", "with_optuna"],
)
def test_scenario5_mixed(optuna_flag, optuna_trials, models_config):
    """
    Scenario 5: MIXED CATEGORIES (No Target Specified)
    - Label Studio config: At least one <Choices> with choice="single" and one with choice="multiple"
    - Preprocessing: Extracts everything in categorized format
    - Training: Binary classifiers for single-label categories + multi-label classifiers for multi-label categories
    - Use Case: Complex annotation projects with both exclusive and non-exclusive labeling
    """
    config_xml = get_labelstudio_config(PROJECT_ID)
    categories = parse_labelstudio_config(config_xml)
    single_label_cats = [k for k, v in categories.items() if v == "single"]
    multi_label_cats = [k for k, v in categories.items() if v == "multiple"]
    if not (single_label_cats and multi_label_cats):
        pytest.skip("No mixed categories for scenario 5.")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "mixed_categories")
        cmd = [
            "python",
            "-m",
            "mc_classifier_pipeline.model_orchestrator",
            "--project-id",
            str(PROJECT_ID),
            "--output-dir",
            output_dir,
            "--models-config",
            models_config,
            "--text-column",
            TEXT_COLUMN,
            "--label-column",
            LABEL_COLUMN,
            "--train-ratio",
            "0.7",
        ]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0, (
            f"Scenario 5 failed\nSTDOUT: {result.stdout.decode()}\nSTDERR: {result.stderr.decode()}"
        )
        validate_output_artifacts_and_evaluation(output_dir, PROJECT_ID)


config_xml = get_labelstudio_config(PROJECT_ID)
all_targets = get_all_targets(config_xml)
representative_targets = get_representative_targets(config_xml)


@pytest.mark.parametrize(
    "optuna_flag, models_config",
    [("False", "configs/quick_test.json"), ("True", "configs/quick_test_optuna.json")],
    ids=["no_optuna", "with_optuna"],
)
@pytest.mark.parametrize("target_type,target_label", representative_targets, ids=lambda x: x[0])
def test_scenario6_targeted(target_type, target_label, optuna_flag, models_config):
    """
    Scenario 6: TARGETED CLASSIFICATION (Target Specified)
    Tests one representative of each type of target label:
    - single_label_category: a single-label category name
    - multi_label_category: a multi-label category name
    - single_label_value: a label from a single-label category
    - multi_label_value: a label from a multi-label category
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, f"targeted_{target_type}_{target_label}")
        cmd = [
            "python",
            "-m",
            "mc_classifier_pipeline.model_orchestrator",
            "--project-id",
            str(PROJECT_ID),
            "--output-dir",
            output_dir,
            "--models-config",
            models_config,
            "--text-column",
            TEXT_COLUMN,
            "--label-column",
            LABEL_COLUMN,
            "--target-label",
            target_label,
            "--train-ratio",
            "0.7",
        ]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0, (
            f"Scenario 6 failed for target {target_type}={target_label}\nSTDOUT: {result.stdout.decode()}\nSTDERR: {result.stderr.decode()}"
        )
        validate_output_artifacts_and_evaluation(output_dir, PROJECT_ID)
