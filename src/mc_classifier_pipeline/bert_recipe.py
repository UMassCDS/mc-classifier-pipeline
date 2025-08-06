import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List
import ast

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import joblib

# Disable MLflow tracking completely
os.environ["MLFLOW_TRACKING_DISABLED"] = "True"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"

from mc_classifier_pipeline.utils import configure_logging

# Set up logging
configure_logging()
logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    """Custom dataset for text classification supporting both single-label and multi-label"""

    def __init__(self, texts, labels, tokenizer, max_length=512, is_multi_label=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_multi_label = is_multi_label

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        if self.is_multi_label:
            # Multi-label: return as float tensor for BCEWithLogitsLoss
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.float),
            }
        else:
            # Single-label: return as long tensor for CrossEntropyLoss
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.long),
            }


class BERTTextClassifier:
    """BERT-based text classifier with training and inference capabilities"""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.label_binarizer = None
        self.training_args = None
        self.metadata = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_data(
        self, project_folder: str, text_column: str = "text", label_column: str = "label"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data from CSV files"""
        train_path = os.path.join(project_folder, "train.csv")
        test_path = os.path.join(project_folder, "test.csv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")

        # Validate required columns
        if text_column not in train_df.columns or label_column not in train_df.columns:
            raise ValueError(f"Required columns '{text_column}' and '{label_column}' not found in training data")

        return train_df, test_df

    def parse_labels(self, label_data, target_label=None, target_labels=None):
        """Parse label data from different formats"""
        if isinstance(label_data, str):
            try:
                # Try to parse as categorized format: {"sentiment": ["Positive"], "tags": ["Opinion"]}
                parsed = ast.literal_eval(label_data)
                if isinstance(parsed, dict):
                    # Categorized format
                    return parsed
                else:
                    # List format: ["Positive", "Opinion"] or single string
                    return parsed if isinstance(parsed, list) else [parsed]
            except (ValueError, SyntaxError):
                # Simple string label
                return [label_data]
        elif isinstance(label_data, list):
            return label_data
        else:
            return [label_data]

    def prepare_binary_labels(self, df, target_label, text_column="text", label_column="label"):
        """Prepare data for binary classification"""
        prepared_data = []

        for _, row in df.iterrows():
            text = row[text_column]
            label_data = self.parse_labels(row[label_column])

            # Check if target label is present
            label_present = False
            if isinstance(label_data, dict):
                # Categorized format
                for category_labels in label_data.values():
                    if isinstance(category_labels, list):
                        if target_label in category_labels:
                            label_present = True
                            break
                    else:
                        if target_label == category_labels:
                            label_present = True
                            break
            else:
                # List or single format
                label_list = label_data if isinstance(label_data, list) else [label_data]
                label_present = target_label in label_list

            prepared_data.append({text_column: text, label_column: 1 if label_present else 0})

        return pd.DataFrame(prepared_data)

    def prepare_multilabel_labels(self, df, target_labels, text_column="text", label_column="label"):
        """Prepare data for multi-label classification"""
        prepared_data = []

        for _, row in df.iterrows():
            text = row[text_column]
            label_data = self.parse_labels(row[label_column])

            # Extract relevant labels for this multi-label task
            relevant_labels = []
            if isinstance(label_data, dict):
                # Categorized format - flatten all labels
                for category_labels in label_data.values():
                    if isinstance(category_labels, list):
                        relevant_labels.extend(category_labels)
                    else:
                        relevant_labels.append(category_labels)
            else:
                # List or single format
                relevant_labels = label_data if isinstance(label_data, list) else [label_data]

            # Filter to only target labels
            present_labels = [label for label in relevant_labels if label in target_labels]

            prepared_data.append({text_column: text, label_column: present_labels})

        return pd.DataFrame(prepared_data)

    def prepare_model(self, num_labels: int, is_multi_label: bool = False):
        """Initialize tokenizer and model"""
        logger.info(f"Loading tokenizer and model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if is_multi_label:
            problem_type = "multi_label_classification"
        else:
            problem_type = "single_label_classification"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels, problem_type=problem_type
        )

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)

    def prepare_datasets(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
        max_length: int = 512,
        is_multi_label: bool = False,
        target_labels: Optional[List[str]] = None,
    ):
        """Prepare training and test datasets"""

        if is_multi_label:
            # Multi-label classification
            if target_labels is None:
                raise ValueError("target_labels must be provided for multi-label classification")

            self.label_binarizer = MultiLabelBinarizer(classes=target_labels)

            train_labels_binary = self.label_binarizer.fit_transform(train_df[label_column].tolist())
            test_labels_binary = self.label_binarizer.transform(test_df[label_column].tolist())

            logger.info(f"Multi-label classes: {self.label_binarizer.classes_.tolist()}")

        else:
            # Binary or single-label classification
            self.label_encoder = LabelEncoder()
            all_labels = pd.concat([train_df[label_column], test_df[label_column]]).unique()
            self.label_encoder.fit(all_labels)

            train_labels_binary = self.label_encoder.transform(train_df[label_column])
            test_labels_binary = self.label_encoder.transform(test_df[label_column])

            logger.info(f"Label classes: {self.label_encoder.classes_.tolist()}")

        # Create datasets
        train_dataset = TextClassificationDataset(
            train_df[text_column].tolist(), train_labels_binary, self.tokenizer, max_length, is_multi_label
        )

        test_dataset = TextClassificationDataset(
            test_df[text_column].tolist(), test_labels_binary, self.tokenizer, max_length, is_multi_label
        )

        return train_dataset, test_dataset

    def compute_metrics_single_label(self, eval_pred):
        """Compute metrics for single-label classification"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        accuracy = accuracy_score(labels, predictions)

        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

    def compute_metrics_multi_label(self, eval_pred):
        """Compute metrics for multi-label classification"""
        predictions, labels = eval_pred
        # Convert logits to probabilities and then to binary predictions
        predictions = (torch.sigmoid(torch.from_numpy(predictions)) > 0.5).numpy()

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )

        # Subset accuracy (exact match ratio)
        subset_accuracy = accuracy_score(labels, predictions)

        return {"subset_accuracy": subset_accuracy, "f1": f1, "precision": precision, "recall": recall}

    def train(
        self,
        project_folder: str,
        save_path: str,
        text_column: str = "text",
        label_column: str = "label",
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        """Train the BERT model"""

        # Default hyperparameters
        default_hyperparams = {
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "num_train_epochs": 1,
            "weight_decay": 0.01,
            "warmup_steps": 0,
            "max_length": 512,
            "save_strategy": "epoch",
            "eval_strategy": "epoch",
            "logging_strategy": "steps",
            "logging_steps": 10,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "save_total_limit": 2,
            # New parameters for task type
            "is_multi_label": False,
            "target_label": None,
            "target_labels": None,
        }

        if hyperparams:
            default_hyperparams.update(hyperparams)

        # Extract task parameters
        is_multi_label = default_hyperparams.get("is_multi_label", False)
        target_label = default_hyperparams.get("target_label")
        target_labels = default_hyperparams.get("target_labels")

        # Load data
        train_df, test_df = self.load_data(project_folder, text_column, label_column)

        # Prepare data based on task type
        if is_multi_label:
            if target_labels is None:
                raise ValueError("target_labels must be provided for multi-label classification")

            train_df_processed = self.prepare_multilabel_labels(train_df, target_labels, text_column, label_column)
            test_df_processed = self.prepare_multilabel_labels(test_df, target_labels, text_column, label_column)
            num_labels = len(target_labels)

        else:
            if target_label is not None:
                # Binary classification
                train_df_processed = self.prepare_binary_labels(train_df, target_label, text_column, label_column)
                test_df_processed = self.prepare_binary_labels(test_df, target_label, text_column, label_column)
                num_labels = 2  # Binary: 0 or 1
            else:
                # Multi-class classification (use original data)
                train_df_processed = train_df
                test_df_processed = test_df
                num_labels = len(pd.concat([train_df[label_column], test_df[label_column]]).unique())

        logger.info(f"Task type: {'Multi-label' if is_multi_label else 'Binary' if target_label else 'Multi-class'}")
        logger.info(f"Number of labels: {num_labels}")

        # Prepare model and datasets
        self.prepare_model(num_labels, is_multi_label)
        train_dataset, test_dataset = self.prepare_datasets(
            train_df_processed,
            test_df_processed,
            text_column,
            label_column,
            default_hyperparams["max_length"],
            is_multi_label,
            target_labels,
        )

        # Set up training arguments
        self.training_args = TrainingArguments(
            output_dir=save_path,
            learning_rate=default_hyperparams["learning_rate"],
            per_device_train_batch_size=default_hyperparams["per_device_train_batch_size"],
            per_device_eval_batch_size=default_hyperparams["per_device_eval_batch_size"],
            num_train_epochs=default_hyperparams["num_train_epochs"],
            weight_decay=default_hyperparams["weight_decay"],
            warmup_steps=default_hyperparams["warmup_steps"],
            save_strategy=default_hyperparams["save_strategy"],
            eval_strategy=default_hyperparams["eval_strategy"],
            logging_strategy=default_hyperparams["logging_strategy"],
            logging_steps=default_hyperparams["logging_steps"],
            load_best_model_at_end=default_hyperparams["load_best_model_at_end"],
            metric_for_best_model=default_hyperparams["metric_for_best_model"],
            greater_is_better=default_hyperparams["greater_is_better"],
            save_total_limit=default_hyperparams["save_total_limit"],
            report_to=[],
            disable_tqdm=False,
        )

        # Choose metrics function based on task type
        compute_metrics_fn = self.compute_metrics_multi_label if is_multi_label else self.compute_metrics_single_label

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics_fn,
        )

        # Start training
        logger.info("Starting training...")
        train_result = trainer.train()

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        eval_result = trainer.evaluate()

        # Save the model and tokenizer
        logger.info(f"Saving model to {save_path}")
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save encoders
        if is_multi_label:
            joblib.dump(self.label_binarizer, os.path.join(save_path, "label_binarizer.pkl"))
        else:
            joblib.dump(self.label_encoder, os.path.join(save_path, "label_encoder.pkl"))

        # Create metadata
        self.metadata = {
            "framework": "transformers",
            "model_name": self.model_name,
            "num_labels": num_labels,
            "is_multi_label": is_multi_label,
            "target_label": target_label,
            "target_labels": target_labels,
            "label_classes": target_labels
            if is_multi_label
            else (self.label_encoder.classes_.tolist() if self.label_encoder else []),
            "training_samples": len(train_df_processed),
            "test_samples": len(test_df_processed),
            "hyperparameters": default_hyperparams,
            "training_time": datetime.now().isoformat(),
            "final_eval_results": eval_result,
            "train_results": {
                "training_loss": getattr(train_result, "training_loss", {}),
                "train_runtime": getattr(train_result, "metrics", {}).get("train_runtime", None),
                "train_samples_per_second": getattr(train_result, "metrics", {}).get("train_samples_per_second", None),
                "train_steps_per_second": getattr(train_result, "metrics", {}).get("train_steps_per_second", None),
                "total_flos": getattr(train_result, "metrics", {}).get("total_flos", None),
                "epoch": getattr(train_result, "metrics", {}).get("epoch", None),
            },
            "text_column": text_column,
            "label_column": label_column,
        }

        # Save metadata
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info("Training completed successfully!")
        logger.info(f"Final evaluation results: {eval_result}")

        return self.metadata

    @classmethod
    def load_for_inference(cls, model_path: str):
        """Load a trained model for inference"""

        # Load metadata
        metadata_path = os.path.join(model_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Initialize classifier
        classifier = cls(metadata["model_name"])
        classifier.metadata = metadata

        # Load model and tokenizer
        classifier.tokenizer = AutoTokenizer.from_pretrained(model_path)
        classifier.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Load appropriate encoder
        is_multi_label = metadata.get("is_multi_label", False)
        if is_multi_label:
            binarizer_path = os.path.join(model_path, "label_binarizer.pkl")
            classifier.label_binarizer = joblib.load(binarizer_path)
        else:
            encoder_path = os.path.join(model_path, "label_encoder.pkl")
            classifier.label_encoder = joblib.load(encoder_path)

        logger.info(f"Model loaded successfully from {model_path}")

        return classifier

    def predict(self, texts, batch_size: int = 32, return_probabilities: bool = False):
        """Make predictions on new text data"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Use load_for_inference() first.")

        is_multi_label = self.metadata.get("is_multi_label", False)
        self.model.eval()
        predictions = []
        probabilities = []

        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.metadata.get("hyperparameters", {}).get("max_length", 512),
                return_tensors="pt",
            )

            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            # Make predictions
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits

                if is_multi_label:
                    # Multi-label: use sigmoid and threshold
                    probs = torch.sigmoid(logits)
                    batch_predictions = (probs > 0.5).cpu().numpy()
                    probabilities.extend(probs.cpu().numpy())
                else:
                    # Single-label: use softmax and argmax
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    probabilities.extend(probs.cpu().numpy())
                    batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()

                predictions.extend(batch_predictions)

        # Convert predictions back to original labels
        if is_multi_label:
            # Convert list of 2D arrays to single 2D numpy array
            predictions = np.vstack(predictions) if predictions else np.array([])
            predicted_labels = self.label_binarizer.inverse_transform(predictions)
        else:
            # For single-label, predictions is already a list of integers
            predicted_labels = self.label_encoder.inverse_transform(predictions)

        if return_probabilities:
            return predicted_labels, np.array(probabilities)
        else:
            return predicted_labels

    def get_model_info(self):
        """Get model information"""
        return self.metadata
