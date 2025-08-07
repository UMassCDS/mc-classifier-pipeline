import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

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

# Disable MLflow tracking completely
os.environ["MLFLOW_TRACKING_DISABLED"] = "True"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"

from mc_classifier_pipeline.base_classifier import BaseTextClassifier
from mc_classifier_pipeline.utils import configure_logging

# Set up logging
configure_logging()
logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    """Custom dataset for text classification"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class BERTTextClassifier(BaseTextClassifier):
    """BERT-based text classifier with training and inference capabilities"""

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__(model_name)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.training_args = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def prepare_model(self, num_labels: int):
        """Initialize tokenizer and model"""
        logger.info(f"Loading tokenizer and model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels, problem_type="single_label_classification"
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
    ):
        """Prepare training and test datasets"""

        # Prepare labels using base class method
        train_labels, test_labels = self.prepare_labels(train_df, test_df, label_column)

        # Create datasets
        train_dataset = TextClassificationDataset(
            train_df[text_column].tolist(), train_labels, self.tokenizer, max_length
        )

        test_dataset = TextClassificationDataset(
            test_df[text_column].tolist(), test_labels, self.tokenizer, max_length
        )

        return train_dataset, test_dataset

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.compute_metrics_from_predictions(labels, predictions)

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
        }

        if hyperparams:
            default_hyperparams.update(hyperparams)

        # Load and prepare data
        train_df, test_df = self.load_data(project_folder, text_column, label_column)

        # Get number of unique labels
        num_labels = len(pd.concat([train_df[label_column], test_df[label_column]]).unique())

        # Prepare model and datasets
        self.prepare_model(num_labels)
        train_dataset, test_dataset = self.prepare_datasets(
            train_df, test_df, text_column, label_column, default_hyperparams["max_length"]
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

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics,
        )

        # Start training
        logger.info("Starting training...")
        train_result = trainer.train()

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        eval_result = trainer.evaluate()

        # Save the model using base class method
        logger.info(f"Saving model to {save_path}")
        trainer.save_model(save_path)

        # Create metadata
        metadata = {
            "framework": "transformers",
            "model_name": self.model_name,
            "num_labels": num_labels,
            "label_classes": self.label_encoder.classes_.tolist(),
            "training_samples": len(train_df),
            "test_samples": len(test_df),
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

        self.save_model(save_path, metadata)
        self.is_trained = True

        logger.info("Training completed successfully!")
        logger.info(f"Final evaluation results: {eval_result}")

        return self.metadata

    @classmethod
    def _create_instance_from_metadata(cls, metadata: Dict[str, Any]):
        """Create an instance from metadata."""
        model_name = metadata.get("model_name", "bert-base-uncased")
        return cls(model_name)

    def _load_model_components(self, model_dir: Path) -> None:
        """Load model-specific components."""
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

    def _save_model_components(self, save_dir: Path) -> None:
        """Save model-specific components."""
        self.tokenizer.save_pretrained(save_dir)
        self.model.save_pretrained(save_dir)

    def get_framework_name(self) -> str:
        """Get the framework name."""
        return "hf"

    def predict(self, texts, batch_size: int = 32, return_probabilities: bool = False):
        """Make predictions on new text data"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Use load_for_inference() first.")

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

                # Get probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probabilities.extend(probs.cpu().numpy())

                # Get predictions
                batch_predictions = torch.argmax(logits, dim=-1)
                predictions.extend(batch_predictions.cpu().numpy())

        # Convert predictions back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)

        if return_probabilities:
            return predicted_labels, np.array(probabilities)
        else:
            return predicted_labels

    def get_model_info(self):
        """Get model information"""
        return super().get_model_info()


if __name__ == "__main__":
    classifier = BERTTextClassifier(model_name="distilbert/distilbert-base-uncased")
    metadata = classifier.train(
        project_folder="data",
        save_path="models/distilbert-base-uncased",
        text_column="text",
        label_column="label",
    )
    print("Metadata: ", metadata)

    classifier = BERTTextClassifier.load_for_inference(model_path="models/distilbert-base-uncased")
    predictions = classifier.predict(
        texts=["That superman movie was so bad. I hated it. I would never watch it again."], return_probabilities=True
    )
    print(predictions)  # (array(['negative'], dtype=object), array([[0.7060923, 0.2939077]], dtype=float32))

    label = classifier.predict(
        texts=["That superman movie was so bad. I hated it. I would never watch it again."], return_probabilities=False
    )
    print(label)  # ['negative']
