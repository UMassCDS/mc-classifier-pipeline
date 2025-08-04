import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

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
from sklearn.preprocessing import LabelEncoder
import joblib

# Disable MLflow tracking completely
os.environ["MLFLOW_TRACKING_DISABLED"] = "True"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"

# from . import utils
from utils import configure_logging  # this is for local running, the above is for running in the pipeline

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


class BERTTextClassifier:
    """BERT-based text classifier with training and inference capabilities"""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
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

        # Encode labels
        all_labels = pd.concat([train_df[label_column], test_df[label_column]]).unique()
        self.label_encoder.fit(all_labels)

        train_labels = self.label_encoder.transform(train_df[label_column])
        test_labels = self.label_encoder.transform(test_df[label_column])

        # Create datasets
        train_dataset = TextClassificationDataset(
            train_df[text_column].tolist(), train_labels, self.tokenizer, max_length
        )

        test_dataset = TextClassificationDataset(
            test_df[text_column].tolist(), test_labels, self.tokenizer, max_length
        )

        logger.info(f"Number of unique labels: {len(self.label_encoder.classes_)}")
        logger.info(
            f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}"
        )

        return train_dataset, test_dataset

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        accuracy = accuracy_score(labels, predictions)

        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

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

        # Save the model and tokenizer
        logger.info(f"Saving model to {save_path}")
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save label encoder
        joblib.dump(self.label_encoder, os.path.join(save_path, "label_encoder.pkl"))

        # Create metadata
        self.metadata = {
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

        # Load label encoder
        label_encoder_path = os.path.join(model_path, "label_encoder.pkl")
        classifier.label_encoder = joblib.load(label_encoder_path)

        logger.info(f"Model loaded successfully from {model_path}")

        return classifier

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
        return self.metadata


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
