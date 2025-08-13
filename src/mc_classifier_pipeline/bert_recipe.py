import gc
import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
import tempfile

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

from mc_classifier_pipeline.utils import configure_logging

# Set up logging
configure_logging()
logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    """
    Custom PyTorch Dataset for text classification tasks.

    Args:
        texts (List[str]): List of input text samples.
        labels (List[int]): List of encoded label values.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding text.
        max_length (int, optional): Maximum sequence length for tokenization. Defaults to 512.
    """

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset and tokenize it.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing input_ids, attention_mask, and label tensor.
        """
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
    """
    BERT-based text classifier supporting training, hyperparameter optimization, and inference.

    Args:
        model_name (str): Name or path of the pretrained BERT model.
        use_optuna (bool, optional): Whether to use Optuna for hyperparameter optimization. Defaults to False.
    """

    def __init__(self, model_name: str = "bert-base-uncased", use_optuna: bool = False):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.training_args = None
        self.metadata = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.use_optuna = use_optuna
        self.best_trial = None

        # Store data for optimization
        self.train_df = None
        self.test_df = None
        self.text_column = None
        self.label_column = None
        self.study = None

    def load_data(
        self, project_folder: str, text_column: str = "text", label_column: str = "label"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data from CSV files in the specified project folder.

        Args:
            project_folder (str): Path to the folder containing train.csv and test.csv.
            text_column (str, optional): Name of the text column. Defaults to "text".
            label_column (str, optional): Name of the label column. Defaults to "label".

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test DataFrames.
        """
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
        """
        Initialize the tokenizer and BERT model for sequence classification.

        Args:
            num_labels (int): Number of unique label classes.
        """
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
        """
        Prepare PyTorch datasets for training and testing.

        Args:
            train_df (pd.DataFrame): Training data.
            test_df (pd.DataFrame): Test data.
            text_column (str, optional): Name of the text column. Defaults to "text".
            label_column (str, optional): Name of the label column. Defaults to "label".
            max_length (int, optional): Maximum sequence length. Defaults to 512.

        Returns:
            Tuple[TextClassificationDataset, TextClassificationDataset]: Training and test datasets.
        """

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
        """
        Compute evaluation metrics (accuracy, precision, recall, F1) for predictions.

        Args:
            eval_pred (Tuple[np.ndarray, np.ndarray]): Tuple of predictions and true labels.

        Returns:
            dict: Dictionary of metric names and values.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        accuracy = accuracy_score(labels, predictions)

        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

    def _objective(self, trial):
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.

        Returns:
            float: Evaluation F1 score for the trial.
        """

        # Suggest hyperparameters
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "num_epochs": trial.suggest_int("num_epochs", 1, 4),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "max_length": trial.suggest_categorical("max_length", [256, 512]),
        }

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                num_labels = len(
                    pd.concat([self.train_df[self.label_column], self.test_df[self.label_column]]).unique()
                )

                self.prepare_model(num_labels)
                train_dataset, test_dataset = self.prepare_datasets(
                    self.train_df, self.test_df, self.text_column, self.label_column, params["max_length"]
                )

                total_steps = (len(self.train_df) // params["batch_size"]) * params["num_epochs"]
                warmup_steps = int(total_steps * params["warmup_ratio"])

                training_args = TrainingArguments(
                    output_dir=temp_dir,
                    learning_rate=params["learning_rate"],
                    per_device_train_batch_size=params["batch_size"],
                    per_device_eval_batch_size=params["batch_size"],
                    num_train_epochs=params["num_epochs"],
                    weight_decay=params["weight_decay"],
                    warmup_steps=warmup_steps,
                    save_strategy="epoch",
                    eval_strategy="epoch",
                    logging_strategy="no",
                    load_best_model_at_end=True,
                    metric_for_best_model="f1",
                    greater_is_better=True,
                    save_total_limit=1,
                    report_to=[],
                    disable_tqdm=False,
                )

                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    tokenizer=self.tokenizer,
                    data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
                    compute_metrics=self.compute_metrics,
                )

                trainer.train()
                eval_result = trainer.evaluate()

                del trainer
                self.model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return eval_result.get("eval_f1", 0.0)

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return 0.0

    def optimize_hyperparameters(
        self,
        project_folder: str,
        text_column: str = "text",
        label_column: str = "label",
        n_trials: int = 5,
        timeout: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        """
        Run Optuna hyperparameter optimization for the BERT model.

        Args:
            project_folder (str): Path to project folder with data.
            text_column (str, optional): Name of the text column. Defaults to "text".
            label_column (str, optional): Name of the label column. Defaults to "label".
            n_trials (int, optional): Number of Optuna trials. Defaults to 5.
            timeout (Optional[int], optional): Timeout in seconds. Defaults to None.
            save_path (Optional[str], optional): Path to save Optuna study. Defaults to None.

        Returns:
            optuna.Study: The completed Optuna study object.
        """
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from optuna.samplers import TPESampler
        except ImportError:
            raise ImportError("Optuna required: pip install optuna")

        if self.train_df is None:
            self.train_df, self.test_df = self.load_data(project_folder, text_column, label_column)
            self.text_column, self.label_column = text_column, label_column

        study_path = os.path.join(save_path if save_path else project_folder, "optuna_study.pkl")
        if os.path.exists(study_path):
            study = joblib.load(study_path)
            logger.info(f"Resuming existing study with {len(study.trials)} trials")
        else:
            study = optuna.create_study(
                direction="maximize", sampler=TPESampler(), pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
            )

        logger.info(f"Starting optimization with {n_trials} trials...")

        def save_callback(study, trial):
            save_dir = save_path if save_path else project_folder
            joblib.dump(study, os.path.join(save_dir, "optuna_study.pkl"))

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            gc_after_trial=True,
            callbacks=[save_callback],
        )

        self.best_trial = study.best_trial
        self.study = study

        logger.info(f"Best F1: {study.best_value:.4f}, Best params: {study.best_params}")
        return study

    def train(
        self,
        project_folder: str,
        save_path: str,
        text_column: str = "text",
        label_column: str = "label",
        hyperparams: Optional[Dict[str, Any]] = None,
        optimize_hyperparams: Optional[bool] = None,
        n_trials: int = 5,
        timeout: Optional[int] = None,
    ):
        """
        Train the BERT model, optionally using Optuna for hyperparameter optimization.

        Args:
            project_folder (str): Path to project folder with data.
            save_path (str): Path to save trained model and artifacts.
            text_column (str, optional): Name of the text column. Defaults to "text".
            label_column (str, optional): Name of the label column. Defaults to "label".
            hyperparams (Optional[Dict[str, Any]], optional): Hyperparameters for training. Defaults to None.
            optimize_hyperparams (Optional[bool], optional): Whether to optimize hyperparameters. Defaults to None.
            n_trials (int, optional): Number of Optuna trials. Defaults to 5.
            timeout (Optional[int], optional): Timeout for optimization. Defaults to None.

        Returns:
            dict: Metadata about the trained model and training process.
        """

        # Determine if we should optimize
        should_optimize = optimize_hyperparams if optimize_hyperparams is not None else self.use_optuna

        if should_optimize:
            logger.info("Using Optuna optimization...")
            study = self.optimize_hyperparameters(
                project_folder, text_column, label_column, n_trials, timeout, save_path
            )

            # Convert best params and train
            best_params = study.best_params
            hyperparams = {
                "learning_rate": best_params["learning_rate"],
                "per_device_train_batch_size": best_params["batch_size"],
                "per_device_eval_batch_size": best_params["batch_size"],
                "num_train_epochs": best_params["num_epochs"],
                "weight_decay": best_params["weight_decay"],
                "max_length": best_params["max_length"],
                "warmup_steps": int(
                    (len(self.train_df) // best_params["batch_size"])
                    * best_params["num_epochs"]
                    * best_params["warmup_ratio"]
                ),
            }

            # Train with optimized params
            metadata = self._train_standard(save_path, hyperparams, use_stored_data=True)

            # Add optimization info
            metadata["optuna_optimization"] = {
                "best_f1_score": study.best_value,
                "best_parameters": best_params,
                "optimization_datetime": datetime.now().isoformat(),
            }

            with open(os.path.join(save_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            return metadata
        else:
            return self._train_standard(project_folder, save_path, text_column, label_column, hyperparams)

    def _train_standard(
        self,
        project_folder_or_save_path: str,
        save_path_or_hyperparams=None,
        text_column: str = "text",
        label_column: str = "label",
        hyperparams: Optional[Dict[str, Any]] = None,
        use_stored_data: bool = False,
    ):
        """
        Standard training implementation for BERT model.

        Args:
            project_folder_or_save_path (str): Project folder or save path depending on call pattern.
            save_path_or_hyperparams: Save path or hyperparameters depending on call pattern.
            text_column (str, optional): Name of the text column. Defaults to "text".
            label_column (str, optional): Name of the label column. Defaults to "label".
            hyperparams (Optional[Dict[str, Any]], optional): Training hyperparameters. Defaults to None.
            use_stored_data (bool, optional): Whether to use stored data. Defaults to False.

        Returns:
            dict: Metadata about the trained model and training process.
        """

        # Handle different call patterns
        if use_stored_data:
            save_path = project_folder_or_save_path
            hyperparams = save_path_or_hyperparams
            train_df, test_df = self.train_df, self.test_df
        else:
            project_folder = project_folder_or_save_path
            save_path = save_path_or_hyperparams
            train_df, test_df = self.load_data(project_folder, text_column, label_column)

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
        """
        Load a trained BERT model and associated artifacts for inference.

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            BERTTextClassifier: Loaded classifier instance ready for inference.
        """

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
        classifier.model.to(classifier.device)

        # Load label encoder
        label_encoder_path = os.path.join(model_path, "label_encoder.pkl")
        classifier.label_encoder = joblib.load(label_encoder_path)

        # Load Optuna study if exists
        study_path = os.path.join(model_path, "optuna_study.pkl")
        if os.path.exists(study_path):
            classifier.study = joblib.load(study_path)
            logger.info("Optuna study loaded successfully")
        else:
            classifier.study = None

        logger.info(f"Model loaded successfully from {model_path}")

        return classifier

    def predict(self, texts, batch_size: int = 32, return_probabilities: bool = False):
        """
        Make predictions on new text data using the trained model.

        Args:
            texts (List[str]): List of input text samples.
            batch_size (int, optional): Batch size for prediction. Defaults to 32.
            return_probabilities (bool, optional): Whether to return class probabilities. Defaults to False.

        Returns:
            np.ndarray or Tuple[np.ndarray, np.ndarray]: Predicted labels, optionally with probabilities.
        """
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
        """
        Retrieve metadata information about the trained model.

        Returns:
            dict: Model metadata.
        """
        return self.metadata

    def get_optimization_history(self):
        """
        Get Optuna optimization history if available.

        Returns:
            dict or None: Dictionary containing optimization history, or None if not available.
        """
        if not hasattr(self, "study") or self.study is None:
            return None
        trials_df = self.study.trials_dataframe()
        return {
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "n_trials": len(self.study.trials),
            "trials_dataframe": trials_df,
            "optimization_history": [
                {"trial": i, "value": trial.value, "params": trial.params}
                for i, trial in enumerate(self.study.trials)
                if trial.value is not None
            ],
        }


# if __name__ == "__main__":
    # # Standard training
    # classifier = BERTTextClassifier(model_name="distilbert/distilbert-base-uncased")
    # metadata = classifier.train(
    #     project_folder="data",
    #     save_path="models/distilbert-base-uncased",
    #     text_column="text",
    #     label_column="label",
    # )
    # print("Standard training completed!")

    # # Training with optimization
    # classifier_opt = BERTTextClassifier(model_name="distilbert/distilbert-base-uncased", use_optuna=True)
    # metadata_opt = classifier_opt.train(
    #     project_folder="data",
    #     save_path="models/optimized-distilbert",
    #     text_column="text",
    #     label_column="label",
    #     n_trials=1,
    # )
    # print("Optimized training completed!")

    # history = classifier_opt.get_optimization_history()
    # print(history)

    # # Inference
    # classifier = BERTTextClassifier.load_for_inference(model_path="models/optimized-distilbert")
    # history = classifier.get_optimization_history()
    # print(history)
    # print("Model loaded for inference!")
    # predictions = classifier.predict(
    #     texts=["That superman movie was so bad. I hated it. I would never watch it again."], return_probabilities=True
    # )
    # print(predictions)
