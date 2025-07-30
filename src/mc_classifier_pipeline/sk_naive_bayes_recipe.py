import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import joblib

# from . import utils
from utils import configure_logging  # for local running

# Set up logging
configure_logging()
logger = logging.getLogger(__name__)


class SKNaiveBayesTextClassifier:
    """Naive Bayes text classifier with training and inference capabilities (scikit-learn)"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()
        self.label_encoder = LabelEncoder()
        self.metadata = {}

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

    def prepare_datasets(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
    ):
        """Prepare training and test datasets"""
        # Encode labels
        all_labels = pd.concat([train_df[label_column], test_df[label_column]]).unique()
        self.label_encoder.fit(all_labels)

        train_labels = self.label_encoder.transform(train_df[label_column])
        test_labels = self.label_encoder.transform(test_df[label_column])

        train_texts = train_df[text_column].tolist()
        test_texts = test_df[text_column].tolist()

        logger.info(f"Number of unique labels: {len(self.label_encoder.classes_)}")
        logger.info(
            f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}"
        )

        return (train_texts, train_labels), (test_texts, test_labels)

    def compute_metrics(self, y_true, y_pred):
        """Compute metrics for evaluation"""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        accuracy = accuracy_score(y_true, y_pred)
        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

    def train(
        self,
        project_folder: str,
        save_path: str,
        text_column: str = "text",
        label_column: str = "label",
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        """Train the Naive Bayes model"""
        # Default hyperparameters
        default_hyperparams = {
            "ngram_range": (1, 1),
            "min_df": 1,
            "max_df": 1.0,
            "alpha": 1.0,
        }
        if hyperparams:
            default_hyperparams.update(hyperparams)

        # Load and prepare data
        train_df, test_df = self.load_data(project_folder, text_column, label_column)
        (train_texts, train_labels), (test_texts, test_labels) = self.prepare_datasets(
            train_df, test_df, text_column, label_column
        )

        # Prepare vectorizer and model
        self.vectorizer = TfidfVectorizer(
            ngram_range=default_hyperparams["ngram_range"],
            min_df=default_hyperparams["min_df"],
            max_df=default_hyperparams["max_df"],
        )
        X_train = self.vectorizer.fit_transform(train_texts)
        X_test = self.vectorizer.transform(test_texts)

        self.model = MultinomialNB(alpha=default_hyperparams["alpha"])

        # Train
        logger.info("Starting training...")
        self.model.fit(X_train, train_labels)

        # Evaluate
        logger.info("Evaluating on test set...")
        y_pred = self.model.predict(X_test)
        eval_result = self.compute_metrics(test_labels, y_pred)

        # Save model, vectorizer, and label encoder
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(self.model, os.path.join(save_path, "model.pkl"))
        joblib.dump(self.vectorizer, os.path.join(save_path, "vectorizer.pkl"))
        joblib.dump(self.label_encoder, os.path.join(save_path, "label_encoder.pkl"))

        # Create metadata
        self.metadata = {
            "framework": "naive-bayes",
            "model_type": "sklearn-naive-bayes",
            "num_labels": len(self.label_encoder.classes_),
            "label_classes": self.label_encoder.classes_.tolist(),
            "training_samples": len(train_df),
            "test_samples": len(test_df),
            "hyperparameters": default_hyperparams,
            "training_time": datetime.now().isoformat(),
            "final_eval_results": eval_result,
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
        classifier = cls()
        classifier.metadata = metadata

        # Load model, vectorizer, and label encoder
        classifier.model = joblib.load(os.path.join(model_path, "model.pkl"))
        classifier.vectorizer = joblib.load(os.path.join(model_path, "vectorizer.pkl"))
        classifier.label_encoder = joblib.load(os.path.join(model_path, "label_encoder.pkl"))

        logger.info(f"Model loaded successfully from {model_path}")
        return classifier

    def predict(self, texts, return_probabilities: bool = False):
        """Make predictions on new text data"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not loaded. Use load_for_inference() first.")

        X = self.vectorizer.transform(texts)
        probs = self.model.predict_proba(X)
        predictions = np.argmax(probs, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predictions)

        if return_probabilities:
            return predicted_labels, probs
        else:
            return predicted_labels

    def get_model_info(self):
        """Get model information"""
        return self.metadata


# if __name__ == "__main__":
#     classifier = SKNaiveBayesTextClassifier()
#     metadata = classifier.train(
#         project_folder="data",
#         save_path="models/sk-naive-bayes",
#         text_column="text",
#         label_column="label",
#     )
#     print("Metadata: ", metadata)
#
#     classifier = SKNaiveBayesTextClassifier.load_for_inference(model_path="models/sk-naive-bayes")
#     predictions = classifier.predict(
#         texts=["That superman movie was so bad. I hated it. I would never watch it again."], return_probabilities=True
#     )
#     print(predictions)
#
#     label = classifier.predict(
#         texts=["That superman movie was so bad. I hated it. I would never watch it again."], return_probabilities=False
#     )
#     print(label)
