import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List
import ast

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
import joblib

from mc_classifier_pipeline.utils import configure_logging

# Set up logging
configure_logging()
logger = logging.getLogger(__name__)


class SKNaiveBayesTextClassifier:
    """Naive Bayes text classifier with training and inference capabilities (scikit-learn)"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()
        self.label_encoder = None
        self.label_binarizer = None
        self.metadata = {}
        self.is_multi_label = False

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

    def prepare_datasets(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
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

            train_texts = train_df[text_column].tolist()
            test_texts = test_df[text_column].tolist()

            return (train_texts, train_labels_binary), (test_texts, test_labels_binary)

        else:
            # Single-label or binary classification
            self.label_encoder = LabelEncoder()
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

    def compute_metrics_single_label(self, y_true, y_pred):
        """Compute metrics for single-label classification"""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        accuracy = accuracy_score(y_true, y_pred)
        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

    def compute_metrics_multi_label(self, y_true, y_pred):
        """Compute metrics for multi-label classification"""
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

        # Subset accuracy (exact match ratio)
        subset_accuracy = accuracy_score(y_true, y_pred)

        return {"subset_accuracy": subset_accuracy, "f1": f1, "precision": precision, "recall": recall}

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

        self.is_multi_label = is_multi_label

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

        # Prepare datasets
        (train_texts, train_labels), (test_texts, test_labels) = self.prepare_datasets(
            train_df_processed, test_df_processed, text_column, label_column, is_multi_label, target_labels
        )

        # Prepare vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=default_hyperparams["ngram_range"],
            min_df=default_hyperparams["min_df"],
            max_df=default_hyperparams["max_df"],
        )
        X_train = self.vectorizer.fit_transform(train_texts)
        X_test = self.vectorizer.transform(test_texts)

        # Prepare model
        base_model = MultinomialNB(alpha=default_hyperparams["alpha"])

        if is_multi_label:
            # Use MultiOutputClassifier for multi-label classification
            self.model = MultiOutputClassifier(base_model)
        else:
            self.model = base_model

        # Train
        logger.info("Starting training...")
        self.model.fit(X_train, train_labels)

        # Evaluate
        logger.info("Evaluating on test set...")
        y_pred = self.model.predict(X_test)

        if is_multi_label:
            eval_result = self.compute_metrics_multi_label(test_labels, y_pred)
        else:
            eval_result = self.compute_metrics_single_label(test_labels, y_pred)

        # Save model, vectorizer, and encoders
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(self.model, os.path.join(save_path, "model.pkl"))
        joblib.dump(self.vectorizer, os.path.join(save_path, "vectorizer.pkl"))

        if is_multi_label:
            joblib.dump(self.label_binarizer, os.path.join(save_path, "label_binarizer.pkl"))
        else:
            joblib.dump(self.label_encoder, os.path.join(save_path, "label_encoder.pkl"))

        # Create metadata
        self.metadata = {
            "framework": "sklearn",
            "model_type": "sklearn-naive-bayes",
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
        classifier.is_multi_label = metadata.get("is_multi_label", False)

        # Load model and vectorizer
        classifier.model = joblib.load(os.path.join(model_path, "model.pkl"))
        classifier.vectorizer = joblib.load(os.path.join(model_path, "vectorizer.pkl"))

        # Load appropriate encoder
        if classifier.is_multi_label:
            binarizer_path = os.path.join(model_path, "label_binarizer.pkl")
            classifier.label_binarizer = joblib.load(binarizer_path)
        else:
            encoder_path = os.path.join(model_path, "label_encoder.pkl")
            classifier.label_encoder = joblib.load(encoder_path)

        logger.info(f"Model loaded successfully from {model_path}")
        return classifier

    def predict(self, texts, return_probabilities: bool = False):
        """Make predictions on new text data"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not loaded. Use load_for_inference() first.")

        X = self.vectorizer.transform(texts)

        if self.is_multi_label:
            # Multi-label prediction
            predictions = self.model.predict(X)
            predicted_labels = self.label_binarizer.inverse_transform(predictions)

            if return_probabilities:
                # For multi-label, we need to get probabilities from each binary classifier
                try:
                    # This will work if the base estimator supports predict_proba
                    probs = []
                    for i, estimator in enumerate(self.model.estimators_):
                        if hasattr(estimator, "predict_proba"):
                            # Get probability of positive class for this label
                            prob_positive = estimator.predict_proba(X)[:, 1]
                            probs.append(prob_positive)
                        else:
                            # Fallback: use decision function or predictions
                            if hasattr(estimator, "decision_function"):
                                scores = estimator.decision_function(X)
                                # Convert to probabilities using sigmoid-like transformation
                                prob_positive = 1 / (1 + np.exp(-scores))
                            else:
                                # Last resort: use predictions as probabilities
                                prob_positive = estimator.predict(X).astype(float)
                            probs.append(prob_positive)

                    probs = np.column_stack(probs)
                    return predicted_labels, probs
                except Exception as e:
                    logger.warning(f"Could not compute probabilities for multi-label: {e}")
                    return predicted_labels, None
            else:
                return predicted_labels
        else:
            # Single-label prediction
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
