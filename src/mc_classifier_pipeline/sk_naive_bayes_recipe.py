import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from mc_classifier_pipeline.base_classifier import BaseTextClassifier
from mc_classifier_pipeline.utils import configure_logging

# Set up logging
configure_logging()
logger = logging.getLogger(__name__)


class SKNaiveBayesTextClassifier(BaseTextClassifier):
    """Naive Bayes text classifier with training and inference capabilities (scikit-learn)"""

    def __init__(self):
        super().__init__("sklearn_naive_bayes")
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()

    def prepare_datasets(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
    ):
        """Prepare training and test datasets"""
        # Prepare labels using base class method
        train_labels, test_labels = self.prepare_labels(train_df, test_df, label_column)

        train_texts = train_df[text_column].tolist()
        test_texts = test_df[text_column].tolist()

        return (train_texts, train_labels), (test_texts, test_labels)

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

        # Create metadata
        metadata = {
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

        # Save model using base class method
        self.save_model(save_path, metadata)
        self.is_trained = True

        logger.info("Training completed successfully!")
        logger.info(f"Final evaluation results: {eval_result}")

        return self.metadata

    @classmethod
    def _create_instance_from_metadata(cls, metadata: Dict[str, Any]):
        """Create an instance from metadata."""
        return cls()

    def _load_model_components(self, model_dir: Path) -> None:
        """Load model-specific components."""
        import joblib

        self.model = joblib.load(model_dir / "model.pkl")
        self.vectorizer = joblib.load(model_dir / "vectorizer.pkl")

    def _save_model_components(self, save_dir: Path) -> None:
        """Save model-specific components."""
        import joblib

        joblib.dump(self.model, save_dir / "model.pkl")
        joblib.dump(self.vectorizer, save_dir / "vectorizer.pkl")

    def get_framework_name(self) -> str:
        """Get the framework name."""
        return "sklearn"

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
        return super().get_model_info()


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
