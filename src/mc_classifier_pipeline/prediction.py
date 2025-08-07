"""
Unified prediction module for text classification models.
"""

import gc
import logging
import os
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

from mc_classifier_pipeline.utils import detect_model_framework

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Unified predictor for both HuggingFace and sklearn models.
    """

    def __init__(self, model_dir: str):
        """
        Initialize predictor with a trained model.

        Args:
            model_dir: Path to the trained model directory
        """
        self.model_dir = model_dir
        self.framework = detect_model_framework(model_dir)

        if not self.framework:
            raise ValueError(f"Could not detect framework for model at {model_dir}")

        # Load model components
        self._load_model()

    def _load_model(self):
        """Load model components based on detected framework."""
        if self.framework == "hf":
            self._load_huggingface_model()
        elif self.framework == "sklearn":
            self._load_sklearn_model()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def _load_huggingface_model(self):
        """Load HuggingFace model components."""
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

        # Load label encoder
        label_encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")
        if os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
        else:
            raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded HuggingFace model on {self.device}")

    def _load_sklearn_model(self):
        """Load sklearn model components."""
        # Load vectorizer
        vectorizer_path = os.path.join(self.model_dir, "vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

        # Load model
        model_path = os.path.join(self.model_dir, "model.pkl")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load label encoder
        label_encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")
        if os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
        else:
            raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")

        logger.info("Loaded sklearn model")

    def predict(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: Optional[int] = None,
        return_probabilities: bool = False,
    ) -> List[str]:
        """
        Make predictions on texts.

        Args:
            texts: List of texts to predict on
            batch_size: Batch size for prediction (for HuggingFace models)
            max_length: Maximum sequence length (for HuggingFace models)
            return_probabilities: Whether to return probabilities (not implemented yet)

        Returns:
            List of predicted labels
        """
        if self.framework == "hf":
            return self._predict_huggingface(texts, batch_size, max_length)
        elif self.framework == "sklearn":
            return self._predict_sklearn(texts)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def _predict_huggingface(
        self, texts: List[str], batch_size: int = 32, max_length: Optional[int] = None
    ) -> List[str]:
        """Make predictions using HuggingFace model."""
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()

            # Decode predictions
            batch_labels = self.label_encoder.inverse_transform(batch_predictions)
            predictions.extend(batch_labels)

        return predictions

    def _predict_sklearn(self, texts: List[str]) -> List[str]:
        """Make predictions using sklearn model."""
        # Vectorize texts
        X = self.vectorizer.transform(texts)

        # Predict
        predictions = self.model.predict(X)

        # Decode predictions
        labels = self.label_encoder.inverse_transform(predictions)

        return labels.tolist()

    def cleanup(self):
        """Clean up resources (especially for GPU memory)."""
        if self.framework == "hf" and hasattr(self, "model"):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        gc.collect()


def predict_labels_hf(
    model_dir: str,
    texts: List[str],
    max_length: Optional[int] = None,
    batch_size: int = 32,
) -> List[str]:
    """
    Predict labels using HuggingFace model (legacy function for backward compatibility).

    Args:
        model_dir: Path to HuggingFace model directory
        texts: List of texts to predict on
        max_length: Maximum sequence length
        batch_size: Batch size for prediction

    Returns:
        List of predicted labels
    """
    predictor = ModelPredictor(model_dir)
    try:
        return predictor.predict(texts, batch_size, max_length)
    finally:
        predictor.cleanup()


def predict_labels_sklearn(
    model_dir: str,
    texts: List[str],
) -> List[str]:
    """
    Predict labels using sklearn model (legacy function for backward compatibility).

    Args:
        model_dir: Path to sklearn model directory
        texts: List of texts to predict on

    Returns:
        List of predicted labels
    """
    predictor = ModelPredictor(model_dir)
    try:
        return predictor.predict(texts)
    finally:
        predictor.cleanup()


def cleanup_memory():
    """Clean up memory and GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
