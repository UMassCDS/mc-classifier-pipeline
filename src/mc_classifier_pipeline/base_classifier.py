"""
Base classifier class providing common functionality for text classification models.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

from mc_classifier_pipeline.utils import (
    load_data_splits,
    compute_classification_metrics,
    save_metadata,
)

logger = logging.getLogger(__name__)


class BaseTextClassifier(ABC):
    """
    Abstract base class for text classification models.

    Provides common functionality for data loading, metrics computation,
    model saving/loading, and metadata management.
    """

    def __init__(self, model_name: str = "base_classifier"):
        self.model_name = model_name
        self.label_encoder = LabelEncoder()
        self.metadata = {}
        self.is_trained = False

    def load_data(
        self, project_folder: str, text_column: str = "text", label_column: str = "label"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test data from CSV files.

        Args:
            project_folder: Path to folder containing train.csv and test.csv
            text_column: Name of the text column
            label_column: Name of the label column

        Returns:
            Tuple of (train_df, test_df)
        """
        train_df, test_df = load_data_splits(project_folder, text_column, label_column)

        logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")

        return train_df, test_df

    def prepare_labels(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, label_column: str = "label"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and encode labels for training.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            label_column: Name of the label column

        Returns:
            Tuple of (train_labels, test_labels) as encoded arrays
        """
        # Encode labels
        all_labels = pd.concat([train_df[label_column], test_df[label_column]]).unique()
        self.label_encoder.fit(all_labels)

        train_labels = self.label_encoder.transform(train_df[label_column])
        test_labels = self.label_encoder.transform(test_df[label_column])

        logger.info(f"Number of unique labels: {len(self.label_encoder.classes_)}")
        logger.info(
            f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}"
        )

        return train_labels, test_labels

    def compute_metrics(self, y_true: List, y_pred: List) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary containing accuracy, precision, recall, and f1 scores
        """
        return compute_classification_metrics(y_true, y_pred)

    @abstractmethod
    def prepare_datasets(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
        **kwargs,
    ):
        """
        Prepare training and test datasets for the specific model type.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            text_column: Name of the text column
            label_column: Name of the label column
            **kwargs: Additional arguments specific to the model type
        """
        pass

    @abstractmethod
    def train(
        self,
        project_folder: str,
        save_path: str,
        text_column: str = "text",
        label_column: str = "label",
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        """
        Train the model.

        Args:
            project_folder: Path to folder containing train.csv and test.csv
            save_path: Path to save the trained model
            text_column: Name of the text column
            label_column: Name of the label column
            hyperparams: Optional hyperparameters for training
        """
        pass

    @abstractmethod
    def predict(self, texts: List[str], **kwargs) -> List[str]:
        """
        Make predictions on new texts.

        Args:
            texts: List of texts to predict on
            **kwargs: Additional arguments specific to the model type

        Returns:
            List of predicted labels
        """
        pass

    def save_model(self, save_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the trained model and metadata.

        Args:
            save_path: Path to save the model
            metadata: Optional additional metadata to save
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save label encoder
        label_encoder_path = save_dir / "label_encoder.pkl"
        joblib.dump(self.label_encoder, label_encoder_path)

        # Prepare metadata
        model_metadata = {
            "model_name": self.model_name,
            "framework": self.get_framework_name(),
            "created_at": datetime.now().isoformat(),
            "label_encoder_classes": self.label_encoder.classes_.tolist(),
            "num_classes": len(self.label_encoder.classes_),
            "is_trained": self.is_trained,
            **(metadata or {}),
        }

        # Save metadata
        metadata_path = save_dir / "metadata.json"
        save_metadata(model_metadata, metadata_path)

        # Save model-specific components
        self._save_model_components(save_dir)

        logger.info(f"Model saved to {save_path}")

    @abstractmethod
    def _save_model_components(self, save_dir: Path) -> None:
        """
        Save model-specific components (to be implemented by subclasses).

        Args:
            save_dir: Directory to save components in
        """
        pass

    @classmethod
    def load_for_inference(cls, model_path: str):
        """
        Load a trained model for inference.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded model instance
        """
        model_dir = Path(model_path)

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # Load label encoder
        label_encoder_path = model_dir / "label_encoder.pkl"
        if not label_encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")

        label_encoder = joblib.load(label_encoder_path)

        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Create instance and load model-specific components
        instance = cls._create_instance_from_metadata(metadata)
        instance.label_encoder = label_encoder
        instance.metadata = metadata
        instance.is_trained = True

        # Load model-specific components
        instance._load_model_components(model_dir)

        logger.info(f"Model loaded from {model_path}")
        return instance

    @classmethod
    @abstractmethod
    def _create_instance_from_metadata(cls, metadata: Dict[str, Any]):
        """
        Create an instance of the classifier from metadata (to be implemented by subclasses).

        Args:
            metadata: Model metadata

        Returns:
            Classifier instance
        """
        pass

    @abstractmethod
    def _load_model_components(self, model_dir: Path) -> None:
        """
        Load model-specific components (to be implemented by subclasses).

        Args:
            model_dir: Directory containing model components
        """
        pass

    @abstractmethod
    def get_framework_name(self) -> str:
        """
        Get the framework name for this classifier.

        Returns:
            Framework name (e.g., "hf", "sklearn")
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "framework": self.get_framework_name(),
            "is_trained": self.is_trained,
            "num_classes": len(self.label_encoder.classes_) if hasattr(self, "label_encoder") else 0,
            "classes": self.label_encoder.classes_.tolist() if hasattr(self, "label_encoder") else [],
            "metadata": self.metadata,
        }
