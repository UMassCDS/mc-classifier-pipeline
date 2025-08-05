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
from sklearn.model_selection import cross_val_score
import joblib
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

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
        self.best_params = None
        self.study = None

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

    def objective(self, trial, train_texts, train_labels, cv_folds=3):
        """Optuna objective function for hyperparameter optimization"""

        # Suggest hyperparameters
        ngram_min = trial.suggest_int("ngram_min", 1, 2)
        ngram_max = trial.suggest_int("ngram_max", ngram_min, 3)
        min_df = trial.suggest_int("min_df", 1, 10)
        max_df = trial.suggest_float("max_df", 0.5, 1.0)
        alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)

        # Additional TF-IDF parameters
        max_features = trial.suggest_categorical("max_features", [None, 1000, 5000, 10000, 20000])
        use_idf = trial.suggest_categorical("use_idf", [True, False])
        sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False])

        try:
            # Create vectorizer with trial parameters
            vectorizer = TfidfVectorizer(
                ngram_range=(ngram_min, ngram_max),
                min_df=min_df,
                max_df=max_df,
                max_features=max_features,
                use_idf=use_idf,
                sublinear_tf=sublinear_tf,
                stop_words="english",
            )

            # Transform texts
            X_train = vectorizer.fit_transform(train_texts)

            # Create model with trial parameters
            model = MultinomialNB(alpha=alpha)

            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, train_labels, cv=cv_folds, scoring="f1_weighted", n_jobs=-1)

            # Return mean CV score
            mean_score = cv_scores.mean()

            # Report intermediate value for pruning
            trial.report(mean_score, step=0)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            return mean_score

        except Exception as e:
            logger.warning(f"Trial failed with error: {e}")
            return 0.0

    def optimize_hyperparameters(
        self,
        train_texts,
        train_labels,
        n_trials: int = 100,
        cv_folds: int = 3,
        direction: str = "maximize",
        sampler_seed: Optional[int] = 42,
        study_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""

        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")

        # Create study
        if study_name is None:
            study_name = f"naive_bayes_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        sampler = TPESampler(seed=sampler_seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

        self.study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner, study_name=study_name)

        # Optimize
        self.study.optimize(
            lambda trial: self.objective(trial, train_texts, train_labels, cv_folds),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        # Get best parameters
        self.best_params = self.study.best_params
        best_score = self.study.best_value

        logger.info("Optimization completed!")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        # Return optimization results
        return {
            "best_params": self.best_params,
            "best_score": best_score,
            "n_trials": len(self.study.trials),
            "study_name": study_name,
        }

    def train(
        self,
        project_folder: str,
        save_path: str,
        text_column: str = "text",
        label_column: str = "label",
        hyperparams: Optional[Dict[str, Any]] = None,
        optimize_hyperparams: bool = False,
        optuna_trials: int = 100,
        cv_folds: int = 3,
    ):
        """Train the Naive Bayes model with optional hyperparameter optimization"""

        # Load and prepare data
        train_df, test_df = self.load_data(project_folder, text_column, label_column)
        (train_texts, train_labels), (test_texts, test_labels) = self.prepare_datasets(
            train_df, test_df, text_column, label_column
        )

        # Default hyperparameters
        default_hyperparams = {
            "ngram_range": (1, 1),
            "min_df": 1,
            "max_df": 1.0,
            "alpha": 1.0,
            "max_features": None,
            "use_idf": True,
            "sublinear_tf": False,
            "stop_words": "english",
        }

        optimization_results = None

        if optimize_hyperparams:
            # Optimize hyperparameters
            optimization_results = self.optimize_hyperparameters(
                train_texts, train_labels, n_trials=optuna_trials, cv_folds=cv_folds
            )

            # Update hyperparameters with optimized values
            optimized_params = {
                "ngram_range": (self.best_params["ngram_min"], self.best_params["ngram_max"]),
                "min_df": self.best_params["min_df"],
                "max_df": self.best_params["max_df"],
                "alpha": self.best_params["alpha"],
                "max_features": self.best_params["max_features"],
                "use_idf": self.best_params["use_idf"],
                "sublinear_tf": self.best_params["sublinear_tf"],
                "stop_words": "english",
            }
            default_hyperparams.update(optimized_params)
            logger.info("Using optimized hyperparameters")

        if hyperparams:
            default_hyperparams.update(hyperparams)
            logger.info("Using provided hyperparameters (may override optimized ones)")

        # Prepare vectorizer and model with final hyperparameters
        self.vectorizer = TfidfVectorizer(
            ngram_range=default_hyperparams["ngram_range"],
            min_df=default_hyperparams["min_df"],
            max_df=default_hyperparams["max_df"],
            max_features=default_hyperparams["max_features"],
            use_idf=default_hyperparams["use_idf"],
            sublinear_tf=default_hyperparams["sublinear_tf"],
            stop_words=default_hyperparams["stop_words"],
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

        # Save Optuna study if optimization was performed
        if optimize_hyperparams and self.study is not None:
            joblib.dump(self.study, os.path.join(save_path, "optuna_study.pkl"))

        # Create metadata
        self.metadata = {
            "framework": "naive-bayes",
            "model_type": "sklearn-naive-bayes",
            "num_labels": len(self.label_encoder.classes_),
            "label_classes": self.label_encoder.classes_.tolist(),
            "training_samples": len(train_df),
            "test_samples": len(test_df),
            "hyperparameters": default_hyperparams,
            "hyperparameter_optimization": {"enabled": optimize_hyperparams, "results": optimization_results},
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

        # Load Optuna study if it exists
        study_path = os.path.join(model_path, "optuna_study.pkl")
        if os.path.exists(study_path):
            classifier.study = joblib.load(study_path)
            logger.info("Optuna study loaded successfully")

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

    def get_optimization_history(self):
        """Get optimization history if available"""
        if self.study is None:
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


# Example usage
if __name__ == "__main__":
    # Train with hyperparameter optimization
    classifier = SKNaiveBayesTextClassifier()
    metadata = classifier.train(
        project_folder="data",
        save_path="models/sk-naive-bayes-optimized",
        text_column="text",
        label_column="label",
        optimize_hyperparams=True,
        optuna_trials=50,
        cv_folds=3,
    )
    print("Metadata: ", metadata)

    # Load and make predictions
    classifier = SKNaiveBayesTextClassifier.load_for_inference(model_path="models/sk-naive-bayes-optimized")
    predictions = classifier.predict(
        texts=["That superman movie was so bad. I hated it. I would never watch it again."], return_probabilities=True
    )
    print("Predictions:", predictions)

    # Get optimization history
    opt_history = classifier.get_optimization_history()
    if opt_history:
        print(f"Best optimization score: {opt_history['best_value']:.4f}")
        print(f"Best parameters: {opt_history['best_params']}")
