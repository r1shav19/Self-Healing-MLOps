"""
Model Training Module

Handles model training, validation, and versioning.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with config."""
        self.config = config
        self.model = None
        self.model_path = Path(config.get("model_path", "models"))
        self.model_path.mkdir(parents=True, exist_ok=True)

    def split_data(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        logger.info(f"Splitting data with test_size={test_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""
        logger.info("Starting model training")

        model_type = self.config.get("model_type", "random_forest")

        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config.get("n_estimators", 100),
                max_depth=self.config.get("max_depth", None),
                random_state=self.config.get("random_state", 42),
                n_jobs=-1
            )

        self.model.fit(X_train, y_train)
        logger.info("Model training completed")

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        logger.info("Evaluating model")

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        y_pred = self.model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save_model(self, version: str = "latest") -> str:
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        filepath = self.model_path / f"model_{version}.joblib"
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        return str(filepath)

    def load_model(self, version: str = "latest") -> None:
        """Load trained model."""
        filepath = self.model_path / f"model_{version}.joblib"
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not loaded. Load or train a model first.")
        return self.model.predict(X)


def main():
    """Entry point for training."""
    pass


if __name__ == "__main__":
    main()
