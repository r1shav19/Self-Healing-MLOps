"""
Model Evaluation Module

Handles comprehensive model evaluation and metrics tracking.
"""

import logging
from typing import Dict, Any
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles model evaluation and metrics computation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator with config."""
        self.config = config

    def compute_classification_metrics(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> Dict[str, Any]:
        """Compute classification metrics."""
        logger.info("Computing classification metrics")

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        return metrics

    def compute_confusion_matrix(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> Dict[str, Any]:
        """Compute confusion matrix."""
        logger.info("Computing confusion matrix")
        cm = confusion_matrix(y_true, y_pred)
        return {"confusion_matrix": cm.tolist()}

    def generate_classification_report(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> str:
        """Generate detailed classification report."""
        logger.info("Generating classification report")
        report = classification_report(y_true, y_pred)
        return report

    def evaluate(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> Dict[str, Any]:
        """Execute full evaluation."""
        logger.info("Starting model evaluation")

        results = {
            **self.compute_classification_metrics(y_true, y_pred),
            **self.compute_confusion_matrix(y_true, y_pred),
        }

        report = self.generate_classification_report(y_true, y_pred)
        results["classification_report"] = report

        logger.info("Evaluation completed")
        return results

    def check_performance_degradation(
        self, current_metrics: Dict[str, float], baseline_metrics: Dict[str, float]
    ) -> bool:
        """Check if model performance has degraded."""
        logger.info("Checking for performance degradation")

        threshold = self.config.get("degradation_threshold", 0.05)
        metric_to_track = self.config.get("metric_to_track", "f1")

        if metric_to_track not in current_metrics or metric_to_track not in baseline_metrics:
            logger.warning(f"Metric {metric_to_track} not found in metrics")
            return False

        degradation = baseline_metrics[metric_to_track] - current_metrics[metric_to_track]

        if degradation > threshold:
            logger.warning(
                f"Performance degradation detected: {metric_to_track} degraded by {degradation:.4f}"
            )
            return True

        logger.info("No performance degradation detected")
        return False


def main():
    """Entry point for evaluation."""
    pass


if __name__ == "__main__":
    main()
