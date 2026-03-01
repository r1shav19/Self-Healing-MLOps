"""
Training Pipeline Module

Orchestrates the complete training workflow.
"""

import logging
from typing import Dict, Any
import yaml
import pandas as pd
from pathlib import Path

from src.data_ingestion import DataIngestion
from src.preprocessing import DataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator


logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates the complete ML training pipeline."""

    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.data_ingestion = DataIngestion(self.config.get("data_ingestion", {}))
        self.preprocessor = DataPreprocessor(self.config.get("preprocessing", {}))
        self.trainer = ModelTrainer(self.config.get("training", {}))
        self.evaluator = ModelEvaluator(self.config.get("evaluation", {}))

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def run(self, data_source: str, target_column: str) -> Dict[str, Any]:
        """Execute the complete training pipeline."""
        logger.info("Starting training pipeline")

        # Step 1: Data Ingestion
        logger.info("Step 1: Data Ingestion")
        df = self.data_ingestion.load_from_csv(data_source)

        if not self.data_ingestion.validate_data(df):
            raise ValueError("Data validation failed")

        # Step 2: Preprocessing
        logger.info("Step 2: Data Preprocessing")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X = self.preprocessor.preprocess(X, fit=True)

        # Step 3: Train/Test Split
        logger.info("Step 3: Train/Test Split")
        X_train, X_test, y_train, y_test = self.trainer.split_data(X, y)

        # Step 4: Model Training
        logger.info("Step 4: Model Training")
        self.trainer.train(X_train, y_train)

        # Step 5: Model Evaluation
        logger.info("Step 5: Model Evaluation")
        results = self.evaluator.evaluate(y_test, self.trainer.predict(X_test))

        # Step 6: Model Saving
        logger.info("Step 6: Model Saving")
        model_path = self.trainer.save_model(version="latest")

        results["model_path"] = model_path
        logger.info("Training pipeline completed successfully")

        return results


def main():
    """Entry point for training pipeline."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.training_pipeline <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    pipeline = TrainingPipeline(config_path)
    results = pipeline.run(
        data_source="data/raw/data.csv",
        target_column="target"
    )
    print("Training completed. Results:", results)


if __name__ == "__main__":
    main()
