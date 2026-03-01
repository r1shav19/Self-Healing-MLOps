"""
Preprocessing Module

Handles data cleaning, transformation, and feature engineering.
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing and feature engineering."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor with config."""
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        logger.info("Handling missing values")
        strategy = self.config.get("missing_value_strategy", "mean")

        if strategy == "drop":
            df = df.dropna()
        elif strategy == "mean":
            for col in df.select_dtypes(include=["float64", "int64"]).columns:
                df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == "forward_fill":
            df.fillna(method="ffill", inplace=True)

        logger.info(f"Missing values handled using {strategy} strategy")
        return df

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        logger.info("Removing outliers")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        logger.info(f"Removed outliers. Remaining rows: {len(df)}")
        return df

    def encode_categorical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical variables."""
        logger.info("Encoding categorical variables")
        categorical_cols = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            elif col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))

        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info("Scaling features")
        numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns

        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])

        return df

    def preprocess(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Execute full preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline")
        df = self.handle_missing_values(df)
        df = self.remove_outliers(df)
        df = self.encode_categorical(df, fit=fit)
        df = self.scale_features(df, fit=fit)
        logger.info("Preprocessing completed")
        return df


def main():
    """Entry point for preprocessing."""
    pass


if __name__ == "__main__":
    main()
