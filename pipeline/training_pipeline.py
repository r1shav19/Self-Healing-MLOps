from src.preprocessing import preprocess_data
from src.train import train_model

import os
import pandas as pd


def run_training_pipeline():

    print("Loading fraud dataset...")

    BASE_DATA = "data/raw/creditcard.csv"
    LIVE_DATA = "monitoring/transaction_log.csv"

    # Load base dataset
    df = pd.read_csv(BASE_DATA)

    # ✅ Inject live production data
    if os.path.exists(LIVE_DATA):

        print("Adding live transaction data for retraining...")

        live = pd.read_csv(LIVE_DATA)

        if "Prediction" in live.columns:
            live = live.drop(columns=["Prediction"])

        # pseudo label (demo)
        live["Class"] = 0

        df = pd.concat([df, live], ignore_index=True)

    print("Final Training Dataset Shape:", df.shape)

    # ======================
    # PREPROCESS
    # ======================
    print("Preprocessing transactions...")
    X, y = preprocess_data(df)

    # ======================
    # TRAIN
    # ======================
    print("Training fraud detection model...")
    train_model(X, y)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    run_training_pipeline()