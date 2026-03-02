import pandas as pd
import numpy as np


TRAIN_PATH = "data/raw/creditcard.csv"
LOG_PATH = "monitoring/transaction_log.csv"


def detect_drift():

    train = pd.read_csv(TRAIN_PATH)
    live = pd.read_csv(LOG_PATH)

    train = train.drop(["Time", "Class"], axis=1)

    drift_scores = {}

    for col in train.columns:
        train_mean = train[col].mean()
        live_mean = live[col].mean()

        drift = abs(train_mean - live_mean)

        drift_scores[col] = drift

    avg_drift = np.mean(list(drift_scores.values()))

    if avg_drift > 0.5:
        print("⚠️ DATA DRIFT DETECTED")
    else:
        print("✅ Data Stable")

    return avg_drift

if __name__ == "__main__":
    drift = detect_drift()
    print(f"Average Drift Score: {drift}")    