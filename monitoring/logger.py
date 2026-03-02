import pandas as pd
import os

LOG_FILE = "monitoring/transaction_log.csv"

columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Prediction"]


def log_transaction(features, prediction):

    row = features + [prediction]
    df = pd.DataFrame([row], columns=columns)

    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)