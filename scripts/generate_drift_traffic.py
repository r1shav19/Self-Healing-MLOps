import pandas as pd
import requests
import time
import random

URL = "http://127.0.0.1:8000/predict_transaction"

df = pd.read_csv("data/raw/creditcard.csv")

print(" Sending live traffic...")

for _ in range(50):

    row = df.sample(1).iloc[0]
    features = row.drop(["Class", "Time"]).tolist()

    response = requests.post(
        URL,
        json={"features": features}
    )

    print("Sent transaction:", response.status_code)

    time.sleep(random.uniform(0.3, 1.2))

print(" Traffic simulation complete")
