from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
from monitoring.logger import log_transaction

import joblib

MODEL_PATH = "models/fraud_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()



app = FastAPI(
    title="Fraud Detection API",
    description="Self-Healing Cybersecurity Fraud Detection System",
    version="1.0"
)


# Input schema
class Transaction(BaseModel):
    features: list


@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}


@app.post("/predict_transaction")
def predict(transaction: Transaction):

    global model
    model = load_model()   # ⭐ HOT RELOAD

    if len(transaction.features) != 29:
        return {"error": "Expected 29 features"}

    data = np.array(transaction.features).reshape(1, -1)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    log_transaction(transaction.features, int(prediction))

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(float(probability), 6)
    }