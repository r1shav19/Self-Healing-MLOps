from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model
model = joblib.load("models/fraud_model.pkl")

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

    if len(transaction.features) != 29:
        return {"error": "Expected 29 features"}

    data = np.array(transaction.features).reshape(1, -1)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    
    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(float(probability), 6)
    }