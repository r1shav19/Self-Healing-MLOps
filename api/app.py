from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from monitoring.logger import log_transaction
import threading
import time
from monitoring.drift_detector import detect_drift
import pickle

# Load trained model
model = joblib.load("models/fraud_model.pkl")
def background_drift_monitor(): 
    while True:
        print("🔎 Checking for drift...")
        detect_drift()
        time.sleep(150)  


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
    
    log_transaction(transaction.features, int(prediction))
    
    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(float(probability), 6)
    }
@app.on_event("startup")
def start_monitor():
    thread = threading.Thread(target=background_drift_monitor)
    thread.daemon = True
    thread.start()