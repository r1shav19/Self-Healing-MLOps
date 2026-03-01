"""
FastAPI Application for Model Inference

Provides REST API endpoints for model predictions.
"""

import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.train import ModelTrainer


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Self-Healing MLOps API",
    description="API for model inference and monitoring",
    version="0.1.0"
)

# Initialize model
trainer = ModelTrainer({})


class PredictionRequest(BaseModel):
    """Request model for prediction."""
    features: List[float]


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    prediction: int
    confidence: float


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting up API server")
    try:
        trainer.load_model(version="latest")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/info")
async def model_info():
    """Get model information."""
    return {
        "name": "Self-Healing MLOps Model",
        "version": "1.0.0",
        "description": "Classification model for production",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction."""
    try:
        if trainer.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert to DataFrame for consistency
        X = pd.DataFrame([request.features])
        prediction = trainer.predict(X)[0]

        return PredictionResponse(
            prediction=int(prediction),
            confidence=0.95  # This would be computed from model probabilities
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Make batch predictions."""
    try:
        if trainer.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        features = [req.features for req in requests]
        X = pd.DataFrame(features)
        predictions = trainer.predict(X)

        return {
            "predictions": [int(p) for p in predictions],
            "count": len(predictions),
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
