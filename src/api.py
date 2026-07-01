"""API de inferencia FastAPI para predecir churn.

Endpoints:
    GET  /health        -> estado del servicio y del modelo
    GET  /model/info    -> metadata del modelo cargado
    POST /predict       -> prediccion individual
    POST /predict/batch -> prediccion para varios clientes

Ejecucion local:
    PYTHONPATH=src uvicorn api:app --host 0.0.0.0 --port 8000

En produccion corre dentro del contenedor Docker en EC2 y el modelo se descarga
del Model Registry (DynamoDB + S3) la primera vez que se invoca.

Ejemplo:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
      "gender": "Male", "contract_type": "Month-to-Month", "internet_service": "Fiber",
      "senior_citizen": 0, "tenure_months": 12, "monthly_charges": 70.5,
      "support_tickets_last_6m": 2 }'
"""

from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
import inference


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    contract_type: Literal["Month-to-Month", "One Year", "Two Year"]
    internet_service: Literal["DSL", "Fiber", "None"]
    senior_citizen: int = Field(..., ge=0, le=1)
    tenure_months: int = Field(..., ge=0)
    monthly_charges: float = Field(..., gt=0)
    support_tickets_last_6m: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_label: Literal["LOW", "MEDIUM", "HIGH"]


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures]


app = FastAPI(title="Churn Prediction API", version="1.0.0")


@app.get("/health")
def health():
    try:
        bundle = inference.load_model()
        return {"status": "ok", "model_type": bundle.get("model_type", "unknown")}
    except Exception as exc:  # noqa: BLE001 - el health no debe tumbar el servicio
        return {"status": "degraded", "error": str(exc)}


@app.get("/model/info")
def model_info():
    try:
        bundle = inference.load_model()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "model_type": bundle.get("model_type", "unknown"),
        "hyperparameters": bundle.get("hyperparameters", {}),
        "training_metrics": bundle.get("metrics", {}),
        "feature_columns": bundle.get("feature_columns", config.ALL_FEATURES),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        bundle = inference.load_model()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    df = pd.DataFrame([customer.model_dump()])
    return inference.predict_frame(bundle["model"], df)[0]


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if not request.customers:
        raise HTTPException(status_code=400, detail="La lista de clientes esta vacia")
    try:
        bundle = inference.load_model()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    df = pd.DataFrame([c.model_dump() for c in request.customers])
    results = inference.predict_frame(bundle["model"], df)
    return {"predictions": results, "total": len(results)}
