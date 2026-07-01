"""Tests de la logica de inferencia y de la API FastAPI."""

import os

import pandas as pd
from fastapi.testclient import TestClient

import config
import inference

SAMPLE = {
    "gender": "Male",
    "contract_type": "Month-to-Month",
    "internet_service": "Fiber",
    "senior_citizen": 0,
    "tenure_months": 3,
    "monthly_charges": 95.0,
    "support_tickets_last_6m": 5,
}


def test_risk_label_thresholds():
    assert inference.risk_label(0.1) == "LOW"
    assert inference.risk_label(0.5) == "MEDIUM"
    assert inference.risk_label(0.9) == "HIGH"


def test_predict_frame_shape(trained_model):
    import joblib

    bundle = joblib.load(trained_model)
    df = pd.DataFrame([SAMPLE])
    results = inference.predict_frame(bundle["model"], df)
    assert len(results) == 1
    assert set(results[0]) == {"churn_prediction", "churn_probability", "risk_label"}
    assert 0.0 <= results[0]["churn_probability"] <= 1.0


def test_api_predict(trained_model):
    # Forzar a la API a usar el modelo local entrenado en el fixture
    os.environ["MODEL_PATH"] = trained_model
    inference._bundle = None  # limpiar cache
    config.MODEL_PATH = trained_model

    import api

    client = TestClient(api.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    resp = client.post("/predict", json=SAMPLE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_label"] in ("LOW", "MEDIUM", "HIGH")
