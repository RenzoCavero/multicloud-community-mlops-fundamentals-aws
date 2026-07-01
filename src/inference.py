"""Logica de inferencia compartida por la API y el batch.

Resuelve el modelo en este orden:
  1. Si MODEL_PATH apunta a un archivo local existente -> lo usa (dev / tests).
  2. Si no, consulta el Model Registry (DynamoDB) la version Production mas
     reciente, descarga el model.joblib desde S3 y lo carga (produccion en EC2).

Asi el mismo codigo funciona en local y en el contenedor desplegado.
"""

import os
import tempfile

import joblib
import pandas as pd

import config

_bundle = None


def _download_from_registry() -> str:
    """Devuelve la ruta local del model.joblib de la version Production mas reciente."""
    import boto3

    dynamodb = boto3.resource("dynamodb", region_name=config.AWS_REGION)
    table = dynamodb.Table(config.DYNAMODB_TABLE)
    resp = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("model_name").eq(config.MODEL_NAME),
        ScanIndexForward=False,  # version descendente
    )
    items = [i for i in resp.get("Items", []) if i.get("status") == "Production"]
    if not items:
        raise RuntimeError(
            f"No hay modelos Production en DynamoDB '{config.DYNAMODB_TABLE}'. "
            "Corre primero el entrenamiento (CodeBuild)."
        )
    latest = sorted(items, key=lambda i: i["created_at"], reverse=True)[0]
    s3_uri = latest["s3_model_uri"]
    print(f"[inference] cargando modelo del registry: {latest['version']} ({s3_uri})")

    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    local_path = os.path.join(tempfile.gettempdir(), "model.joblib")
    boto3.client("s3", region_name=config.AWS_REGION).download_file(bucket, key, local_path)
    return local_path


def load_model():
    """Carga (con cache) el bundle del modelo."""
    global _bundle
    if _bundle is not None:
        return _bundle

    local_path = os.getenv("MODEL_PATH", config.MODEL_PATH)
    if os.path.exists(local_path):
        print(f"[inference] cargando modelo local: {local_path}")
        path = local_path
    else:
        path = _download_from_registry()

    raw = joblib.load(path)
    _bundle = raw if isinstance(raw, dict) else {"model": raw}
    return _bundle


def risk_label(prob: float) -> str:
    if prob < config.RISK_MEDIUM_THRESHOLD:
        return "LOW"
    if prob < config.RISK_HIGH_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def predict_frame(model, df: pd.DataFrame) -> list[dict]:
    """Predice sobre un DataFrame con las columnas de ALL_FEATURES."""
    X = df[config.ALL_FEATURES]
    probs = model.predict_proba(X)[:, 1]
    return [
        {
            "churn_prediction": int(prob >= 0.5),
            "churn_probability": round(float(prob), 4),
            "risk_label": risk_label(float(prob)),
        }
        for prob in probs
    ]
