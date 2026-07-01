"""Configuracion central del proyecto.

Solo constantes y rutas como strings. Nada de path managers ni clases.
Los valores de AWS se pueden sobreescribir con variables de entorno para no
hardcodear nada sensible (el bucket lleva el account id, no es secreto).
"""

import os

# ── Esquema del dataset ───────────────────────────────────────────────────────
TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"
CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service"]
NUMERIC_FEATURES = [
    "senior_citizen",
    "tenure_months",
    "monthly_charges",
    "support_tickets_last_6m",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# ── Rutas locales (dentro del repo / del contenedor de build) ─────────────────
RAW_DATA_PATH = "data/raw/telco_customer_churn_mlops.csv"
CLEAN_DATA_PATH = "data/processed/clean.csv"
TEST_DATA_PATH = "data/processed/test.csv"
FEATURE_STORE_PATH = "data/feature_store/features.csv"
MODEL_DIR = "models"
MODEL_PATH = "models/model.joblib"
METRICS_PATH = "models/metrics.json"
EVALUATION_PATH = "models/evaluation.json"
MLRUNS_DIR = "mlruns"

# ── AWS ───────────────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Bucket unico por cuenta. Se puede sobreescribir con S3_BUCKET.
S3_BUCKET = os.getenv("S3_BUCKET", "churn-mlops-313694531227")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "churn-model-registry")
ECR_REPOSITORY = os.getenv("ECR_REPOSITORY", "churn-api")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")

# Prefijos (carpetas) dentro del bucket S3
S3_RAW_KEY = "raw/telco_customer_churn_mlops.csv"
S3_FEATURE_STORE_KEY = "feature_store/features.csv"
S3_MODELS_PREFIX = "models"          # models/<version>/model.joblib
S3_MLRUNS_PREFIX = "mlruns"
S3_REPORTS_PREFIX = "reports"

# CloudWatch (monitoreo)
CLOUDWATCH_NAMESPACE = "ChurnMLOps"

# Umbrales de riesgo para risk_label
RISK_MEDIUM_THRESHOLD = 0.40
RISK_HIGH_THRESHOLD = 0.70
