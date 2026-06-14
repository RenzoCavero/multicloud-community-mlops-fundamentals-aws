"""Paso 3 del pipeline: entrenar el modelo de churn.

Que hace:
  - Lee el feature store (CSV).
  - Hace split train/test (estratificado).
  - Entrena un Pipeline sklearn: OneHotEncoder + StandardScaler + clasificador.
  - Registra parametros y metricas en MLflow (file store local ./mlruns).
  - Guarda el modelo (joblib), las metricas (json) y el set de test (para evaluate).

MLflow corre como file store local porque el entrenamiento se ejecuta dentro de
un job efimero de CodeBuild (no hay servidor MLflow persistente). El buildspec
sube ./mlruns a S3 al terminar, asi queda el historial de experimentos.

Uso:
    python src/train.py --model-type logistic-regression
    python src/train.py --model-type random-forest --n-estimators 200
"""

import argparse
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def build_pipeline(model_type: str, hyperparams: dict) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    if model_type == "logistic-regression":
        classifier = LogisticRegression(
            C=hyperparams.get("C", 1.0),
            max_iter=hyperparams.get("max_iter", 1000),
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "random-forest":
        classifier = RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 200),
            max_depth=hyperparams.get("max_depth") or None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"model_type invalido: {model_type!r}")

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.0,
    }


def train(features_path: str, model_type: str, hyperparams: dict, test_size: float = 0.2) -> dict:
    df = pd.read_csv(features_path)
    X = df[config.ALL_FEATURES]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = build_pipeline(model_type, hyperparams)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_score)

    # ── Guardar artefactos ────────────────────────────────────────────────────
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": config.ALL_FEATURES,
        "target_column": config.TARGET_COLUMN,
        "model_type": model_type,
        "hyperparameters": hyperparams,
        "metrics": metrics,
    }
    joblib.dump(bundle, config.MODEL_PATH)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Guardar el set de test para el paso de evaluacion
    os.makedirs(os.path.dirname(config.TEST_DATA_PATH), exist_ok=True)
    test_df = X_test.copy()
    test_df[config.TARGET_COLUMN] = y_test.values
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo de churn")
    parser.add_argument("--features", default=config.FEATURE_STORE_PATH)
    parser.add_argument(
        "--model-type",
        default=os.getenv("MODEL_TYPE", "logistic-regression"),
        choices=["logistic-regression", "random-forest"],
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    hyperparams = {
        "C": args.C,
        "max_iter": args.max_iter,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
    }

    # MLflow: file store local. El buildspec sube ./mlruns a S3.
    mlflow.set_tracking_uri(f"file:{config.MLRUNS_DIR}")
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_params({k: v for k, v in hyperparams.items() if v is not None})
        metrics = train(args.features, args.model_type, hyperparams, args.test_size)
        mlflow.log_metrics(metrics)
        bundle = joblib.load(config.MODEL_PATH)
        mlflow.sklearn.log_model(bundle["model"], artifact_path="model")

    print(f"[train] modelo={args.model_type} -> {config.MODEL_PATH}")
    for k, v in metrics.items():
        print(f"[train] {k}={v:.4f}")


if __name__ == "__main__":
    main()
