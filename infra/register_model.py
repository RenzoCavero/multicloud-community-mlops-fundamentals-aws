"""Paso 5 del pipeline: registrar el modelo en el Model Registry (S3 + DynamoDB).

Por que S3 + DynamoDB en lugar de SageMaker Model Registry:
  - La cuenta tiene cuotas de SageMaker restringidas.
  - S3 guarda el artefacto (model.joblib) y DynamoDB guarda la metadata/version.
  - Es el patron "registry casero" mas simple y suficiente para una clase.

Que hace:
  1. Calcula una version (numero de build de CodeBuild o timestamp).
  2. Sube model.joblib, metrics.json y evaluation.json a
     s3://<bucket>/models/<version>/
  3. Escribe un item en DynamoDB con la metadata y status="Production"
     (solo llega aqui si paso el quality gate de evaluate.py).
  4. Crea la tabla DynamoDB si no existe (comodo para la clase).

Uso:
    python infra/register_model.py
    PYTHONPATH=src python infra/register_model.py   # si config no esta en el path
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from decimal import Decimal

import boto3

# config.py vive en src/. Lo agregamos al path para poder ejecutar este script
# desde la raiz del repo (asi lo llama el buildspec).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config  # noqa: E402


def resolve_version() -> str:
    # CodeBuild expone CODEBUILD_BUILD_NUMBER; en local usamos timestamp.
    build_number = os.getenv("CODEBUILD_BUILD_NUMBER")
    if build_number:
        return f"build-{build_number}"
    return "local-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def ensure_table(dynamodb) -> None:
    # describe_table (con alcance a la tabla) en vez de ListTables (que exige
    # Resource "*"), para mantener el permiso IAM al minimo.
    client = dynamodb.meta.client
    try:
        client.describe_table(TableName=config.DYNAMODB_TABLE)
        return
    except client.exceptions.ResourceNotFoundException:
        pass
    print(f"[register] creando tabla DynamoDB {config.DYNAMODB_TABLE}...")
    table = dynamodb.create_table(
        TableName=config.DYNAMODB_TABLE,
        KeySchema=[
            {"AttributeName": "model_name", "KeyType": "HASH"},
            {"AttributeName": "version", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "model_name", "AttributeType": "S"},
            {"AttributeName": "version", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()


def register(model_path: str, metrics_path: str, evaluation_path: str) -> dict:
    version = resolve_version()
    s3_prefix = f"{config.S3_MODELS_PREFIX}/{version}"

    s3 = boto3.client("s3", region_name=config.AWS_REGION)
    uploads = {
        model_path: f"{s3_prefix}/model.joblib",
        metrics_path: f"{s3_prefix}/metrics.json",
        evaluation_path: f"{s3_prefix}/evaluation.json",
    }
    for local_path, key in uploads.items():
        if os.path.exists(local_path):
            s3.upload_file(local_path, config.S3_BUCKET, key)
            print(f"[register] s3://{config.S3_BUCKET}/{key}")

    with open(evaluation_path, encoding="utf-8") as f:
        evaluation = json.load(f)
    metrics = evaluation["metrics"]

    dynamodb = boto3.resource("dynamodb", region_name=config.AWS_REGION)
    ensure_table(dynamodb)
    table = dynamodb.Table(config.DYNAMODB_TABLE)

    item = {
        "model_name": config.MODEL_NAME,
        "version": version,
        "status": "Production",
        "s3_model_uri": f"s3://{config.S3_BUCKET}/{s3_prefix}/model.joblib",
        "s3_evaluation_uri": f"s3://{config.S3_BUCKET}/{s3_prefix}/evaluation.json",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", "local"),
        # DynamoDB no acepta float -> usar Decimal
        "f1": Decimal(str(round(metrics["f1"], 4))),
        "roc_auc": Decimal(str(round(metrics["roc_auc"], 4))),
        "accuracy": Decimal(str(round(metrics["accuracy"], 4))),
    }
    table.put_item(Item=item)

    print(f"[register] modelo registrado: {config.MODEL_NAME} version={version} status=Production")
    print(f"[register] f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Registra el modelo en S3 + DynamoDB")
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--metrics", default=config.METRICS_PATH)
    parser.add_argument("--evaluation", default=config.EVALUATION_PATH)
    args = parser.parse_args()
    register(args.model, args.metrics, args.evaluation)


if __name__ == "__main__":
    main()
