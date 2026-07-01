"""Monitoreo simple del modelo en produccion.

Idea didactica: a partir de un CSV de predicciones calcula metricas operativas
(volumen, tasa de churn predicha, probabilidad promedio, distribucion de riesgo)
y las publica en CloudWatch + guarda un reporte JSON en S3.

Sirve para detectar data drift basico (ej: si de pronto el % HIGH se dispara) y
para tener dashboards/alarmas en CloudWatch.

Uso:
    PYTHONPATH=src python src/monitor.py --predictions reports/predictions.csv
    PYTHONPATH=src python src/monitor.py --predictions reports/predictions.csv --no-aws
"""

import argparse
import json
from datetime import UTC, datetime

import pandas as pd

import config


def compute_report(predictions_path: str) -> dict:
    df = pd.read_csv(predictions_path)
    total = len(df)
    risk_counts = df["risk_label"].value_counts().to_dict()
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_predictions": total,
        "predicted_churn_rate": round(float(df["churn_prediction"].mean()), 4),
        "avg_churn_probability": round(float(df["churn_probability"].mean()), 4),
        "risk_distribution": {
            "LOW": int(risk_counts.get("LOW", 0)),
            "MEDIUM": int(risk_counts.get("MEDIUM", 0)),
            "HIGH": int(risk_counts.get("HIGH", 0)),
        },
    }


def publish_cloudwatch(report: dict) -> None:
    import boto3

    cw = boto3.client("cloudwatch", region_name=config.AWS_REGION)
    cw.put_metric_data(
        Namespace=config.CLOUDWATCH_NAMESPACE,
        MetricData=[
            {"MetricName": "PredictedChurnRate", "Value": report["predicted_churn_rate"], "Unit": "None"},
            {"MetricName": "AvgChurnProbability", "Value": report["avg_churn_probability"], "Unit": "None"},
            {"MetricName": "HighRiskCustomers", "Value": report["risk_distribution"]["HIGH"], "Unit": "Count"},
            {"MetricName": "TotalPredictions", "Value": report["total_predictions"], "Unit": "Count"},
        ],
    )
    print(f"[monitor] metricas publicadas en CloudWatch namespace={config.CLOUDWATCH_NAMESPACE}")


def upload_report(report: dict) -> None:
    import boto3

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    key = f"{config.S3_REPORTS_PREFIX}/monitoring-{stamp}.json"
    boto3.client("s3", region_name=config.AWS_REGION).put_object(
        Bucket=config.S3_BUCKET,
        Key=key,
        Body=json.dumps(report, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"[monitor] reporte -> s3://{config.S3_BUCKET}/{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoreo simple de churn")
    parser.add_argument("--predictions", default="reports/predictions.csv")
    parser.add_argument("--no-aws", action="store_true", help="Solo imprime, sin CloudWatch/S3")
    args = parser.parse_args()

    report = compute_report(args.predictions)
    print(json.dumps(report, indent=2))

    if not args.no_aws:
        publish_cloudwatch(report)
        upload_report(report)


if __name__ == "__main__":
    main()
