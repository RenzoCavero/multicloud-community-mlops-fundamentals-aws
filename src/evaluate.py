"""Paso 4 del pipeline: evaluar el modelo entrenado.

Que hace:
  - Carga el modelo (bundle joblib) y el set de test guardado por train.py.
  - Calcula metricas sobre test.
  - Aplica un umbral minimo de calidad (gate): si no se cumple, falla el build.
  - Escribe evaluation.json con metricas + decision del gate.

El gate es lo que evita registrar modelos malos en produccion.

Uso:
    python src/evaluate.py --min-f1 0.5 --min-auc 0.6
"""

import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config


def evaluate(model_path: str, test_path: str, output_path: str, min_f1: float, min_auc: float) -> dict:
    bundle = joblib.load(model_path)
    model = bundle["model"] if isinstance(bundle, dict) else bundle

    df = pd.read_csv(test_path)
    X = df[config.ALL_FEATURES]
    y_true = df[config.TARGET_COLUMN]

    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.0,
    }

    passed = metrics["f1"] >= min_f1 and metrics["roc_auc"] >= min_auc
    payload = {
        "metrics": metrics,
        "quality_gate": {
            "min_f1": min_f1,
            "min_auc": min_auc,
            "passed": passed,
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[evaluate] metricas: {metrics}")
    print(f"[evaluate] quality gate passed={passed} (min_f1={min_f1}, min_auc={min_auc})")

    if not passed:
        raise SystemExit(
            f"[evaluate] FALLO el quality gate: f1={metrics['f1']:.3f} auc={metrics['roc_auc']:.3f}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua el modelo y aplica el quality gate")
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--test", default=config.TEST_DATA_PATH)
    parser.add_argument("--output", default=config.EVALUATION_PATH)
    parser.add_argument("--min-f1", type=float, default=0.5)
    parser.add_argument("--min-auc", type=float, default=0.6)
    args = parser.parse_args()
    evaluate(args.model, args.test, args.output, args.min_f1, args.min_auc)


if __name__ == "__main__":
    main()
