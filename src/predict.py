"""Pipeline de inferencia batch.

Lee un CSV de clientes, predice churn con el modelo del registry (o local) y
escribe un CSV con churn_prediction, churn_probability y risk_label.

Uso:
    PYTHONPATH=src python src/predict.py --input clientes.csv --output predicciones.csv
"""

import argparse
import os

import pandas as pd

import config
import inference


def run(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    missing = [c for c in config.ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de features en el input: {missing}")

    bundle = inference.load_model()
    results = inference.predict_frame(bundle["model"], df)

    out = df.copy()
    out["churn_prediction"] = [r["churn_prediction"] for r in results]
    out["churn_probability"] = [r["churn_probability"] for r in results]
    out["risk_label"] = [r["risk_label"] for r in results]
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"[predict] {len(out)} predicciones -> {output_path}")
    print(f"[predict] distribucion de riesgo: {out['risk_label'].value_counts().to_dict()}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia batch de churn")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/predictions.csv")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
