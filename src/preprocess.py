"""Paso 1 del pipeline: limpiar el dataset crudo.

Que hace:
  - Lee el CSV crudo.
  - Elimina filas duplicadas y con nulos en columnas clave.
  - Convierte el target churn de Yes/No a 1/0.
  - Mantiene customer_id como identificador (lo usa el feature store).
  - Escribe un CSV limpio.

Uso:
    python src/preprocess.py
    python src/preprocess.py --input data/raw/telco_customer_churn_mlops.csv \
                             --output data/processed/clean.csv
"""

import argparse
import os

import pandas as pd

import config


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    expected = [config.ID_COLUMN, *config.ALL_FEATURES, config.TARGET_COLUMN]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")

    df = df.drop_duplicates()
    df = df.dropna(subset=expected)
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=[config.TARGET_COLUMN])
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(int)

    df = df[expected]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[preprocess] {len(df)} filas limpias -> {output_path}")
    print(f"[preprocess] distribucion churn: {df[config.TARGET_COLUMN].value_counts().to_dict()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpia el dataset crudo de churn")
    parser.add_argument("--input", default=config.RAW_DATA_PATH)
    parser.add_argument("--output", default=config.CLEAN_DATA_PATH)
    args = parser.parse_args()
    preprocess(args.input, args.output)


if __name__ == "__main__":
    main()
