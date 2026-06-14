"""Paso 2 del pipeline: construir el feature store offline (CSV).

En este proyecto el "feature store offline" es simplemente un CSV versionado en S3.
No usamos SageMaker Feature Store porque la cuenta tiene cuotas restringidas y para
una clase un CSV en S3 cumple el mismo rol didactico: una tabla de features lista
para entrenar e inferir, con identificador y marca de tiempo de ingesta.

Que hace:
  - Lee el CSV limpio.
  - Selecciona las features del modelo + customer_id + target.
  - Agrega event_timestamp (cuando se materializaron las features).
  - Escribe el feature store como CSV.

El buildspec sube este CSV a s3://<bucket>/feature_store/features.csv

Uso:
    python src/build_feature_store.py
"""

import argparse
import os
from datetime import UTC, datetime

import pandas as pd

import config


def build_feature_store(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    columns = [config.ID_COLUMN, *config.ALL_FEATURES, config.TARGET_COLUMN]
    features = df[columns].copy()
    features["event_timestamp"] = datetime.now(UTC).isoformat()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"[feature_store] {len(features)} registros, {len(config.ALL_FEATURES)} features -> {output_path}")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el feature store offline en CSV")
    parser.add_argument("--input", default=config.CLEAN_DATA_PATH)
    parser.add_argument("--output", default=config.FEATURE_STORE_PATH)
    args = parser.parse_args()
    build_feature_store(args.input, args.output)


if __name__ == "__main__":
    main()
