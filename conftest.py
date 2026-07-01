"""Configuracion de tests: hace src/ importable y entrena un modelo de prueba."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    """Corre el pipeline (preprocess -> feature store -> train -> evaluate) en un
    directorio temporal usando el dataset real, y devuelve la ruta del modelo."""
    import build_feature_store
    import config
    import evaluate
    import preprocess
    import train

    d = tmp_path_factory.mktemp("artifacts")
    config.CLEAN_DATA_PATH = str(d / "clean.csv")
    config.FEATURE_STORE_PATH = str(d / "features.csv")
    config.MODEL_DIR = str(d)
    config.MODEL_PATH = str(d / "model.joblib")
    config.METRICS_PATH = str(d / "metrics.json")
    config.TEST_DATA_PATH = str(d / "test.csv")
    config.EVALUATION_PATH = str(d / "evaluation.json")

    preprocess.preprocess(config.RAW_DATA_PATH, config.CLEAN_DATA_PATH)
    build_feature_store.build_feature_store(config.CLEAN_DATA_PATH, config.FEATURE_STORE_PATH)
    train.train(config.FEATURE_STORE_PATH, "logistic-regression", {"C": 1.0, "max_iter": 1000})
    evaluate.evaluate(
        config.MODEL_PATH, config.TEST_DATA_PATH, config.EVALUATION_PATH, min_f1=0.3, min_auc=0.5
    )
    return config.MODEL_PATH
