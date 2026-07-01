"""Tests del pipeline de datos y entrenamiento."""

import joblib
import pandas as pd

import build_feature_store
import config
import preprocess
import train


def test_preprocess_cleans_and_encodes(tmp_path):
    out = tmp_path / "clean.csv"
    df = preprocess.preprocess(config.RAW_DATA_PATH, str(out))
    assert out.exists()
    # target binarizado
    assert set(df[config.TARGET_COLUMN].unique()) <= {0, 1}
    # columnas esperadas
    for col in [config.ID_COLUMN, *config.ALL_FEATURES, config.TARGET_COLUMN]:
        assert col in df.columns


def test_feature_store_has_event_timestamp(tmp_path):
    clean = tmp_path / "clean.csv"
    preprocess.preprocess(config.RAW_DATA_PATH, str(clean))
    fs = tmp_path / "features.csv"
    df = build_feature_store.build_feature_store(str(clean), str(fs))
    assert "event_timestamp" in df.columns
    assert len(df) > 0


def test_build_pipeline_metrics_in_range():
    metrics = train.compute_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.2, 0.8, 0.4, 0.1])
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_trained_model_bundle(trained_model):
    bundle = joblib.load(trained_model)
    assert "model" in bundle
    assert bundle["feature_columns"] == config.ALL_FEATURES
    # el modelo predice sobre un cliente de ejemplo
    sample = pd.DataFrame([{
        "gender": "Male", "contract_type": "Month-to-Month", "internet_service": "Fiber",
        "senior_citizen": 0, "tenure_months": 12, "monthly_charges": 70.5,
        "support_tickets_last_6m": 2,
    }])
    proba = bundle["model"].predict_proba(sample[config.ALL_FEATURES])
    assert proba.shape == (1, 2)
