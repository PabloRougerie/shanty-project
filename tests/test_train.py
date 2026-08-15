"""Tests for scripts/train.py plumbing (no full LGBM training)."""

import json

import pandas as pd
from sklearn.dummy import DummyRegressor

import train
from vessel_tracker.features import build_long_dataset, model_feature_column_names


def test_build_calibration_frame_excludes_train_mmsi(mini_ais_df, monkeypatch):
    monkeypatch.setattr(train, "_lgbm_estimator", lambda: DummyRegressor())

    long_df = build_long_dataset(mini_ais_df, horizon_grid_minutes=[120])
    train_mmsi = {100001}
    feature_cols = model_feature_column_names(long_df.columns, include_h=True)

    calib = train._build_calibration_frame(long_df, train_mmsi, feature_cols)

    assert not calib["MMSI"].isin(train_mmsi).any()
    assert calib["MMSI"].nunique() >= 1
    assert "error_km" in calib.columns
    assert (calib["error_km"] >= 0).all()


def test_build_baseline_calibration_frame_excludes_train_mmsi(mini_ais_df):
    long_df = build_long_dataset(mini_ais_df, horizon_grid_minutes=[120])
    train_mmsi = {100001}

    calib = train._build_baseline_calibration_frame(long_df, train_mmsi)

    assert not calib["MMSI"].isin(train_mmsi).any()
    assert calib["MMSI"].nunique() >= 1
    assert "error_km" in calib.columns
    assert (calib["error_km"] >= 0).all()


def test_train_writes_metadata_structure(mini_ais_df, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "_lgbm_estimator", lambda: DummyRegressor())

    processed = tmp_path / "processed"
    models = tmp_path / "models"
    predictions = tmp_path / "predictions"
    processed.mkdir()
    models.mkdir()

    train_df = mini_ais_df[mini_ais_df["MMSI"] == 100001]
    val_df = mini_ais_df[mini_ais_df["MMSI"] == 100002]
    train_df.to_parquet(processed / "df_train.parquet")
    val_df.to_parquet(processed / "df_val.parquet")

    monkeypatch.setattr(train, "DATA_PROCESSED", processed)
    monkeypatch.setattr(train, "MODELS_DIR", models)
    monkeypatch.setattr(train, "PREDICTIONS_DIR", predictions)
    monkeypatch.setattr(train, "LGBM_FINAL_PATH", models / "lgbm_final.pkl")
    monkeypatch.setattr(train, "LGBM_FINAL_METADATA_PATH", models / "lgbm_final.metadata.json")
    monkeypatch.setattr(train, "R90_LOOKUP_LGBM_PATH", models / "r90_lookup_lgbm.parquet")
    monkeypatch.setattr(train, "R90_LOOKUP_BASELINE_PATH", models / "r90_lookup_baseline.parquet")
    monkeypatch.setattr(
        train, "CALIB_PREDICTIONS_LGBM_PATH", predictions / "calib_predictions_lgbm.parquet"
    )
    monkeypatch.setattr(
        train,
        "CALIB_PREDICTIONS_BASELINE_PATH",
        predictions / "calib_predictions_baseline.parquet",
    )

    train.train(force=True)

    metadata_path = models / "lgbm_final.metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())

    expected_types = {
        "feature_cols": list,
        "lookback_minutes": int,
        "horizon_grid_minutes": list,
        "random_seed": int,
        "resample_interval_min": int,
        "created_at": str,
        "n_trainval_rows": int,
        "n_trainval_vessels": int,
        "n_calibration_rows": int,
        "n_calibration_vessels": int,
        "fit_time_s": float,
        "artifacts": dict,
    }
    for key, typ in expected_types.items():
        assert key in metadata
        assert isinstance(metadata[key], typ)

    assert {
        "model",
        "r90_lookup_lgbm",
        "r90_lookup_baseline",
        "calib_predictions_lgbm",
        "calib_predictions_baseline",
    }.issubset(metadata["artifacts"])
    # test-set scoring moved to scripts/evaluate_test.py — must not leak in here
    assert "test_metrics_by_horizon" not in metadata
    assert "test_inference_time_s" not in metadata

    assert (models / "lgbm_final.pkl").exists()
    assert (models / "r90_lookup_lgbm.parquet").exists()
    assert (models / "r90_lookup_lgbm.meta.json").exists()
    assert (models / "r90_lookup_baseline.parquet").exists()
    assert (models / "r90_lookup_baseline.meta.json").exists()

    calib_lgbm_path = predictions / "calib_predictions_lgbm.parquet"
    calib_baseline_path = predictions / "calib_predictions_baseline.parquet"
    assert calib_lgbm_path.exists()
    assert calib_baseline_path.exists()

    expected_calib_cols = {
        "MMSI",
        "h",
        "LAT_lag_0",
        "LON_lag_0",
        "vx_true",
        "vy_true",
        "vx_pred",
        "vy_pred",
        "effective_speed_knots",
        "error_km",
    }
    calib_lgbm_df = pd.read_parquet(calib_lgbm_path)
    calib_baseline_df = pd.read_parquet(calib_baseline_path)
    assert expected_calib_cols.issubset(calib_lgbm_df.columns)
    assert expected_calib_cols.issubset(calib_baseline_df.columns)
