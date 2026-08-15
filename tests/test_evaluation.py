"""Smoke tests for v3 evaluation pipeline."""

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from vessel_tracker.evaluation import (
    attach_error_km,
    fit_model,
    fit_predict,
    predict_model,
    project_position,
)
from vessel_tracker.features import build_long_dataset, model_feature_column_names
from vessel_tracker.metrics import compute_metrics


def test_fit_predict_returns_velocities(mini_ais_df):
    long_df = build_long_dataset(mini_ais_df, horizon_grid_minutes=[120])
    feature_cols = model_feature_column_names(long_df.columns, include_h=True)
    X = long_df[feature_cols]
    y = long_df[["vx", "vy"]]

    result = fit_predict(X.iloc[:20], y.iloc[:20], X.iloc[20:30], DummyRegressor())
    assert len(result["vx_pred"]) == 10
    assert len(result["vy_pred"]) == 10
    assert result["fit_time_s"] >= 0
    assert result["inference_time_s"] >= 0
    assert "model" in result


def test_fit_model_returns_fitted_predictor(mini_ais_df):
    long_df = build_long_dataset(mini_ais_df, horizon_grid_minutes=[120])
    feature_cols = model_feature_column_names(long_df.columns, include_h=True)
    X = long_df[feature_cols]
    y = long_df[["vx", "vy"]]

    result = fit_model(X.iloc[:20], y.iloc[:20], DummyRegressor())
    assert "model" in result
    assert result["fit_time_s"] >= 0

    y_pred = result["model"].predict(X.iloc[20:30])
    assert y_pred.shape == (10, 2)


def test_predict_model_matches_fit_predict(mini_ais_df):
    long_df = build_long_dataset(mini_ais_df, horizon_grid_minutes=[120])
    feature_cols = model_feature_column_names(long_df.columns, include_h=True)
    X_train = long_df[feature_cols].iloc[:20]
    y_train = long_df[["vx", "vy"]].iloc[:20]
    X_pred = long_df[feature_cols].iloc[20:30]

    estimator = DummyRegressor(strategy="mean")
    one_shot = fit_predict(X_train, y_train, X_pred, estimator)
    fit_out = fit_model(X_train, y_train, estimator)
    pred_out = predict_model(fit_out["model"], X_pred)

    np.testing.assert_allclose(one_shot["vx_pred"], pred_out["vx_pred"])
    np.testing.assert_allclose(one_shot["vy_pred"], pred_out["vy_pred"])
    np.testing.assert_allclose(
        one_shot["model"].predict(X_pred),
        np.column_stack([pred_out["vx_pred"], pred_out["vy_pred"]]),
    )


def test_attach_error_km_adds_column(mini_ais_df):
    long_df = build_long_dataset(mini_ais_df, horizon_grid_minutes=[120])
    row = long_df.iloc[:1].copy()
    row = row.rename(columns={"vx": "vx_true", "vy": "vy_true"})
    row["vx_pred"] = row["vx_true"]
    row["vy_pred"] = row["vy_true"]

    out = attach_error_km(row)

    assert "error_km" in out.columns
    assert out["error_km"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_project_position_constant_velocity():
    lat, lon = project_position(10.0, -90.0, vx=0.01, vy=0.02, h_minutes=120)
    assert lat == pytest.approx(10.0 + 0.02 * 120)
    assert lon == pytest.approx(-90.0 + 0.01 * 120)


def test_compute_metrics_from_oof(mini_ais_df):
    long_df = build_long_dataset(mini_ais_df, horizon_grid_minutes=[120, 360])
    row = long_df.iloc[:20].copy()
    row = row.rename(columns={"vx": "vx_true", "vy": "vy_true"})
    row["model_name"] = "dummy"
    row["vx_pred"] = row["vx_true"]
    row["vy_pred"] = row["vy_true"]
    oof_df = attach_error_km(row)

    summary = compute_metrics(oof_df)

    assert set(summary.columns) == {
        "model_name",
        "h",
        "mae_mean",
        "r50_km",
        "r80_km",
        "r90_km",
        "r95_km",
    }
    assert summary["h"].isin([120, 360]).all()
    assert (summary["r90_km"] >= summary["r50_km"]).all()
