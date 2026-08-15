"""Tests for conditional radius calibration."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from vessel_tracker.calibration import (
    RadiusLookup,
    coverage_report,
    fit_conditional_radius,
)
from vessel_tracker.evaluation import attach_error_km, fit_model, predict_model
from vessel_tracker.features import build_long_dataset, model_feature_column_names


def _synthetic_calibration_frame(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    speeds = rng.uniform(5, 20, n_rows)
    horizons = rng.choice([120, 360, 720], n_rows)
    errors = rng.uniform(1, 30, n_rows) * (1 + speeds / 20)

    return pd.DataFrame(
        {
            "h": horizons,
            "effective_speed_knots": speeds,
            "error_km": errors,
        }
    )


def test_fit_conditional_radius_structure():
    calib = _synthetic_calibration_frame()
    lookup = fit_conditional_radius(calib, n_speed_bins=5)

    assert isinstance(lookup, RadiusLookup)
    assert lookup.percentile == pytest.approx(0.90)
    assert len(lookup.speed_bin_edges) >= 6
    assert {"speed_bin_index", "horizon_hours", "radius_km"}.issubset(lookup.lookup_df.columns)
    assert not lookup.marginal_df.empty


def test_coverage_report_columns():
    calib = _synthetic_calibration_frame(n_rows=300)
    test_df = _synthetic_calibration_frame(n_rows=150, seed=1)
    lookup = fit_conditional_radius(calib, n_speed_bins=5)

    report = coverage_report(test_df, lookup)
    assert {
        "speed_bin_index",
        "horizon_hours",
        "n_test",
        "r90_conditional_km",
        "coverage_conditional_pct",
        "r90_marginal_km",
        "coverage_marginal_pct",
    }.issubset(report.columns)
    assert len(report) > 0


def test_radius_lookup_roundtrip(tmp_path):
    calib = _synthetic_calibration_frame()
    lookup = fit_conditional_radius(calib, n_speed_bins=4)

    out = tmp_path / "r90_lookup.parquet"
    lookup.save(out)
    loaded = RadiusLookup.load(out)

    assert loaded.percentile == lookup.percentile
    pd.testing.assert_frame_equal(loaded.lookup_df, lookup.lookup_df)
    pd.testing.assert_series_equal(loaded.marginal_df, lookup.marginal_df)


def test_fit_conditional_radius_raises_without_error_km(mini_ais_df):
    long_df = build_long_dataset(mini_ais_df, horizon_grid_minutes=[120])

    with pytest.raises(ValueError, match="error_km"):
        fit_conditional_radius(long_df.drop(columns=["error_km"], errors="ignore"), n_speed_bins=3)


def test_fit_conditional_radius_via_evaluation_helpers(mini_ais_df):
    long_df = build_long_dataset(mini_ais_df, horizon_grid_minutes=[120])
    feature_cols = model_feature_column_names(long_df.columns, include_h=True)
    train_df = long_df.iloc[:20]
    val_df = long_df.iloc[20:40]

    fit_out = fit_model(train_df[feature_cols], train_df[["vx", "vy"]], DummyRegressor())
    pred_out = predict_model(fit_out["model"], val_df[feature_cols])

    calib = val_df.copy()
    calib["vx_pred"] = pred_out["vx_pred"]
    calib["vy_pred"] = pred_out["vy_pred"]
    calib["vx_true"] = calib["vx"]
    calib["vy_true"] = calib["vy"]
    calib = attach_error_km(calib)

    lookup = fit_conditional_radius(calib, n_speed_bins=3)
    assert lookup.lookup_df["radius_km"].notna().all()
