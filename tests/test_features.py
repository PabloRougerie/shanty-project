"""Tests for v3 feature engineering (NB07)."""

import numpy as np
import pytest

from vessel_tracker.config import settings
from vessel_tracker.features import (
    build_long_dataset,
    create_dx_dy,
    create_effective_speed,
    create_lag_features,
    create_path_and_straightness,
    create_target,
    model_feature_column_names,
    path_length_km,
    straightness_index,
)


def test_model_feature_column_names(mini_ais_df):
    long_df = build_long_dataset(mini_ais_df, horizon_grid_minutes=[120])
    cols = model_feature_column_names(long_df.columns, include_h=True)

    # lags (read back from the frame) + geometry/dx/dy (settings) + h
    lag_cols = [c for c in cols if c.startswith(("LAT_lag_", "LON_lag_"))]
    assert lag_cols == [c for c in long_df.columns if c.startswith(("LAT_lag_", "LON_lag_"))]
    assert cols[-6:] == ["Length", "Width", "Draft", "dx", "dy", "h"]
    assert len(cols) == 18

    # include_h=False drops only the trailing h
    assert model_feature_column_names(long_df.columns, include_h=False) == cols[:-1]


def test_create_lag_features_single_vessel(mini_ais_df):
    vessel = mini_ais_df[mini_ais_df["MMSI"] == 100001].copy()
    out = create_lag_features(vessel, settings.nb_lags, settings.lookback_minutes)
    assert "LAT" not in out.columns
    assert "LAT_lag_0" in out.columns
    assert f"LAT_lag_{settings.lookback_steps()}" in out.columns
    assert len(out) > 0


def test_create_dx_dy_from_lags(mini_ais_df):
    vessel = mini_ais_df[mini_ais_df["MMSI"] == 100001].copy()
    lagged = create_lag_features(vessel, settings.nb_lags, settings.lookback_minutes)
    out = create_dx_dy(lagged, settings.lookback_minutes)
    row = out.iloc[0]
    lookback = settings.lookback_steps()
    assert row["dx"] == pytest.approx(row["LON_lag_0"] - row[f"LON_lag_{lookback}"])
    assert row["dy"] == pytest.approx(row["LAT_lag_0"] - row[f"LAT_lag_{lookback}"])


def test_create_target_velocity_grid(mini_ais_df):
    vessel = mini_ais_df[mini_ais_df["MMSI"] == 100001].copy()
    lagged = create_lag_features(vessel, settings.nb_lags, settings.lookback_minutes)
    lagged = create_dx_dy(lagged, settings.lookback_minutes)
    out = create_target(lagged, horizon_grid_minutes=[120])

    assert {"vx", "vy", "h"}.issubset(out.columns)
    assert (out["h"] == 120).all()

    # Hand-check first row: constant velocity track (+0.01 deg / 10 min per step)
    row0 = out.iloc[0]
    h_steps = 120 // settings.resample_interval_min
    lat0 = row0["LAT_lag_0"]
    lon0 = row0["LON_lag_0"]
    expected_vy = (vessel.iloc[settings.lookback_steps() + h_steps]["LAT"] - lat0) / 120
    expected_vx = (vessel.iloc[settings.lookback_steps() + h_steps]["LON"] - lon0) / 120
    assert row0["vy"] == pytest.approx(expected_vy)
    assert row0["vx"] == pytest.approx(expected_vx)


def test_build_long_dataset_smoke(mini_ais_df):
    out = build_long_dataset(mini_ais_df, horizon_grid_minutes=[120, 360])
    assert len(out) > 0
    assert out["h"].isin([120, 360]).all()
    for col in model_feature_column_names(out.columns, include_h=True):
        assert col in out.columns


def test_create_effective_speed_straight_track(mini_ais_df):
    vessel = mini_ais_df[mini_ais_df["MMSI"] == 100001].copy()
    lookback = settings.lookback_minutes
    warm_up = settings.lookback_steps()

    out = create_effective_speed(vessel, lookback_minutes=lookback)

    assert "effective_speed_knots" in out.columns
    assert out["effective_speed_knots"].iloc[:warm_up].isna().all()
    after = out["effective_speed_knots"].iloc[warm_up:]
    assert after.notna().all()
    assert (after > 0).all()
    # Constant diagonal track (+0.01 deg / 10 min): speed should be modest, not absurd.
    assert after.between(1.0, 30.0).all()


def test_create_path_and_straightness_straight_track(mini_ais_df):
    vessel = mini_ais_df[mini_ais_df["MMSI"] == 100001].copy()
    lookback = settings.lookback_minutes
    warm_up = settings.lookback_steps()

    out = create_path_and_straightness(vessel, lookback_minutes=lookback)

    assert {"path_length_km", "straightness"}.issubset(out.columns)
    assert out["path_length_km"].iloc[: warm_up - 1].isna().all()
    after_path = out["path_length_km"].iloc[warm_up - 1 :]
    after_str = out["straightness"].iloc[warm_up:]
    assert (after_path > 0).all()
    assert after_str.notna().all()
    np.testing.assert_allclose(after_str.to_numpy(), 1.0, atol=1e-3)


def test_path_and_straightness_helpers():
    lat = np.array([0.0, 0.01, 0.02])
    lon = np.array([0.0, 0.01, 0.02])
    assert path_length_km(lat, lon) > 0
    assert straightness_index(lat, lon) == pytest.approx(1.0, abs=1e-6)
