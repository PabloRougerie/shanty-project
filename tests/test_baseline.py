"""Tests for v3 constant-velocity baseline (predict_baseline)."""

import pandas as pd
import pytest

from vessel_tracker.baseline import predict_baseline
from vessel_tracker.config import settings


def _lag_feature_row() -> pd.DataFrame:
    lookback_steps = settings.lookback_steps()
    return pd.DataFrame(
        {
            "LAT_lag_0": [48.1],
            "LON_lag_0": [-5.1],
            f"LAT_lag_{lookback_steps}": [48.0],
            f"LON_lag_{lookback_steps}": [-5.2],
        }
    )


def test_predict_baseline_velocity_deg_per_min():
    df = _lag_feature_row()
    out = predict_baseline(df)
    lookback_minutes = settings.lookback_minutes
    assert out["vy_pred"][0] == (48.1 - 48.0) / lookback_minutes
    assert out["vx_pred"][0] == (-5.1 - (-5.2)) / lookback_minutes


def test_predict_baseline_constant_velocity_track():
    lookback_steps = settings.lookback_steps()
    i = lookback_steps
    df = pd.DataFrame(
        {
            "LAT_lag_0": [48.0 + 0.01 * i],
            "LON_lag_0": [-5.0 + 0.01 * i],
            f"LAT_lag_{lookback_steps}": [48.0],
            f"LON_lag_{lookback_steps}": [-5.0],
        }
    )
    out = predict_baseline(df)
    expected = 0.01 * lookback_steps / settings.lookback_minutes
    assert out["vy_pred"][0] == pytest.approx(expected)
    assert out["vx_pred"][0] == pytest.approx(expected)
