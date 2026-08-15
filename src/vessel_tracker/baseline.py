import time

import pandas as pd

from vessel_tracker.config import settings


def predict_baseline(X_pred: pd.DataFrame, lookback_minutes: int | None = None) -> dict:
    """Constant-velocity baseline velocity (vx, vy) in deg/min from lag features.

    Args:
        X_pred: Rows with LAT_lag_0/LON_lag_0 and oldest lookback lag columns.
        lookback_minutes: Lookback width in minutes; defaults to settings.lookback_minutes.

    Returns:
        Dict with vx_pred, vy_pred (deg/min) and inference_time_s.
    """
    if lookback_minutes is None:
        lookback_minutes = settings.lookback_minutes

    lookback_steps = lookback_minutes // settings.resample_interval_min
    required_cols = [
        "LAT_lag_0",
        "LON_lag_0",
        f"LAT_lag_{lookback_steps}",
        f"LON_lag_{lookback_steps}",
    ]
    missing = [col for col in required_cols if col not in X_pred.columns]
    if missing:
        raise ValueError(f"Missing required columns in X_pred: {missing}")

    t0 = time.perf_counter()

    lat_now = X_pred["LAT_lag_0"]
    lon_now = X_pred["LON_lag_0"]
    lat_old = X_pred[f"LAT_lag_{lookback_steps}"]
    lon_old = X_pred[f"LON_lag_{lookback_steps}"]

    # deg/min: same vx/vy convention as create_target and the LGBM model.
    vy_pred = (lat_now - lat_old) / lookback_minutes
    vx_pred = (lon_now - lon_old) / lookback_minutes

    return {
        "vx_pred": vx_pred.to_numpy(),
        "vy_pred": vy_pred.to_numpy(),
        "inference_time_s": time.perf_counter() - t0,
    }
