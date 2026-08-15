import time

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from vessel_tracker.metrics import haversine_distance


def fit_model(X_train, y_train, estimator) -> dict:
    """Fit estimator wrapped in MultiOutputRegressor.

    Returns:
        Dict with model (fitted MultiOutputRegressor) and fit_time_s.
    """
    model = MultiOutputRegressor(estimator=estimator)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time_s = time.perf_counter() - t0
    return {"model": model, "fit_time_s": fit_time_s}


def predict_model(model, X) -> dict:
    """Predict vx/vy with an already-fitted MultiOutputRegressor.

    Returns:
        Dict with vx_pred, vy_pred, inference_time_s.
    """
    t0 = time.perf_counter()
    y_pred = model.predict(X)
    inference_time_s = time.perf_counter() - t0
    return {
        "vx_pred": y_pred[:, 0],
        "vy_pred": y_pred[:, 1],
        "inference_time_s": inference_time_s,
    }


def fit_predict(X_train, y_train, X_pred, estimator) -> dict:
    """Fit on training data and predict vx/vy on X_pred.

    Returns:
        Dict with model, vx_pred, vy_pred, fit_time_s, inference_time_s.
    """
    fit_result = fit_model(X_train, y_train, estimator)
    pred_result = predict_model(fit_result["model"], X_pred)
    return {**fit_result, **pred_result}


def project_position(lat0, lon0, vx, vy, h_minutes):
    """Project (lat, lon) forward by constant velocity over h_minutes (deg/min)."""
    lat0 = np.asarray(lat0, dtype=float)
    lon0 = np.asarray(lon0, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    h = np.asarray(h_minutes, dtype=float)
    lat = lat0 + vy * h
    lon = lon0 + vx * h
    if lat.ndim == 0:
        return float(lat), float(lon)
    return lat, lon


def attach_error_km(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct positions from velocities and append haversine error_km."""
    lat_pred, lon_pred = project_position(
        oof_df["LAT_lag_0"], oof_df["LON_lag_0"], oof_df["vx_pred"], oof_df["vy_pred"], oof_df["h"]
    )
    lat_true, lon_true = project_position(
        oof_df["LAT_lag_0"], oof_df["LON_lag_0"], oof_df["vx_true"], oof_df["vy_true"], oof_df["h"]
    )
    oof_df = oof_df.copy()
    oof_df["error_km"] = haversine_distance(lat_true, lon_true, lat_pred, lon_pred)
    return oof_df
