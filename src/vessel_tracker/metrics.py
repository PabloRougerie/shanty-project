import numpy as np
import pandas as pd


def haversine_distance(LAT_true, LON_true, LAT_pred, LON_pred):
    """Great-circle distance between two points in kilometers."""
    earth_radius = 6371

    LAT_true_rad = np.radians(LAT_true)
    LON_true_rad = np.radians(LON_true)
    LAT_pred_rad = np.radians(LAT_pred)
    LON_pred_rad = np.radians(LON_pred)

    d_LAT = LAT_pred_rad - LAT_true_rad
    d_LON = LON_pred_rad - LON_true_rad

    a = (
        np.sin(d_LAT / 2.0) ** 2
        + np.cos(LAT_true_rad) * np.cos(LAT_pred_rad) * np.sin(d_LON / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c * earth_radius


def compute_metrics(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-row haversine errors into MAE and R50–R95 by model and horizon.

    Args:
        oof_df: Rows with model_name, h, error_km columns.

    Returns:
        One row per (model_name, h) with mae_mean and r50_km..r95_km.
    """
    rows = []
    for (name, h), horizon_subset in oof_df.groupby(["model_name", "h"]):
        r50, r80, r90, r95 = np.percentile(horizon_subset["error_km"], [50, 80, 90, 95])
        rows.append(
            {
                "model_name": name,
                "h": h,
                "mae_mean": float(horizon_subset["error_km"].mean()),
                "r50_km": float(r50),
                "r80_km": float(r80),
                "r90_km": float(r90),
                "r95_km": float(r95),
            }
        )

    return pd.DataFrame(
        rows, columns=["model_name", "h", "mae_mean", "r50_km", "r80_km", "r90_km", "r95_km"]
    )
