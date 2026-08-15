"""Feature engineering for v3 LGBM pipeline (NB06–07)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vessel_tracker.config import settings
from vessel_tracker.metrics import haversine_distance


def lag_step_indices(nb_lags: int | None = None, lookback_minutes: int | None = None) -> np.ndarray:
    """Compute evenly spaced lag step indices for a lookback window.

    Args:
        nb_lags: Lag count; defaults to settings.nb_lags.
        lookback_minutes: Lookback in minutes; defaults to settings.lookback_minutes.

    Returns:
        1D int array of step indices from 0 (present) to lookback_steps (oldest).
    """
    nb = settings.nb_lags if nb_lags is None else nb_lags
    lookback_steps = settings.lookback_steps(
        lookback_minutes
    )  # convert minute in number of time steps
    available_lags = lookback_steps + 1
    if not (2 <= nb <= available_lags):
        raise ValueError(
            f"nb_of_lags={nb} does not fit lookback_steps={lookback_steps}: "
            f"expected 2..{available_lags}."
        )
    # linspace then round can collapse distinct indices; fail if count drops below nb_lags.
    lags = np.unique(np.linspace(0, lookback_steps, nb).round().astype(int))
    if len(lags) != nb:
        raise ValueError(
            f"rounding over a {lookback_steps}-step window reduced the number of lags: "
            f"asked {nb}, got {len(lags)}"
        )
    return lags


def model_feature_column_names(columns, include_h: bool = False) -> list[str]:
    """Select and order the model input columns from an existing feature frame.

    Lag columns are read back from the actual columns produced by create_lag_features,
    then concatenated with settings.feature_columns
    (vessel geometry + dx/dy) and, optionally, the horizon feature h.

    Args:
        columns: Iterable of column names (e.g. df.columns) from a built feature frame.
        include_h: If True, append horizon feature h (minutes).

    Returns:
        Ordered list of model input column names.
    """
    lag_cols = [c for c in columns if c.startswith(("LAT_lag_", "LON_lag_"))]
    return lag_cols + list(settings.feature_columns) + (["h"] if include_h else [])


def create_lag_features(df, nb_lags: int, lookback_minutes: int):
    """Add evenly spaced LAT/LON lag columns per vessel; drop rows without full lookback.

    Args:
        df: Raw AIS rows with MMSI, LAT, LON (degrees), sorted by vessel/time.
        nb_lags: Number of lag indices (including 0 = present).
        lookback_minutes: Lookback window in minutes; converted to resample steps internally.

    Returns:
        Same rows minus warm-up; LAT/LON replaced by LAT_lag_i / LON_lag_i (i = step index).
    """
    lags = lag_step_indices(nb_lags, lookback_minutes)

    vessels = df.groupby("MMSI")
    # Lag index i = i resample steps back; lag 0 is the present ping.
    lag_col = {
        f"{coord}_lag_{lag}": vessels[coord].shift(lag) for lag in lags for coord in ("LAT", "LON")
    }
    # Raw LAT/LON removed so downstream features use lags only (see create_dx_dy).
    return df.assign(**lag_col).drop(columns=["LAT", "LON"]).dropna(axis=0)


def create_dx_dy(df, lookback_minutes: int):
    """Net east/north displacement over the lookback window, in degrees.

    Args:
        df: Rows with LAT_lag_0, LON_lag_0 and LAT_lag_{lookback_steps}, LON_lag_{lookback_steps}.
        lookback_minutes: Lookback width in minutes (must match create_lag_features).

    Returns:
        Input df with dx (LON delta) and dy (LAT delta) columns added.
    """
    lookback_steps = lookback_minutes // settings.resample_interval_min
    required_cols = [
        "LAT_lag_0",
        "LON_lag_0",
        f"LAT_lag_{lookback_steps}",
        f"LON_lag_{lookback_steps}",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    # Requires LAT_lag_0/LON_lag_0 and LAT_lag_{lookback_steps}/LON_lag_{lookback_steps},
    # produced by create_lag_features.
    return df.assign(
        dx=df["LON_lag_0"] - df[f"LON_lag_{lookback_steps}"],
        dy=df["LAT_lag_0"] - df[f"LAT_lag_{lookback_steps}"],
    )


def create_effective_speed(df, lookback_minutes: int):
    """Mean speed along the actual track over the lookback window, in knots.

    Args:
        df: Per-vessel track with raw LAT, LON (degrees); sorted by MMSI, BaseDateTime.
        lookback_minutes: Rolling window width in minutes.

    Returns:
        Input df with effective_speed_knots; first lookback window per MMSI is NaN.
    """
    lookback_steps = lookback_minutes // settings.resample_interval_min

    vessels = df.groupby("MMSI")
    lat_prev = vessels["LAT"].shift(1)
    lon_prev = vessels["LON"].shift(1)
    step_km = haversine_distance(lat_prev, lon_prev, df["LAT"], df["LON"])

    # step_km is a flat Series; re-groupby MMSI so rolling stays vessel-local after shift.
    path_km = (
        step_km.groupby(df["MMSI"], sort=False)
        .rolling(lookback_steps, min_periods=lookback_steps)
        .sum()
        .reset_index(level=0, drop=True)  # drop MMSI from MultiIndex, keep row alignment
    )
    # km -> NM (/1.852), then NM/h -> knots (window duration in hours).
    effective_speed_knots = (path_km / 1.852) / (lookback_minutes / 60)
    return df.assign(effective_speed_knots=effective_speed_knots)


def create_path_and_straightness(df, lookback_minutes: int):
    """Path length and straightness over the lookback window (diagnostic, not model inputs).

    Args:
        df: Per-vessel track with raw LAT, LON (degrees).
        lookback_minutes: Window width in minutes.

    Returns:
        Input df with path_length_km (km) and straightness (net/path, NaN if path is zero).
    """
    steps = lookback_minutes // settings.resample_interval_min
    grouped = df.groupby("MMSI", sort=False)

    lat_prev = grouped["LAT"].shift(1)
    lon_prev = grouped["LON"].shift(1)
    step_km = haversine_distance(
        lat_prev.to_numpy(),
        lon_prev.to_numpy(),
        df["LAT"].to_numpy(),
        df["LON"].to_numpy(),
    )
    step_km = pd.Series(step_km, index=df.index).fillna(0.0)  # first ping per MMSI has no segment

    path_length_km = (
        step_km.groupby(df["MMSI"], sort=False)
        .rolling(steps, min_periods=steps)
        .sum()
        .reset_index(level=0, drop=True)  # restore flat index aligned to df rows
    )

    lat_start = grouped["LAT"].shift(steps)
    lon_start = grouped["LON"].shift(steps)
    net_km = haversine_distance(
        lat_start.to_numpy(),
        lon_start.to_numpy(),
        df["LAT"].to_numpy(),
        df["LON"].to_numpy(),
    )

    # straightness = net displacement / path; skip divide where path_length_km == 0.
    straightness = np.divide(
        net_km,
        path_length_km.to_numpy(),
        out=np.full(len(df), np.nan),
        where=path_length_km.to_numpy() > 0,
    )
    return df.assign(path_length_km=path_length_km, straightness=straightness)


def create_target(df: pd.DataFrame, horizon_grid_minutes: list[int]):
    """Expand wide rows into long-format velocity targets vx/vy for multiple horizons.

    Args:
        df: Wide rows with LAT_lag_0, LON_lag_0 (present position in degrees).
        horizon_grid_minutes: Horizons in minutes; deduplicated and snapped to resample grid.

    Returns:
        Long df: one row per (observation, horizon) with vx, vy (deg/min) and h (minutes, int16).
    """
    horizons = np.asarray(horizon_grid_minutes)
    # Convert horizons in minutes to steps
    h_steps = np.unique(np.round(horizons / settings.resample_interval_min)).astype(int)
    h_steps = h_steps[h_steps > 0]
    if len(h_steps) == 0:
        raise ValueError("no prediction horizon > 0")

    required_cols = ["MMSI", "BaseDateTime", "LAT_lag_0", "LON_lag_0"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(by=["MMSI", "BaseDateTime"]).reset_index(drop=True)

    lat0 = df["LAT_lag_0"].to_numpy()
    lon0 = df["LON_lag_0"].to_numpy()
    grp = df.groupby("MMSI")

    # create the dataframe parts specific ton a given horizon (target + h)
    target_parts = []
    for h_step in h_steps:
        lat_fut = grp["LAT_lag_0"].shift(-h_step).to_numpy()
        lon_fut = grp["LON_lag_0"].shift(-h_step).to_numpy()
        dt = int(h_step * settings.resample_interval_min)
        target_parts.append(
            pd.DataFrame(
                {
                    "row_id": np.arange(len(df), dtype=np.int64),
                    # vx/vy in deg/min (model target unit).
                    "vx": (lon_fut - lon0) / dt,
                    "vy": (lat_fut - lat0) / dt,
                    "h": np.int16(dt),
                }
            )
        )

    targets = pd.concat(target_parts, ignore_index=True).dropna(subset=["vx", "vy"])
    # Reattach feature rows by position index (targets may be shorter after dropna).

    df_out = df.iloc[targets["row_id"].to_numpy()].reset_index(
        drop=True
    )  # creates as many duplicates as row_ids is duplicated
    df_out["vx"] = targets["vx"].to_numpy()
    df_out["vy"] = targets["vy"].to_numpy()
    df_out["h"] = targets["h"].to_numpy()
    return df_out


def build_long_dataset(
    df: pd.DataFrame,
    *,
    horizon_grid_minutes: list[int] | None = None,
) -> pd.DataFrame:
    """Run the retained NB06/NB07 feature pipeline and return a long training dataset.

    Args:
        df: Clean resampled AIS (multiple vessels).
        horizon_grid_minutes: Target horizons in minutes; defaults to settings.horizon_grid_minutes.

    Returns:
        Long df with model features, vx/vy targets, and h; rows with any NaN dropped.
        Also includes effective_speed_knots, path_length_km, straightness — unused by the
        model, needed downstream for R90 calibration (see calibration.fit_conditional_radius).
    """
    # defines the horizon prediction used for training (feature h and target creation)
    if horizon_grid_minutes is None:
        horizon_grid_minutes = settings.horizon_grid_minutes

    out = df.sort_values(["MMSI", "BaseDateTime"]).copy()
    lookback_minutes = settings.lookback_minutes

    # kinetic and trajectory metrics need raw LAT/LON;
    # must run before create_lag_features drops them.
    out = create_effective_speed(out, lookback_minutes=lookback_minutes)
    out = create_path_and_straightness(out, lookback_minutes=lookback_minutes)

    # Order fixed: lags drop raw coords -> dx/dy from lags -> target from lag_0 positions.
    out = create_lag_features(out, nb_lags=settings.nb_lags, lookback_minutes=lookback_minutes)
    out = create_dx_dy(out, lookback_minutes=lookback_minutes)

    # create target: this is where the long format take shape:
    # duplicates along axis0 for all the horizons
    # add horizon in minutes
    out = create_target(out, horizon_grid_minutes=horizon_grid_minutes)
    return out.dropna(axis=0).reset_index(drop=True)


def path_length_km(lat: np.ndarray, lon: np.ndarray) -> float:
    """Sum of haversine distances between consecutive points (km)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if len(lat) < 2:
        return 0.0
    return float(np.sum(haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])))


def straightness_index(lat, lon) -> float:
    """Direct distance / path length in [0, 1]. NaN if path length is zero."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if len(lat) < 2:
        return np.nan
    path_len = path_length_km(lat, lon)
    if path_len == 0:
        return np.nan
    direct = haversine_distance(lat[0], lon[0], lat[-1], lon[-1])
    return float(min(direct / path_len, 1.0))
