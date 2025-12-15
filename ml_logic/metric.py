import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer


def haversine_distance(LAT_true, LON_true, LAT_pred, LON_pred):
    """
    Calculate great-circle distance between two points Haversine formula.

    Parameters:
    LAT_true, LON_true : array. True latitude and longitude coordinates in degrees
    LAT_pred, LON_pred : array. Predicted latitude and longitude coordinates in degrees

    Returns:
    float: Great-circle distance in kilometers
    """
    earth_radius = 6371  # Earth mean radius in kilometers

    # Convert degrees to radians
    LAT_true_rad = np.radians(LAT_true)
    LON_true_rad = np.radians(LON_true)
    LAT_pred_rad = np.radians(LAT_pred)
    LON_pred_rad = np.radians(LON_pred)

    # Calculate differences
    d_LAT = LAT_pred_rad - LAT_true_rad
    d_LON = LON_pred_rad - LON_true_rad

    # Haversine formula
    a = (np.sin(d_LAT / 2.0)**2 + np.cos(LAT_true_rad)*np.cos(LAT_pred_rad)*np.sin(d_LON/2.0)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))  # Angular distance in radians
    d = c * earth_radius  # Convert to kilometers
    return d


def haversine_mae(y_true, y_pred):
    """
    Calculate MAE using Haversine distance between positions.
    Evaluates prediction quality in km rather than degrees.

    Parameters:

    y_true : array, shape (n_samples, 2)
    True positions where column 0 = LAT, column 1 = LON (in degrees)
    y_pred : array, shape (n_samples, 2)
    Predicted positions where column 0 = LAT, column 1 = LON (in degrees)

    Returns:
    float: Mean Absolute Error in kilometers
    """

    #make sure we're converting in numpy
    y_true= np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Calculate haversine distance for each prediction
    mae = np.mean(abs(haversine_distance(y_true[:,0],
                                         y_true[:,1],
                                         y_pred[:,0],
                                         y_pred[:,1]) ))
    return mae

# Create sklearn-compatible scorer
haversine_scorer = make_scorer(haversine_mae, greater_is_better= False)


def position_extrapolation(df: pd.DataFrame, time_horizon):
    """
    Naive baseline: extrapolate position at time t+ time_horizon based on
    linear displacement between time t - time_horizon and t.

    Assumes constant velocity: future_displacement = past_displacement
    Mathematically: position(t+time horizon) = position(t) + [position(t) - position(t-time horizon)]

    Parameters:
    -----------
    df : DataFrame
        Must contain columns: LAT, LON, "LAT_lag_{time_horizon*5}min", LON_lag_{time_horizon*5}min"

    Returns:
    --------
    LAT_pred, LON_pred : Series
        Predicted latitude and longitude 30 minutes ahead
    """
    # Calculate displacement over the last 30 minutes
    dLAT = df["LAT"] - df[f"LAT_lag_{time_horizon*5}min"]
    dLON = df["LON"] - df[f"LON_lag_{time_horizon*5}min"]

    # Extrapolate: assume same displacement for next 30 minutes
    LAT_pred = df["LAT"] + dLAT
    LON_pred = df["LON"] + dLON

    return LAT_pred, LON_pred
